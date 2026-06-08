# CHANGED(baseline): NEW FILE — not present in the Akama et al. upstream. This is the CLAP extension.
"""CLAP-based audio encoder for the EEG <-> audio contrastive task.

Drop-in replacement for ``SampleCNN2DEEG`` when ``audio_repr: "clap"`` in the
YAML config. The module wraps a frozen LAION-CLAP audio encoder followed by a
small trainable projection head, so the only learnable audio-side weights are
in that head — keeping the comparison with the Akama baseline as clean as
possible.

Design choices, documented for the thesis methodology section:

* **CLAP is frozen.** ``requires_grad = False`` on every parameter; the module
  is put in ``.eval()`` mode at load time. Only the projection head is updated
  by the optimizer. This makes the comparison "CLAP representation + learned
  projection" vs "SampleCNN2DEEG + learned projection" head-to-head.

* **Resampling 44.1 kHz -> 48 kHz happens inside the model**, via a registered
  ``torchaudio.transforms.Resample`` sub-module. The dataset stays untouched
  (still emits 44.1 kHz audio for the baseline), and ``.to(device)`` moves the
  resampling filter onto the GPU together with the rest of the model.

* **Same padded input as the baseline.** The dataset pads each 3-second clip
  to ``3**11 = 177147`` samples at 44.1 kHz (~4.016 s), which translates to
  ~4.016 s of audio at 48 kHz after resampling — i.e. roughly ~1 s of silence
  on each side of 3 s of music. CLAP was pre-trained on 10–30 s clips, so
  this is mildly out-of-distribution. We accept it for the first comparison
  because changing the dataset would break the baseline; if results suggest
  the padding hurts, a follow-up can strip the silence in a dataset wrapper.

* **One CLAPEncoder shared across the 4 audio slots.** The LAION-CLAP
  checkpoint is ~1.5 GB; instantiating it 4 times would burn memory for
  nothing. ``_LazyCLAP`` ensures the backbone is loaded once per process even
  if the user inadvertently constructs several CLAPEncoders. The caller in
  ``main.py`` is also responsible for passing the *same* CLAPEncoder object
  into all 4 audio encoder slots of ``EEGContrastiveLearning`` — that way
  PyTorch's parameter deduplication leaves the optimizer with exactly one
  copy of the projection head.

Forward contract:
    Input:  tensor of shape ``(B, 1, samples_44k)``, with ``samples_44k`` =
            ``3**11`` from the dataset pipeline.
    Output: tensor of shape ``(B, out_dim)`` matching the baseline head's
            ``projector2`` output width (100 by default).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)


class _LazyCLAP:
    """Class-level cache for the heavy LAION-CLAP backbone.

    Loading the checkpoint allocates ~1.5 GB of memory and several seconds of
    init time; we cache it on the class so that re-instantiating CLAPEncoder
    inside the same process is effectively free.
    """

    _model: Optional[object] = None
    _ckpt_id: Optional[str] = None

    @classmethod
    def get(cls, pretrained: str):
        # Cached value is keyed on the requested checkpoint id so that toggling
        # `clap_pretrained` in the YAML triggers a reload.
        if cls._model is not None and cls._ckpt_id == pretrained:
            return cls._model

        # Lazy import so that "import models" stays cheap when audio_repr != "clap"
        # and laion_clap is not installed in the env (CPU-only dev box).
        import laion_clap  # type: ignore[import-not-found]

        # enable_fusion=False uses the standard single-branch HTSAT encoder.
        # amodel="HTSAT-tiny" is the architecture matching laion_clap's bundled
        # default checkpoint (the one fetched by `load_ckpt()` with no args).
        # The audio_projection still maps to 512-dim, which is what we feed to
        # the trainable head — so this is invisible downstream.
        # For a music-specific checkpoint you can pass `clap_pretrained` to a
        # local .pt file and (if needed for a different ckpt) adjust amodel
        # here, but the default path stays "tiny" so a fresh clone works
        # out-of-the-box.
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")

        # If the user gave us a path to a local .pt file, load that. Otherwise
        # let laion_clap fetch its bundled default into ~/.cache/laion_clap/.
        # We intentionally don't expose checkpoint aliases here: the laion_clap
        # API for ckpt selection has drifted across patch versions; passing
        # an explicit file path is the only stable contract.
        if pretrained and os.path.isfile(pretrained):
            model.load_ckpt(ckpt=pretrained)
        else:
            if pretrained:
                logger.warning(
                    "clap_pretrained=%r is not a readable file path; falling "
                    "back to laion_clap's default checkpoint.",
                    pretrained,
                )
            model.load_ckpt()

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        cls._model = model
        cls._ckpt_id = pretrained
        logger.info("LAION-CLAP loaded (pretrained=%r)", pretrained)
        return model


class CLAPEncoder(nn.Module):
    """Frozen LAION-CLAP backbone + trainable projection head.

    Parameters
    ----------
    out_dim:
        Output width. Must match the EEG encoder's projection width (100 in
        the Akama baseline, set by ``SampleCNN2DEEG.projector2``) so the CLIP
        loss can compare them in the same space.
    hidden_dim:
        Width of the hidden layer in the projection MLP.
    pretrained:
        Either an empty string / unknown alias (-> laion_clap's default
        checkpoint, downloaded to ~/.cache/ on first use) or an absolute path
        to a local ``.pt`` checkpoint.
    """

    INPUT_SAMPLE_RATE = 44100  # mirrors the dataset's audio_sample_rate
    CLAP_SAMPLE_RATE = 48000   # what LAION-CLAP expects on its input
    CLAP_EMBED_DIM = 512       # output width of laion_clap's audio tower

    def __init__(
        self,
        out_dim: int = 100,
        hidden_dim: int = 256,
        pretrained: str = "",
    ):
        super().__init__()

        # Heavy backbone (frozen, deduplicated across instances).
        self.clap = _LazyCLAP.get(pretrained)

        # On-the-fly resampling, registered as a sub-module so .to(device)
        # moves the lowpass filter weights onto the GPU together with us.
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.INPUT_SAMPLE_RATE,
            new_freq=self.CLAP_SAMPLE_RATE,
        )

        # Small trainable projection head. This is the only audio-side weight
        # that gets updated; everything else is fixed.
        self.proj = nn.Sequential(
            nn.Linear(self.CLAP_EMBED_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @torch.no_grad()
    def _clap_embed(self, audio_48k: torch.Tensor) -> torch.Tensor:
        """Run the frozen CLAP backbone. Returns (B, 512), no gradient."""
        # laion_clap's API: get_audio_embedding_from_data(x=..., use_tensor=True)
        # accepts a (B, samples) tensor at 48 kHz and returns (B, 512).
        return self.clap.get_audio_embedding_from_data(x=audio_48k, use_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dataset emits (B, C, samples) per stem. The Akama wavs in this
        # dataset are stereo (C=2), while LAION-CLAP expects a mono signal,
        # so we average the channels into a mono mix before resampling.
        # SampleCNN2DEEG in the baseline implicitly does the same via its
        # learned channel-aggregating conv stack, so the comparison stays
        # valid: both pipelines see "all the audio energy", just mixed
        # differently before being passed to the encoder.
        if x.ndim != 3:
            raise ValueError(
                f"CLAPEncoder expects (B, C, samples), got {tuple(x.shape)}"
            )
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)   # (B, 1, samples), mono mix

        x = x.squeeze(1)               # (B, samples_44k)
        x = self.resampler(x)          # (B, samples_48k)
        emb = self._clap_embed(x)      # (B, 512), detached from CLAP graph
        out = self.proj(emb)           # (B, out_dim), trainable

        return out
