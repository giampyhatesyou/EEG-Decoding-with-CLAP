# CHANGED(baseline): new file -- not in upstream. SpectraCLIP-style EEG front-end, an
#                    optional alternative to SampleCNN2DEEG selected by `--eeg_repr spectra`.
#                    Mirrors the `--audio_repr` extension: with eeg_repr == "raw" (the
#                    default) the baseline SampleCNN2DEEG is used unchanged, so the Akama
#                    baseline is reproduced bit-for-bit; "spectra" swaps in this dual-branch
#                    encoder. Loss, similarity metric, audio encoders and data splits are
#                    untouched -- only the EEG feature extractor changes.
"""Time + frequency EEG encoder (SpectraCLIP-inspired).

The time branch is the unchanged SampleCNN2DEEG (a 2D-CNN over the raw EEG window).
In parallel, a frequency branch computes per-channel band power in canonical EEG bands
(delta/theta/alpha/beta) with a fixed rFFT and passes it through a small MLP. The two
embeddings are concatenated and projected back to the SAME 100-d width as
SampleCNN2DEEG, so the contrastive InfoNCE loss, the cosine-similarity metric, the four
audio encoders and the train/valid/test splits are all unchanged.

The frequency transform itself is fixed (not learned); only the MLP and fusion heads are
trainable. Band power -- the alpha band in particular -- is a long-standing correlate of
selective auditory attention that a time-domain CNN over the raw signal may not surface.
"""
import torch
import torch.nn as nn

from .model import Model
from .sample_cnn2d_eeg import SampleCNN2DEEG

# Canonical EEG bands (Hz). delta/theta carry stimulus tracking, alpha/beta carry
# attentional state. All below the 128 Hz Nyquist of the 256 Hz EEG.
DEFAULT_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))


class SpectraEEG(Model):
    """Drop-in replacement for SampleCNN2DEEG: same (out_dim, kernal_size) signature
    and the same 100-d output, with an added band-power frequency branch."""

    def __init__(self, out_dim, kernal_size, eeg_sample_rate=256, eeg_channels=4,
                 bands=DEFAULT_BANDS, freq_dim=64):
        super(SpectraEEG, self).__init__()
        # Time branch: the unchanged baseline encoder (outputs 100-d).
        self.time = SampleCNN2DEEG(out_dim, kernal_size)
        self.eeg_sample_rate = eeg_sample_rate
        self.eeg_channels = eeg_channels
        self.bands = bands
        n_feat = eeg_channels * len(bands)
        # Frequency branch: per-channel band power -> MLP.
        self.freq = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, freq_dim),
            nn.ReLU(),
        )
        # Fusion: concat([time(100), freq(freq_dim)]) -> 100-d, same width as the
        # baseline so everything downstream is unchanged.
        self.fuse = nn.Sequential(
            nn.Linear(100 + freq_dim, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100, bias=False),
        )

    def _band_power(self, x):
        # x: (B, C, T) with time on the last axis (the EEG window). -> (B, C * n_bands).
        assert x.dim() == 3 and x.size(1) == self.eeg_channels, (
            f"SpectraEEG expects (B, {self.eeg_channels}, T); got {tuple(x.shape)}")
        spectrum = torch.fft.rfft(x, dim=-1)
        psd = spectrum.real ** 2 + spectrum.imag ** 2            # (B, C, F) power
        freqs = torch.fft.rfftfreq(
            x.size(-1), d=1.0 / self.eeg_sample_rate).to(x.device)
        feats = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)
            feats.append(psd[..., mask].sum(dim=-1))             # (B, C) power in band
        power = torch.stack(feats, dim=-1)                       # (B, C, n_bands)
        power = torch.log1p(power)                               # compress dynamic range
        return power.flatten(start_dim=1)                        # (B, C * n_bands)

    def forward(self, x):
        z_time = self.time(x)                                    # (B, 100)
        z_freq = self.freq(self._band_power(x))                  # (B, freq_dim)
        return self.fuse(torch.cat([z_time, z_freq], dim=1))     # (B, 100)
