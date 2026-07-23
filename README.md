# EEG-based Auditory Attention Decoding in Polyphonic Music

Master's thesis project on decoding selective auditory attention to musical
elements (vocal, drum, bass, others) from consumer-grade EEG (Muse 2) during
naturalistic music listening.

The repository builds on Akama et al. (Sony CSL, 2025):

1. Reproduces their contrastive EEG↔audio baseline (four raw-CNN audio encoders,
   `audio_repr=raw`).
2. Adds an alternative audio representation: a frozen LAION-CLAP backbone + a
   small trainable projection head, selected by `audio_repr=clap`.
3. Adds cross-validation routing (`within` / `leave_subject_out` /
   `leave_song_out`) and test-time negative-control hooks (`none` / `labels` /
   `audio_pair`).

The thesis methodology draft is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
Per-file changes from upstream are in
[`docs/CHANGES_FROM_BASELINE.md`](docs/CHANGES_FROM_BASELINE.md).

## Reference paper

> Akama T., Zhang Z., Nagashima T., Takagi Y., Minamikawa S., Polouliakh N.
> "Decoding Selective Auditory Attention to Musical Elements in Ecologically
> Valid Music Listening". Sony Computer Science Laboratories, 2025.

PDF: [`docs/paper.pdf`](docs/paper.pdf). Original code:
<https://github.com/JURIUENO11/Music_attention> (its README is kept as
[`docs/LEGACY.md`](docs/LEGACY.md)).

## Repository layout

```
.
├── README.md                        This file
├── LICENSE                          CC-BY-SA 4.0 (inherited from upstream)
├── requirements.txt                 Pinned deps (PyTorch 2.2.2, laion_clap, ...)
├── configs/
│   ├── baseline.yaml                Config template + the audio_repr (raw|clap)
│   │                                switch and the test-time audit flags
│   └── tracklist.csv                Per-song stem/track metadata
├── src/
│   ├── run.py                       Interactive launcher (subprocess wrapper)
│   ├── main.py                      Training entry point
│   ├── checkpoint_test.py           Evaluation entry point (writes the breakdown)
│   ├── datasets/                    EEG+music dataset; CV routing
│   │                                (within / leave_song_out / leave_subject_out)
│   ├── models/
│   │   ├── sample_cnn2d_eeg.py      Akama 2D-CNN EEG / raw-audio encoder
│   │   ├── clap_encoder.py          Frozen LAION-CLAP + projection head (extension)
│   │   └── model.py                 Base nn.Module (weight-init helper) the encoders subclass
│   ├── modules/
│   │   ├── contrastive_learning.py  LightningModule: loss, audit hooks, breakdown
│   │   └── clip_loss.py             InfoNCE contrastive loss
│   ├── preprocessing/               EEG / audio transforms
│   └── utils/                       Config loader, paths, logging, helpers
├── scripts/
│   ├── train.sh                     Baseline training (audio_repr=raw)
│   ├── train_clap.sh                CLAP-extension training (audio_repr=clap)
│   ├── test.sh                      Evaluate a checkpoint (within-subject)
│   ├── test_sanity.sh               Negative-control sweep (none|labels|audio_pair)
│   ├── train_cv.sh                  One CV fold: train then test
│   ├── extract_checkpoints.sh       Unpack paper checkpoints from archive/
│   ├── madeeg_setup.sh              Fetch the MAD-EEG dataset (Zenodo)
│   └── reproduce_akama.sh           Reproduce Akama Table 1+2 from checkpoints
├── sweeps/                          Multi-fold experiment drivers (resumable)
│   ├── sweep_common.sh              Shared conda env + fixed protocol flags
│   ├── lso_full_sweep.sh            Leave-song-out, 4 models x 20 songs
│   ├── lso_contrastive_sweep.sh     Leave-song-out, contrastive only
│   ├── lso_controls_sweep.sh        Leave-song-out, audio-only / EEG-only controls
│   ├── lso_spectra_sweep.sh         Leave-song-out, spectral EEG variant (parked)
│   ├── converged_contrastive.sh     A few folds trained to convergence (no 80m cap)
│   ├── recon_lso_sweep.sh           Leave-song-out for the reconstruction decoder
│   └── aggregate_lso.py             Per-model MACRO / per-class summary of results/
├── checkpoints/                     Paper weights (.ckpt gitignored; extract them)
├── results/                         Committed run outputs (TensorBoard, breakdowns)
├── archive/                         Compressed paper checkpoints (.7z)
├── docs/
│   ├── paper.pdf                    Reference paper
│   ├── CHANGES_FROM_BASELINE.md     Per-file diff from upstream
│   ├── METHODOLOGY.md               Thesis methodology draft
│   ├── LEGACY.md                    Original upstream README
│   └── model_architecture.png       Model diagram
├── dataset/                         EEG + per-stem audio (gitignored, ~1 GB; not in repo)
└── logs/                            Runtime logs (gitignored)
```

## Quick start

```bash
# 1. Install deps in a Python 3.9 environment
pip install -r requirements.txt

# 2. Extract the paper checkpoints (one-time)
bash scripts/extract_checkpoints.sh

# 3. Interactive launcher (prompts only for the axes that vary)
cd src && python run.py
```

The launcher wraps the two entry points; the equivalent non-interactive commands
are the scripts:

```bash
bash scripts/test.sh                              # evaluate a checkpoint (within-subject)
bash scripts/train.sh                             # train the baseline (raw) from scratch
bash scripts/train_clap.sh                        # train the CLAP extension
bash scripts/test_sanity.sh                       # negative-control sweep on a checkpoint
bash scripts/train_cv.sh leave_song_out 36 lso1   # one cross-song fold
bash scripts/train_cv.sh leave_subject_out 0 cv1  # one cross-subject fold
```

> **Where the real hyperparameters live.** `configs/baseline.yaml` ships loose
> template defaults (LR 3e-4, batch_size 1, max_epochs 200, normalization
> `none`). These are **not** what is actually run. The run command —
> `learning_rate 0.003`, `batch_size 8`, `max_epochs 1000`,
> `eeg_normalization MetaAI`, `attention_values 4 5`, `key all`, window
> 1280 / stride 256 — is encoded in `scripts/train.sh` / `scripts/test.sh` and in
> the `PROTOCOL` constant of `src/run.py`.

## Hardware notes

- Training requires a CUDA GPU. Tested target: NVIDIA L40S on a SLURM cluster.
- The code auto-falls back to CPU if no GPU is available (e.g. on a login node).
- Tested PyTorch 2.2.2 + CUDA 12.1. The CLAP variant additionally needs
  `laion_clap` (it downloads the HTSAT-tiny checkpoint to `~/.cache/laion_clap/`
  on first use). See `requirements.txt` for the full pinned stack.

## License

CC-BY-SA 4.0 (inherited from the upstream repo). See `LICENSE`.
