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
│   ├── replicate.sh                 Reproduce every reported number (released ckpts, no GPU)
│   ├── train.sh                     Train one model on one split (MODEL/CV/HELD)
│   ├── setup_checkpoints.sh         Unpack paper checkpoints from archive/
│   └── madeeg_setup.sh              Fetch the MAD-EEG dataset (Zenodo)
├── sweeps/
│   ├── sweep_common.sh              Conda env, the fixed protocol, resume logic
│   ├── sweep_song_out.sh            Leave-one-song-out, 4 models x 20 songs (resumable)
│   ├── sweep_subject_out.sh         Leave-one-subject-out, 4 models x 8 subjects (resumable)
│   └── report.py                    Renders RESULTS.md from results_manifest.tsv
├── results_manifest.tsv             THE pinned list of folds behind every reported number
├── RESULTS.md                       Generated: the one results table
├── checkpoints/                     Paper weights (.ckpt gitignored; unpack them)
├── archive/                         Compressed paper checkpoints (.7z)
├── docs/
│   ├── paper.pdf                    Reference paper
│   ├── CHANGES_FROM_BASELINE.md     Per-file diff from upstream
│   ├── METHODOLOGY.md               Thesis methodology draft
│   ├── LEGACY.md                    Original upstream README
│   └── model_architecture.png       Model diagram
├── dataset/                         EEG + per-stem audio
└── runs/                            Everything a run emits (gitignored)
    ├── results/                     Per-run CSVs, hparams, checkpoints, TensorBoard
    └── logs/                        stdout of the launcher scripts
```

## Quick start

```bash
# 1. Install deps in a Python 3.9 environment
pip install -r requirements.txt

# 2. Unpack the paper checkpoints (one-time)
bash scripts/setup_checkpoints.sh

# 3. Reproduce every number this project reports (~45 min, CPU, no training)
python src/run.py replicate
```

## The five commands

That is the whole surface. Anything else is a one-off and belongs in a scratch
shell, not in the repo.

```bash
python src/run.py                      # interactive launcher (prompts only for what varies)
python src/run.py replicate            # reproduce every reported number from released ckpts
python src/run.py train --model clap   # retrain: --model baseline|clap|audio_only|eeg_only
python src/run.py sweep --cv song      # resumable leave-one-song-out sweep (all 4 models)
python src/run.py sweep --cv subject   # resumable leave-one-subject-out sweep
python src/run.py report               # re-render RESULTS.md from the manifest
```

Each maps to a script you can also call directly, e.g.
`MODEL=clap CV=song HELD=44 bash scripts/train.sh` or
`CUDA_VISIBLE_DEVICES=0 CAP=80m bash sweeps/sweep_song_out.sh`.

## One result, pinned

`results_manifest.tsv` lists every fold that contributes to a reported number —
model, evaluation, held-out id, run directory, and the code vintage that produced
it. `sweeps/report.py` reads **only** that file and renders `RESULTS.md`.

Nothing is discovered by scanning the disk, so a number cannot change because a
new run directory appeared: to include a fold, pin it. `report.py` also checks
each pin against the run's own `hparams.yaml` and refuses to write the table if
one disagrees. (This replaced an aggregator that globbed and de-duplicated by a
partial key — two runs of the same fold collided and whichever the filesystem
returned last silently won, which moved a reported number from 0.142 to 0.154
with no new evidence behind it.)

> **Where the real hyperparameters live.** `configs/baseline.yaml` ships loose
> template defaults (LR 3e-4, batch_size 1, max_epochs 200, normalization
> `none`). These are **not** what is actually run. The real protocol —
> `learning_rate 0.003`, `batch_size 8`, `max_epochs 1000`,
> `eeg_normalization MetaAI`, `attention_values 4 5`, `key all`, window
> 1280 / stride 256 — lives in `$PROTO` (`sweeps/sweep_common.sh`) for the shell
> side and in `PROTOCOL` (`src/run.py`) for the launcher.
> `python src/run.py --selftest` asserts the two still agree, flag by flag.

## Hardware notes

- Training requires a CUDA GPU. Tested target: NVIDIA L40S on a SLURM cluster.
- The code auto-falls back to CPU if no GPU is available (e.g. on a login node).
- Tested PyTorch 2.2.2 + CUDA 12.1. The CLAP variant additionally needs
  `laion_clap` (it downloads the HTSAT-tiny checkpoint to `~/.cache/laion_clap/`
  on first use). See `requirements.txt` for the full pinned stack.

## License

CC-BY-SA 4.0 (inherited from the upstream repo). See `LICENSE`.
