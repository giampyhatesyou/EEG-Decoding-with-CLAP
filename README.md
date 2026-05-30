# EEG-based Auditory Attention Decoding in Polyphonic Music

Master's thesis project on **decoding selective auditory attention to musical
elements** (vocal, drum, bass, others) from consumer-grade EEG (Muse 2) during
naturalistic music listening.

The repository does three things on top of Akama et al. (Sony CSL, 2025):

1. **Reproduces** their contrastive EEG↔audio baseline (within-subject global
   accuracy **0.865**, recovered bit-for-bit from the authors' own checkpoint).
2. **Extends** it by replacing the four raw-CNN audio encoders with a single
   **frozen LAION-CLAP** backbone + a small trainable projection head
   (`audio_repr=clap`), reaching **0.9237** under the same within-subject split.
3. **Audits** what that accuracy means with experiments the paper lists as
   future work — negative controls, leave-one-subject-out, and
   leave-one-song-out — and finds that the bulk of the within-subject accuracy
   reflects **memorisation of the fixed audio geometry of the training songs**
   rather than a cross-song EEG→attention map.

The full story, with every number traced to a run, is in
[`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md). The thesis methodology draft
is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Reference paper

> Akama T., Zhang Z., Nagashima T., Takagi Y., Minamikawa S., Polouliakh N.
> "Decoding Selective Auditory Attention to Musical Elements in Ecologically
> Valid Music Listening". Sony Computer Science Laboratories, 2025.

PDF: [`docs/paper.pdf`](docs/paper.pdf). Original code:
<https://github.com/JURIUENO11/Music_attention> (its README is kept as
[`docs/LEGACY.md`](docs/LEGACY.md)).

## Results at a glance

| Evaluation | Paper | This repo |
|---|---|---|
| Within-subject (global acc) | 0.865 | 0.865 (raw, authors' ckpt) / **0.9237** (CLAP) |
| Cross-subject — leave-one-subject-out | 0.756 (raw, 3 subj) | 0.947 ± 0.024 (CLAP, 6 folds) |
| Cross-song — leave-one-song-out | — (future work) | collapses below chance (see audit) |

The negative-control decomposition of the 0.9237 figure (chance 0.25 +
shared "stem-type" prior 0.13 + song-specific memorisation 0.54) and the
leave-one-song-out collapse are documented in
[`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md). The CLAP, LOSO and LSO run
directories live on the training cluster; only the reproduced baseline and its
negative-control sweep are committed under `results/`.

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
│   ├── run_cv_sweep.sh              Loop train_cv.sh over held-out ids
│   ├── sweep_audit.sbatch           SLURM batch for the LSO/LOSO sweeps
│   └── extract_checkpoints.sh       Unpack paper checkpoints from archive/
├── checkpoints/                     Paper weights (.ckpt gitignored; extract them)
├── results/
│   ├── baseline_paper_all0/         Reproduced baseline 0.865 (tracked, see RESULTS.md)
│   └── baseline_sanity_{none,labels,audio_pair}/   Negative controls on the baseline
├── archive/                         Compressed paper checkpoints (.7z)
├── docs/
│   ├── paper.pdf                    Reference paper
│   ├── AUDIT_FINDINGS.md            Full audit + extension experiments (headline doc)
│   ├── METHODOLOGY.md               Thesis methodology draft
│   ├── LEGACY.md                    Original upstream README
│   ├── PROJECT_ROADMAP.md           Phased plan + iteration playbook
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

# 3. Reproduce the paper baseline (Model: all-0 ms, within-subject)
bash scripts/test.sh

# 4. (GPU) Train the baseline from scratch
bash scripts/train.sh

# 5. (GPU) Train the CLAP extension
bash scripts/train_clap.sh

# 6. Negative-control sanity sweep on a checkpoint
bash scripts/test_sanity.sh

# 7. One cross-validation fold (cross-song or cross-subject)
bash scripts/train_cv.sh leave_song_out 36 lso1
bash scripts/train_cv.sh leave_subject_out 0 cv1
```

After step 3 you should match
[`results/baseline_paper_all0/RESULTS.md`](results/baseline_paper_all0/RESULTS.md)
(86.50% global accuracy on the within-subject test split).

> **Where the real hyperparameters live.** `configs/baseline.yaml` ships loose
> template defaults (LR 3e-4, batch_size 1, max_epochs 200, normalization
> `none`). These are **not** what was actually run. The authentic run command —
> `learning_rate 0.003`, `batch_size 8`, `max_epochs 1000`,
> `eeg_normalization MetaAI`, `attention_values 4 5`, `key all`, window
> 1280 / stride 256 — is encoded in `scripts/train.sh` and `scripts/test.sh`.
> Read the scripts, not the YAML defaults, to know what produced a number.

## Roadmap

See [`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md) for the phased plan and
the playbook of refactoring iterations to use when extending the architecture.

## Hardware notes

- Training requires a CUDA GPU. Tested target: NVIDIA L40S on a SLURM cluster.
- The code auto-falls back to CPU if no GPU is available (e.g. on a login node),
  but CPU training is impractical (~100 days for 1000 epochs).
- Tested PyTorch 2.2.2 + CUDA 12.1. The CLAP variant additionally needs
  `laion_clap` (it downloads the HTSAT-tiny checkpoint to `~/.cache/laion_clap/`
  on first use). See `requirements.txt` for the full pinned stack.

## License

CC-BY-SA 4.0 (inherited from the upstream repo). See `LICENSE`.
