# EEG-based Auditory Attention Decoding in Polyphonic Music

Master's thesis project on **decoding selective auditory attention to musical
elements** (vocal, drum, bass, others) from consumer-grade EEG (Muse 2) during
naturalistic music listening.

Built on top of the proposed model from Akama et al. (Sony CSL, 2025), used here
as a **baseline**. The thesis goal is to extend and improve this baseline.

## Reference paper

> Akama T., Zhang Z., Nagashima T., Takagi Y., Minamikawa S., Polouliakh N.
> "Decoding Selective Auditory Attention to Musical Elements in Ecologically
> Valid Music Listening". Sony Computer Science Laboratories, 2025.

PDF: `docs/paper.pdf`. Original code: <https://github.com/JURIUENO11/Music_attention>.

## Repository layout

```
.
├── README.md                       This file
├── LICENSE                         CC-BY-SA 4.0 (from original repo)
├── requirements.txt                Python deps with pinned versions
├── configs/
│   ├── baseline.yaml               Default training/eval hyperparameters
│   └── tracklist.csv               Track metadata for the dataset
├── src/                            Source code
│   ├── main.py                     Training entry point
│   ├── checkpoint_test.py          Evaluation entry point
│   ├── datasets/                   Dataset loaders (EEG + music)
│   ├── models/                     CNN encoder definitions
│   ├── modules/                    LightningModule, contrastive loss
│   ├── preprocessing/              EEG / audio preprocessing
│   └── utils/                      Logger, config loader, helpers
├── scripts/
│   ├── train.sh                    Run training (full 1000 epochs)
│   ├── test.sh                     Run evaluation on a checkpoint
│   └── extract_checkpoints.sh      Unpack paper checkpoints from archive/
├── checkpoints/                    Paper-published model weights (gitignored)
├── results/                        Training/eval outputs (mostly gitignored)
│   └── baseline_paper_all0/        Reference baseline (tracked, see RESULTS.md)
├── archive/                        Source archives (compressed binaries)
├── dataset/                        EEG + audio dataset (gitignored, ~GB)
├── docs/
│   ├── paper.pdf                   Reference paper
│   ├── LEGACY.md                   Original repo's README
│   ├── model_architecture.png      Model diagram
│   └── PROJECT_ROADMAP.md          Phased plan + iteration playbook
└── logs/                           Runtime debug logs (gitignored)
```

## Quick start

```bash
# 1. Install deps in a Python 3.9 environment
pip install -r requirements.txt

# 2. Extract paper checkpoints (one-time)
bash scripts/extract_checkpoints.sh

# 3. Reproduce the paper baseline (Model: all-0 ms, within-subject)
bash scripts/test.sh

# 4. (Once GPU is available) Train from scratch
bash scripts/train.sh
```

After step 3 you should match the numbers in
[`results/baseline_paper_all0/RESULTS.md`](results/baseline_paper_all0/RESULTS.md)
(86.50% global accuracy on the within-subject test split).

## Roadmap

See [`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md) for the phased plan and
the playbook of refactoring iterations to use when extending the architecture.

## Hardware notes

- Training requires a CUDA GPU. Tested target: NVIDIA L40S / A30 on a SLURM cluster.
- The code auto-falls back to CPU if no GPU is available (e.g. on a login node),
  but CPU training is impractical (~100 days for 1000 epochs).
- Tested PyTorch 2.2.2 + CUDA 12.1. See `requirements.txt` for the full pinned stack.

## License

CC-BY-SA 4.0 (inherited from the upstream repo). See `LICENSE`.
