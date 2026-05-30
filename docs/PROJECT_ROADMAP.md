# Project status and roadmap

A short orientation to the state of this repository and the work still planned.
For the experimental story and every reproduced number see
[`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md); for the thesis-facing write-up see
[`METHODOLOGY.md`](METHODOLOGY.md).

## What this repo is

EEG-based decoding of selective auditory attention to musical elements (vocal /
drum / bass / others) in polyphonic music, recorded with consumer-grade Muse 2
(4 electrodes). Master's thesis project, built on the publicly available code of
Akama et al. (Sony CSL, 2025) — `docs/paper.pdf`.

The Akama 2025 model is the **baseline** we reproduce (86.50% global accuracy on
the within-subject split). The thesis contribution replaces the four raw-CNN
audio encoders with a single frozen LAION-CLAP encoder + a small trainable
projection head, and audits what the within-subject accuracy actually measures.

## Status

| Item | Status |
|------|--------|
| Repo de-nested into a clean `src/` / `configs/` / `scripts/` layout | Done |
| Dependencies pinned (`requirements.txt`) with conflict notes | Done |
| Baseline reproduces the paper (86.50% global, authors' checkpoint) | Done |
| Reference results committed (`results/baseline_paper_all0/`) | Done |
| Negative-control sweep on the baseline (none / labels / audio_pair) | Done |
| CLAP extension trained within-subject (0.9237 global) | Done |
| Cross-validation routing (leave-song-out / leave-subject-out) | Done |
| LOSO (6 folds) and LSO (several songs, both architectures) | Done |
| Per-window evaluation breakdown + figures | Done |

The CLAP, LOSO, and LSO run directories live on the training cluster; only the
reproduced baseline and its negative-control sweep are committed under
`results/` (see `AUDIT_FINDINGS.md` §5 for the full list of run paths).

## Where to start

1. `README.md` — repository layout and quick start.
2. `docs/AUDIT_FINDINGS.md` — the experiments, the numbers, and what they mean.
3. `docs/METHODOLOGY.md` — the methodology draft for the thesis.
4. `results/baseline_paper_all0/RESULTS.md` — the fixed reference numbers.
5. `docs/paper.pdf` — the reference paper.

## Environment

- Python 3.9, PyTorch 2.2.2 + PyTorch Lightning 1.9.5 (see `requirements.txt`).
- Training needs a CUDA GPU (tested on an NVIDIA L40S, SLURM cluster). The code
  falls back to CPU when no GPU is present, but CPU training is impractical.
- The CLAP variant additionally needs `laion_clap`, which downloads its bundled
  HTSAT-tiny checkpoint to `~/.cache/laion_clap/` on first use.

## Refactoring playbook (incremental, only when needed)

Not a single big-bang refactor — each sub-step is done when a concrete need
triggers it.

| Sub-step | Trigger | Effort |
|----------|---------|--------|
| Pluggable encoders (`BaseEEGEncoder`, `BaseAudioEncoder`) | adding a third audio encoder | ~2-3h |
| Pluggable loss | comparing `CLIP_Loss` vs SupCon vs ... | ~1h |
| Split `EEGContrastiveLearning` into model / hooks / evaluator | when evaluation logic grows | ~4h |
| Hydra config system | systematic ablations / sweeps | ~3-4h |
| Type hints + docstrings | continuous, low priority | ~2h cumulative |

## Open research directions

Roughly ordered by expected impact:

1. **Partial fine-tuning of CLAP** (LoRA, or unfreeze the last block) — does
   limited adaptation restore cross-song generalisation? The frozen variant
   does not (see the leave-one-song-out collapse in `AUDIT_FINDINGS.md`).
2. **Improve bass decoding** — the paper's weakest task. Options: bass-specific
   low-frequency filtering, per-instrument loss weighting, a curriculum.
3. **Time-delay sensitivity** — the paper tested 0 and 200 ms; sweep more delays.
4. **Cross-subject generalisation** — domain adaptation / subject normalisation.
5. **The "Others" category** — heterogeneous; consider a multi-label framing.

## Known caveats

- The `projector1` weights in the published checkpoints are an artefact of an
  earlier multi-task (classification + contrastive) setup; the current model
  only has `projector2`. They are loaded with `strict=False`, and the key diff
  is printed so a genuine mismatch is not masked.
- PyTorch Lightning 1.9.5 uses the deprecated `resume_from_checkpoint` and
  `Trainer.from_argparse_args`. Migration to 2.x is non-trivial; deferred.
- One task name in the dataset uses non-ASCII characters; handled in the
  `transform.py` task mapping.
- `contourpy >= 1.3` breaks matplotlib rendering in this stack; pinned to < 1.2.

## Quick commands

```bash
bash scripts/extract_checkpoints.sh   # one-time: unpack the paper checkpoints
bash scripts/test.sh                  # reproduce the baseline (CPU, ~10 min)
bash scripts/train.sh                 # train the baseline (GPU, several hours)
bash scripts/train_clap.sh            # train the CLAP extension (GPU)
bash scripts/test_sanity.sh           # negative-control sweep on a checkpoint
```
