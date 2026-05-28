# Project Roadmap

Document to bring a new chat / agent up to speed on the state of this repository
and the plan ahead.

## What this repo is

EEG-based decoding of selective auditory attention to musical elements (vocal /
drum / bass / others) in polyphonic music, recorded with consumer-grade Muse 2
(4 electrodes). Master's thesis project, derived from the publicly available
code of Akama et al. (Sony CSL, 2025) — `docs/paper.pdf`.

The Akama 2025 model is the **baseline** we reproduce exactly (86.50% global
accuracy). The thesis goal is to **extend** this baseline.

## State as of 2026-05-25

| Item | Status |
|------|--------|
| Repo cloned from paper authors and de-nested into clean structure | Done |
| Dependencies pinned (`requirements.txt`) with conflict notes | Done |
| Baseline checkpoint loads and runs (`model-all0.ckpt`) | Done |
| **Baseline reproduces paper numbers (86.50% global)** | Done (CPU on edu02) |
| Results saved as reference (`results/baseline_paper_all0/RESULTS.md`) | Done |
| Plotting in `on_test_end` produces paper-style figures | Done |
| Auto-discovery of latest checkpoint after training | Done |
| Safe GPU/CPU auto-detection on broken-driver nodes | Done |
| Train on GPU and reproduce ~86% on our own training | ⏸ Blocked: edu02 fabric manager down (reboot scheduled 2026-05-25) |
| Architecture extensions for thesis (CLAP, transformer EEG, ...) | ⏳ Planned |

## Cluster context

- Host: `baldo.disi.unitn.it` (login node, no GPU)
- GPU access: ICE4HPC web portal → opens VSCode-web on `edu02` (L40S x 4)
- User account `foundation.models25` is **not** in any SLURM partition's
  `AllowAccounts` for the standard partitions; only the `ice4hpc` partition
  (edu02-only) is usable, via the web ticket
- **Known issue**: `nvidia-fabricmanager.service` on edu02 has been `failed`
  since 2026-05-04, causing `cuInit` to return `CUDA_ERROR_UNKNOWN (999)`.
  Reboot scheduled 2026-05-25 09:00-16:00 should fix it. Email to ICE4HPC
  support already advised (template in earlier chat).

## How a fresh agent should orient

1. Read `README.md` for layout and quick start.
2. Read this file (`PROJECT_ROADMAP.md`) for current state and next steps.
3. Read `results/baseline_paper_all0/RESULTS.md` for the reference numbers.
4. Read `docs/paper.pdf` (or its text dump if extraction tools are missing) for
   methodological context.
5. SSH config alias `baldo` is already set up on the user's Mac. Use it for any
   cluster-side operation (`ssh baldo "command"`).
6. Conda environment is `eeg_attention`, with Python 3.9 + PyTorch 2.2.2.
   Python binary path: `/home/andrea.giampietro/.conda/envs/eeg_attention/bin/python`.

## Phase plan

### Phase 1 — Cleanup (Done)
Removed legacy MAD-EEG TF code, dropped 2-epoch demo checkpoint, archived
`.7z` originals, fixed `.gitignore`, tracked baseline as reference.

### Phase 2 — Reorganization (Done)
Promoted code out of `Music_attention-main/codes_attention/attention/` nesting
into a clean top-level layout (`src/`, `configs/`, `scripts/`, etc.).
Updated all relative paths in code, scripts, and config.

### Phase 3 — Scientific refactoring (Pending, do incrementally)
Not a single big-bang refactor — each sub-step is done **only when needed**.

| Sub-step | Trigger to do it | Effort |
|----------|------------------|--------|
| 3a — Pluggable encoders (`BaseEEGEncoder`, `BaseAudioEncoder`) | When swapping audio encoder for CLAP | ~2-3h |
| 3b — Pluggable loss | When comparing CLIP_Loss vs SupCon vs ... | ~1h |
| 3c — Split `EEGContrastiveLearning` (400 LOC) into model / hooks / evaluator | When evaluation logic grows (bootstrap, McNemar) | ~4h |
| 3d — Hydra config system | When doing systematic ablations / sweeps | ~3-4h |
| 3e — Type hints + docstrings | Continuous, low priority | ~2h cumulative |

### Phase 4 — Thesis research (Future)
Open directions, ordered roughly by expected impact:

1. **Replace audio encoders with CLAP** — repo name suggests this is the target.
   Hypothesis: pretrained audio embeddings give richer representations than a
   small CNN trained from scratch on a single song dataset.
2. **Improve bass decoding** — paper's weakest task (15-65% accuracy on bass).
   Possible approaches: bass-specific low-frequency filtering, per-instrument
   loss weighting, two-stage curriculum (vocal first, then bass).
3. **Time-delay sensitivity ablation** — paper tested 0 and 200ms. Sweep more
   delays (50, 100, 150, 250, 300ms) and per-task delays.
4. **Cross-subject generalization** — paper achieves 75% cross-subject vs 86%
   within-subject; the gap is significant. Investigate domain adaptation,
   subject normalization, or per-subject heads.
5. **Investigate the "Others" category** — the paper acknowledges it is
   heterogeneous. Could pose it as a multi-label problem or use unsupervised
   audio embeddings to define sub-categories.

## Important constants / paths

- **Project root on cluster**: `/home/andrea.giampietro/EEG-Attention-decoding-with-CLAP`
- **Conda env**: `eeg_attention` (Python 3.9)
- **Dataset path**: `dataset/eeg_within_sub/{train,valid,test}/` for within-subject;
  `dataset/eeg_sub{2,3,7}_*/` for per-subject
- **Default training_date**: `test` (creates `results/test/nmed-CL-preprocessing_eegmusic/version_N/`)
- **Baseline reference**: `results/baseline_paper_all0/RESULTS.md`

## Known caveats

- The `projector1` weights in the published checkpoints are an artifact of an
  earlier multi-task setup (classification + contrastive). The current
  architecture only has `projector2` (embedding head). Loaded with
  `strict=False`.
- PyTorch Lightning 1.9.5 uses deprecated `resume_from_checkpoint` and
  `Trainer.from_argparse_args`. Migration to 2.x is non-trivial; deferred.
- The dataset directory uses Japanese/special chars in one task name
  ("Other(Drums, Bass, Vocal以外の楽器)"). Handled in `transform.py` task mapping.
- `contourpy 1.3.0+` breaks matplotlib rendering in this stack. Pinned to <1.2.

## Quick command reference (from cluster)

```bash
ssh baldo                                    # connect to login node
cd ~/EEG-Attention-decoding-with-CLAP        # project root
bash scripts/extract_checkpoints.sh          # one-time, restore .ckpt files
bash scripts/test.sh                         # reproduce baseline (CPU ~10 min)
bash scripts/train.sh                        # full training (GPU only, ~6h)
tail -f logs/log.txt                         # monitor training (when wired up)
```
