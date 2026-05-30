# Changes from the Akama baseline — provenance guide

This document tracks, file by file, what changed between the upstream code and
this repository, and why. It exists so that every modification relative to
Akama et al. (2025) is explicit and auditable — which matters when the thesis
claims to *extend* their model rather than to reimplement it.

## Three states

For each file three states are distinguished:

- **A — Upstream (Akama).** The original code as published, i.e. the contents of
  `Music_attention-main/codes_attention/attention/` (kept zipped in the repo;
  the upstream README is preserved verbatim as `docs/LEGACY.md`).
- **B — Extension.** This repository as first built: the CLAP extension, the
  cross-validation routing, the negative-control audit hooks, and the
  cluster/Colab portability work.
- **C — Review pass.** A later correctness/cleanup/de-duplication pass:
  documentation fixes, removal of inherited dead code, and a safety net on
  checkpoint loading. No training math, data split, or metric was touched in C.

## Invariants preserved across A → B → C

These are identical in all three states (verified by diff), so the
baseline-vs-CLAP comparison stays on a single axis:

- The contrastive loss arithmetic (`clip_loss.py::compute_task_loss`).
- The evaluation metric (`compute_evaluation_matrix`: global accuracy =
  P[sim(EEG, target) > max sim(EEG, other)], per-task and pairwise variants).
- The EEG encoder (`SampleCNN2DEEG`: conv stack + `projector2`, output width 100).
- The within-subject train/valid/test split (the `cv_mode="within"` path
  reproduces the upstream subset filter exactly).
- Seeds, optimizer (Adam), batch size, temperature, attention filter, gradient
  accumulation, and the sliding-window parameters used by the run scripts.

The only deliberate architectural difference is the audio side, selected by the
single flag `audio_repr ∈ {raw, clap}`.

---

## Structural changes (A → B)

- **De-nesting.** `codes_attention/attention/` → `src/`; `config/` → `configs/`;
  `runs/` → `results/`. New top-level `scripts/`, `docs/`, `checkpoints/`,
  `archive/`.
- **Files dropped from A** (not part of the extension):
  - `datasets/dataset.py` — a generic loader imported nowhere.
  - `previous_study_training.py`, `previous_study_test.py` — MAD-EEG prior-study
    code, out of scope for this thesis.
  - `sequential.sh`, `sequential_test.sh` — replaced by `scripts/`.
- **Files added in B:**
  - `src/models/clap_encoder.py` — the CLAP extension.
  - `src/utils/paths.py` — host-aware path/worker resolution.

---

## Per-file history

### `src/models/clap_encoder.py` — *the extension itself*
- **A:** did not exist.
- **B:** new module. A frozen LAION-CLAP audio backbone (HTSAT-tiny) + a small
  trainable projection head `Linear(512→256)→GELU→Linear(256→100)`. One shared
  instance is used for all four stems; `out_dim=100` matches the EEG
  `projector2` so the CLIP loss compares vectors in the same space. Resampling
  44.1→48 kHz happens inside the module, leaving the dataset (and therefore the
  baseline) untouched.
- **C:** unchanged.
- **Why:** replace the four small per-stem CNNs with a representation
  pre-trained at scale, while changing nothing else, so the gain (if any) is
  attributable to the representation.

### `src/models/sample_cnn2d_eeg.py` — *EEG / raw-audio encoder*
- **A:** `from simclr.modules.identity import Identity` … `self.fc = Identity()`.
- **B:** `self.fc = nn.Identity()` — same operation, drops a hard dependency on
  `simclr` for this class. Conv stack and `projector2` are unchanged.
- **C:** unchanged.
- **Why:** fewer external dependencies; behaviorally identical.

### `src/models/model.py` — *base `nn.Module`*
- **A = B = C:** identical. It is the tiny base class (weight-init helper) that
  `SampleCNN2DEEG` subclasses, so it must stay.
- *Note:* only the README's *description* of this file was corrected in C (it
  was wrongly called the "assembled contrastive model"; the assembly actually
  lives in `EEGContrastiveLearning`).

### `src/modules/clip_loss.py` — *contrastive loss*
- **A:** on a missing task printed `'No vocal task'` / `'No drum task'` / … and
  printed a line when no high-attention samples were present.
- **B:** those `print()`s are replaced by a structured `stats` dict returned
  alongside the losses (per-task counts, skipped-task flags, high-attention
  count). **The loss arithmetic is byte-for-byte identical.**
- **C:** unchanged.
- **Why:** observability that can be aggregated per epoch, with zero effect on
  the gradient.

### `src/modules/contrastive_learning.py` — *the LightningModule*
- **A:** `training_step` / `validation_step` logged only through `debug_logger`
  (they never called `self.log`), so `Valid/loss` was never recorded — meaning
  the `EarlyStopping`/`ModelCheckpoint` callbacks configured on `Valid/loss`
  never actually fired. The validation matrices were accumulated in `__init__`
  and never reset between epochs, and the evaluation computed in
  `validation_step` / `test_step` was discarded. `on_test_end` was empty (it
  printed no metrics and saved no figures). `NT_Xent` was imported at module top.
- **B:**
  - `self.log("Loss/train")`, `self.log("Valid/loss")`, and
    `on_validation_epoch_end` logs `Accuracy/valid_all` and
    `Accuracy/valid_attention`. `on_validation_epoch_start` resets the matrices
    each epoch, so validation accuracy is per-epoch and the EarlyStopping /
    best-checkpoint selection on `Valid/loss` now works as intended.
  - Per-epoch train/val summaries (sample counts, high-attention counts,
    per-task counts, skipped-task counts) for observability.
  - `test_step`: optional negative-control shuffles (`shuffle_test_mode ∈
    {none, labels, audio_pair}`, default `none`) applied before the forward
    pass, plus per-window record collection gated behind `test_breakdown`.
  - `on_test_end`: now computes and prints the evaluation and saves figures +
    breakdown CSVs — using the **same, unchanged** `compute_evaluation_matrix`.
  - Lazy `NT_Xent` import (only on the non-CLIP fallback path); `devices`
    instead of the deprecated `gpus`.
- **C:** removed inherited dead code only (−94 lines, 0 added): `_shared_step`,
  `Kfold_log`, `save_checkpoint`, `load_checkpoint`, `_get_eeg`, `_get_audio`,
  `_get_tensor_value`, and the unused locals in `test_step`
  (`positive_task0..3`, `filtered_task0..3`, the unused averages). No live
  branch was touched.
- **Why (B):** make optimisation observable, make checkpoint selection
  functional, and add the audit tooling — all without changing the loss or the
  metric. **Why (C):** trim methods that were never called from anywhere.
- *Scientific note:* the per-epoch matrix reset + the now-functional
  `Valid/loss` checkpointing is the one change that affects *which* checkpoint a
  fresh training run selects. It is applied identically to the baseline and the
  CLAP variant (same `main.py`), so the comparison stays fair; the committed
  0.865 reproduction uses the authors' own checkpoint and is unaffected.

### `src/checkpoint_test.py` — *evaluation entrypoint*
- **A:** a template with hardcoded placeholders (`checkpoint_path =
  "/checkpoint_path"`); a plain `load_state_dict`; and, because `on_test_end`
  was empty, no printed metrics at all.
- **B:** repo-relative config; auto-discovery of the latest `best-checkpoint`;
  CV / `audio_repr` / `shuffle_test_mode` arguments; the test DataLoader is
  forced to `shuffle=True` only when a negative control is requested (otherwise
  the 13 consecutive windows per trial make an intra-batch shuffle a no-op);
  `load_state_dict(strict=False)` (required because the authors' checkpoints
  carry an extra legacy `projector1` head absent from the current model).
- **C:** the `strict=False` load now captures and prints the `missing` /
  `unexpected` key diff, so a real checkpoint↔architecture mismatch (e.g.
  loading a raw checkpoint into the CLAP topology) surfaces instead of silently
  loading partial weights; two unused `DataLoader` objects (train/valid, never
  iterated in the test driver) were removed.
- **Why:** usable evaluation entrypoint (B) + a guard against silent
  mis-loading (C).

### `src/main.py` — *training entrypoint*
- **A:** config loaded from the absolute path `/codes_attention/config/config.yaml`;
  TensorBoard logs under `runs/`; `gpus`; `log_every_n_steps=1`. The
  `EarlyStopping`/`ModelCheckpoint` callbacks already monitored `Valid/loss`.
- **B:** config from the repo-relative `../configs/baseline.yaml`; configurable
  `log_dir`; host-aware path/worker resolution via `utils/paths`; CV arguments
  forwarded to `get_dataset`; the `audio_repr` switch building either four
  `SampleCNN2DEEG` encoders (baseline) or one shared `CLAPEncoder`; safe
  GPU/CPU auto-detection and a memlock bump for cluster nodes; a TTY-aware
  progress bar plus a one-line-per-epoch summary callback for non-interactive
  logs; `log_every_n_steps=50`.
- **C:** unchanged.
- **Why:** Colab/cluster portability and the CLAP switch, with quieter logs for
  long runs.

### `src/datasets/preprocessing_eegmusic_dataset.py` — *EEG+music dataset*
- **A:** parsed split metadata with absolute positional path indices
  (`r_part[4..8]`, `f.parts[4]`); read EEG from `root/eeg`; `base_dir`
  defaulted to a `"dataset_path"` placeholder.
- **B:** negative path indices (`r_part[-5..-2]`, `f.parts[-2]`) — equivalent to
  the upstream indices for the original layout but independent of where the
  dataset root sits; reads from `root/eeg_within_sub`; `base_dir` defaults to
  `root`; two new constructor args `cv_mode` (`within` / `leave_song_out` /
  `leave_subject_out`) and `cv_held_out_id`. The `within` branch is exactly the
  upstream `if subset != r_subset: continue`.
- **C:** unchanged.
- **Why:** a relocatable dataset and the cross-song / cross-subject experiments
  the paper lists as future work, while the default path reproduces the original
  within-subject split.

### `src/datasets/__init__.py`
- **A → B:** `get_dataset()` gained `cv_mode` / `cv_held_out_id` (defaults
  `"within"` / `-1`) and forwards them to the dataset. **C:** unchanged.

### `src/preprocessing/transform.py`
- **A → B:** `playlist_path` `./data/raw/audio/tracklist.csv` →
  `../configs/tracklist.csv` (the CSV content is identical to upstream).
  **C:** unchanged.

### `src/utils/logger.py`
- **A → B:** log directory `./log/` (current-working-directory relative) →
  `<project-root>/logs/`, auto-created and independent of the working
  directory. **C:** unchanged.

### `src/utils/paths.py` — NEW
- **B:** new helper resolving dataset dir, log dir, and worker count with the
  precedence env vars > CLI > known-host profile > YAML default. **C:** unchanged.

### `src/utils/__init__.py`
- **A → B:** added the `paths` export.
- **C:** removed the re-export of `checkpoint.py` (see below).

### `src/utils/checkpoint.py`
- **A = B:** three CLMR-era loader functions (`load_encoder_checkpoint`,
  `load_finetuner_checkpoint`, `load_model_checkpoint`), never called.
- **C:** deleted (and its re-export removed). They had zero call sites.

### `configs/config.yaml` → `configs/baseline.yaml`
- **A:** 60+ keys, many inherited dead from the CLMR / Mind-Music lineage;
  `gpus`; an absolute `dataset_dir`.
- **B:** removed ~31 unused keys (`finetuner_*`, `transforms_*`, `spec_aug*`,
  `save_*`, `projection_dim`, `weight_decay`, `loss1/2`, `alpha`, `eeg_type`,
  `rmNoisySubject`, …, each verified to have no reference in `src/`); `gpus` →
  `devices`; repo-relative `dataset_dir`; added `audio_repr`,
  `clap_proj_hidden_dim`, `clap_pretrained`, `cv_mode`, `cv_held_out_id`,
  `shuffle_test_mode`, `test_breakdown`. (The values that actually produced the
  reported numbers live in `scripts/`, not in these template defaults.)
- **C:** corrected `clap_pretrained` — the value `"music"` with a comment
  claiming a "music-specific fine-tune" was misleading: that string is not a
  file path, so `clap_encoder.py` falls back to laion_clap's bundled generic
  HTSAT-tiny checkpoint anyway. Set to `""` with a comment stating that the
  generic default is used and that a specific checkpoint requires an absolute
  `.pt` path. (Behaviorally identical: same checkpoint, no spurious warning.)
- **Why (C):** the comment could have led the thesis to claim a music-specific
  CLAP that was never actually loaded.

### `scripts/` — NEW (replace `sequential*.sh`)
- **B:** `train.sh`, `train_clap.sh` (identical to `train.sh` except
  `--audio_repr clap`), `test.sh`, `test_sanity.sh`, `train_cv.sh`,
  `run_cv_sweep.sh`, `sweep_audit.sbatch`, `extract_checkpoints.sh`. They carry
  the real hyperparameters. **C:** unchanged.

### Documentation
- **A:** a single `Readme.md`, preserved verbatim as `docs/LEGACY.md`.
- **B:** `README.md` (overview), `docs/AUDIT_FINDINGS.md` (experiments and
  numbers), `docs/METHODOLOGY.md` (thesis draft), `docs/PROJECT_ROADMAP.md`,
  `results/baseline_paper_all0/RESULTS.md`, `checkpoints/README.md`.
- **C — factual corrections:** subject count 24 → 8 (the dataset and the LOSO
  sweep both have 8 subjects); `model.py` description; `RESULTS.md` "how to
  reproduce" now points at `scripts/test.sh`; an internal cross-reference and
  the repository URL; a caveat that only the baseline runs are committed.
- **C — readability / de-duplication:** `PROJECT_ROADMAP.md` rewritten as a
  plain project-status doc; `METHODOLOGY.md` legend repaired and its marker
  glyphs replaced by plain `[READY]` / `[TODO]` / `[DEFEND]` tags.

---

## What is *not* reproducible from this repository

The committed `results/` contain only the reproduced baseline (0.865) and its
negative-control sweep. The CLAP headline (0.9237), the leave-one-subject-out,
and the leave-one-song-out runs were produced on the training cluster and their
run directories are not committed; their paths are listed in
`docs/AUDIT_FINDINGS.md` §5. The committed baseline negative controls
(`none` 0.865, `labels` 0.225, `audio_pair` 0.3825) are the part of the audit
that can be checked directly here.
