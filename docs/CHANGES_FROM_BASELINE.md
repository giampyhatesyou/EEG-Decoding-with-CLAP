# Changes from the Akama baseline

> **Scope: Chapter 1 only** — what this repository changed relative to the upstream
> Akama code. The MAD-EEG drivers (`src/madeeg_*.py`) have no upstream to differ from:
> they are new, and their design decisions live in each file's header.

This repository derives from the Akama et al. (2025) code, kept verbatim under
`Music_attention-main/codes_attention/attention/`. This file lists the
difference between that upstream and the repository as it is now. In source,
each edit relative to upstream is marked `# CHANGED(baseline)`.

## Layout

- `codes_attention/attention/` → `src/`
- `config/` → `configs/`, `runs/` → `results/` (2026-07-26: → `runs/results/`)
- new top-level directories: `scripts/`, `docs/`, `checkpoints/`, `archive/`

## Files added (absent upstream)

- `src/models/clap_encoder.py` — frozen LAION-CLAP backbone + a trainable head
  `Linear(512→256)→GELU→Linear(256→100)`, one instance shared across the 4 stems.
- `src/modules/supervised_classification.py` — `SupervisedClassification`, a separate
  LightningModule for the two unimodal diagnostic controls (#1/#2): a
  cross-entropy 4-class classifier over a single modality (`classify_eeg` /
  `classify_audio`), selected by the `objective` flag. It reuses the baseline encoders
  but defines its own loss and argmax metric, so the contrastive loss
  (`clip_loss.py::compute_task_loss`) and metric (`compute_evaluation_matrix`) are left
  untouched. `classify_audio` is a negative control (expected ~chance; above-chance =
  song→label leakage).
- `src/utils/paths.py` — dataset/log/worker resolution (env > CLI > host profile > YAML).
- `src/run.py` — interactive launcher; subprocess wrapper over `main.py` / `checkpoint_test.py`.
- `scripts/`: `train.sh`, `train_clap.sh`, `train_supervised.sh`, `test.sh`,
  `test_sanity.sh`, `train_cv.sh`, `extract_checkpoints.sh`, `reproduce_akama.sh`.
  (2026-07-26: consolidated into `replicate.sh` + `train.sh`; the sweeps into
  `sweep_song_out.sh` + `sweep_subject_out.sh`. `git show 8dd416b:scripts/` for the old ones.)

## Files removed (present upstream)

- `datasets/dataset.py`, `previous_study_training.py`, `previous_study_test.py`,
  `utils/checkpoint.py` — no call sites in this repo.
- `sequential.sh`, `sequential_test.sh` — replaced by `scripts/`.

## Files identical to upstream (verified by diff)

`models/model.py`, `preprocessing/__init__.py`,
`utils/file_helpers.py`, `utils/time_helper.py`, `utils/yaml_config_hook.py`.

## Changed files (upstream → now)

### `src/main.py`
- config path `/codes_attention/config/config.yaml` → `../configs/baseline.yaml`
- added `sys`/`os` bootstrap: GPU-safe CUDA detection, memlock bump, `sys.path` insert
- added host-aware path/worker resolution via `utils.paths`
- added `audio_repr` switch: `raw` → 4× `SampleCNN2DEEG`; `clap` → 1 shared `CLAPEncoder`
- added `objective` switch: `contrastive` (baseline, builds `EEGContrastiveLearning`
  unchanged) vs `classify_eeg` / `classify_audio` (builds `SupervisedClassification`)
- `cv_mode` / `cv_held_out_id` forwarded to the three `get_dataset` calls
- added TTY-aware progress bar + one-line-per-epoch summary callback
- `gpus` → `devices`; `log_every_n_steps` 1 → 50; `log_dir` configurable

### `src/checkpoint_test.py`
- placeholder `checkpoint_path = "/checkpoint_path"` → repo-relative config + auto-discovery of the latest `best-checkpoint`
- same `sys`/`os` bootstrap and path resolution as `main.py`
- added `cv_mode` / `cv_held_out_id` / `audio_repr` / `shuffle_test_mode` / `test_breakdown` handling
- added the same `objective` switch as `main.py` (must match the objective used at training)
- `load_state_dict(strict=False)` + print of missing/unexpected keys
- removed 2 DataLoaders (train/valid) never iterated in the test driver

### `src/modules/contrastive_learning.py`
- `NT_Xent` import moved from module top to lazy (non-CLIP path); `from . import CLIP_Loss`
- `__init__`: added `test_records`, `_shuffle_test_mode`
- added `self.log("Loss/train", ...)` and `self.log("Valid/loss", ...)` (upstream called neither)
- added per-epoch reset, train/valid summaries, and valid-accuracy logging
- `test_step`: added negative-control shuffles and per-window record collection
- `on_test_end`: empty upstream → computes evaluation, saves figures + CSVs
- `gpus` → `devices`; removed 9 methods with no call site

### `src/modules/clip_loss.py`
- removed 5 `print()` calls (`'No vocal task'`, …, `'No samples with attention_score …'`)
- added a `stats` dict (per-task counts, skipped-task flags, high-attention count) to the returned dict

### `src/datasets/preprocessing_eegmusic_dataset.py`
- path indices: positional `parts[4..8]` → negative `parts[-5..-2]`
- EEG root `eeg` → `eeg_within_sub`; `base_dir` default placeholder → `root`
- added `cv_mode` / `cv_held_out_id`: `within` (= upstream filter), `leave_song_out`, `leave_subject_out`

### `src/datasets/__init__.py`
- `get_dataset()` gained `cv_mode` / `cv_held_out_id` (defaults `within` / `-1`), forwarded to the dataset

### `src/models/sample_cnn2d_eeg.py`
- `from simclr.modules.identity import Identity` removed; `self.fc = Identity()` → `nn.Identity()`

### `src/modules/__init__.py`
- added `from .supervised_classification import SupervisedClassification`

### `src/models/__init__.py`
- added `from .clap_encoder import CLAPEncoder`

### `src/utils/__init__.py`
- removed `from .checkpoint import …`; added `from . import paths`

### `src/utils/logger.py`
- log dir `./log/` (cwd-relative) → `<project-root>/logs/` (computed, auto-created)

### `src/preprocessing/transform.py`
- `playlist_path` `./data/raw/audio/tracklist.csv` → `../configs/tracklist.csv`

### `configs/config.yaml` → `configs/baseline.yaml`
- removed unused keys (`finetuner_*`, `transforms_*`, `spec_aug*`, `save_*`, `projection_dim`, `weight_decay`, …)
- `gpus` → `devices`; `dataset_dir` repo-relative
- added `audio_repr`, `clap_proj_hidden_dim`, `clap_pretrained`, `objective`, `cv_mode`, `cv_held_out_id`, `shuffle_test_mode`, `test_breakdown`
