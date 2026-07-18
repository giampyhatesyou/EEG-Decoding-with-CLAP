# Refactor Audit Report

This report presents a code audit of the recent refactoring changes in the master's-thesis research repository. The audit was conducted in accordance with the **PONYTAIL philosophy** (prioritizing code deletion, simplicity, and native stdlib/dependencies over custom solutions) and the **scientific safety constraints** of the project constitution ([CLAUDE.md](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/CLAUDE.md)).

---

## 1. Verdict Summary Table

The table below summarizes the audit results per changed file.

| Changed File | Provenance OK? | Behavior-Preserving? | Leaner? | Issues |
| :--- | :---: | :---: | :---: | :--- |
| [paths.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/utils/paths.py) | **Yes** | **Yes** | **Yes** | None |
| [run.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/run.py) | **Yes** | **Yes** | **Yes** | None |
| [contrastive_learning.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/modules/contrastive_learning.py) | **Yes** | **Yes** | **Yes** | None |
| [main.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/main.py) | **Yes** | **Yes** | **Yes** | None |
| [checkpoint_test.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/checkpoint_test.py) | **Yes** | **Yes** | **Yes** | None |
| [preprocessing_eegmusic_dataset.py](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/datasets/preprocessing_eegmusic_dataset.py) | **Yes** | **Yes** | **Yes** | None |
| [sweep_common.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/sweep_common.sh) | **Yes** | **Yes** | **Yes** | None |
| [lso_contrastive_sweep.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_contrastive_sweep.sh) | **Yes** | **Yes** | **Yes** | 1 Major (Source order bug) |
| [lso_controls_sweep.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_controls_sweep.sh) | **Yes** | **Yes** | **Yes** | 1 Major (Source order bug) |
| [lso_full_sweep.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_full_sweep.sh) | **Yes** | **No** (fails to run folds) | **Yes** | 1 Blocker (Working dir), 1 Major (Mac default bash compatibility) |
| [lso_spectra_sweep.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_spectra_sweep.sh) | **Yes** | **Yes** | **Yes** | 1 Major (Source order bug) |
| [converged_contrastive.sh](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/converged_contrastive.sh) | **Yes** | **Yes** | **Yes** | 1 Major (Source order bug) |

---

## 2. Findings List

### Finding 1: Broken Execution in Leave-Song-Out Full Sweep
* **File & Line**: [lso_full_sweep.sh:10](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_full_sweep.sh#L10) (and lines 54, 58)
* **Severity**: **Blocker**
* **Claim**: The script fails to run any training or testing folds because it executes commands from the repository root rather than the `src/` directory.
* **Evidence**:
  The script performs `cd "$(dirname "$0")"` which targets the repository root. It then calls `timeout $CAP $PY -u main.py` and `$PY -u checkpoint_test.py`. Both of these files reside under `src/`, so Python cannot find them. This failure was confirmed by inspecting the cluster's log file `/home/andrea.giampietro/EEG-Attention-decoding-with-CLAP/sweep_full.log` on `baldo`, where every fold exits immediately with `python: can't open file 'main.py': [Errno 2] No such file or directory`.
* **Recommendation**:
  1. Change `cd "$(dirname "$0")"` to `cd "$(dirname "$0")/src"`.
  2. Prefix the `results/` folder path with `../` in the Python done-set logic (lines 38, 103) and the checkpoint existence check (line 56), so that they resolve to `../results/` relative to `src/`.

### Finding 2: Relational Path Sourcing Bug in Sweep Scripts
* **File & Line**: [lso_contrastive_sweep.sh:7-9](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_contrastive_sweep.sh#L7-L9) (and matching lines in [lso_controls_sweep.sh:8-10](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_controls_sweep.sh#L8-L10), [lso_spectra_sweep.sh:9-11](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_spectra_sweep.sh#L9-L11), [converged_contrastive.sh:7-9](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/converged_contrastive.sh#L7-L9))
* **Severity**: **Major**
* **Claim**: Sourcing `sweep_common.sh` using a relative path fails when running the script from outside the root directory or when using simple relative paths.
* **Evidence**:
  The scripts perform:
  ```bash
  cd "$(dirname "$0")/src"
  source "$(dirname "$0")/sweep_common.sh"
  ```
  If run as `./lso_contrastive_sweep.sh`, `dirname "$0"` is `.`. The first command changes the working directory to `src/`. The second command expands to `source "./sweep_common.sh"`, which searches for `sweep_common.sh` inside `src/`. Since it does not exist there, the script crashes with `No such file or directory` and subsequent `unbound variable` errors under `set -u`.
* **Recommendation**:
  Source `sweep_common.sh` *before* changing the directory, or resolve the script's directory cleanly. Sourcing before changing directory works perfectly:
  ```bash
  source "$(dirname "$0")/sweep_common.sh"
  cd "$(dirname "$0")/src"
  ```

### Finding 3: Incompatibility with Default macOS Bash
* **File & Line**: [lso_full_sweep.sh:65](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_full_sweep.sh#L65) (and matching lines in [lso_contrastive_sweep.sh:59](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_contrastive_sweep.sh#L59), [lso_controls_sweep.sh:61](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/lso_controls_sweep.sh#L61))
* **Severity**: **Major**
* **Claim**: The use of associative arrays (`declare -A`) prevents these scripts from running on macOS platforms.
* **Evidence**:
  The default macOS shell is Bash 3.2.57, which does not support associative arrays (introduced in Bash 4.0). Running the scripts on macOS default bash crashes immediately with a declaration error and `vocal: unbound variable` under `set -u`.
* **Recommendation**:
  Replace the associative array lookup with a standard shell function and a `case` statement, which is compatible across Bash 3.x and 4.x/5.x:
  ```bash
  get_songs() {
    case "$1" in
      vocal) echo "2 3 5 33 45" ;;
      drum)  echo "7 18 59 112 145" ;;
      bass)  echo "44 50 55 68 140" ;;
      others) echo "8 16 43 62 120" ;;
    esac
  }
  ```
  And extract song values with `arr=($(get_songs "$cls"))`.

### Finding 4: Fragile Hand-Rolled YAML Parsing
* **File & Line**: [run.py:401-411](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/run.py#L401-L411)
* **Severity**: **Nit**
* **Claim**: Parsing `hparams.yaml` line-by-line using basic string starts-with matching is redundant and fragile.
* **Evidence**:
  The script implements `_run_config` which reads `hparams.yaml` line-by-line and splits on `:`. This is susceptible to failures on nested fields, spacing changes, or comments. PyYAML (`yaml`) is already an installed dependency of the project (used in `yaml_config_hook.py` and PyTorch Lightning).
* **Recommendation**:
  Leverage PyYAML's `safe_load` to parse YAML configurations robustly:
  ```python
  import yaml
  with open(hp, "r") as f:
      data = yaml.safe_load(f) or {}
  for k in cfg:
      if k in data:
          cfg[k] = str(data[k])
  ```

### Finding 5: Duplicate and Dead Imports in Contrastive Module
* **File & Line**: [contrastive_learning.py:434-438](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/modules/contrastive_learning.py#L434-L438)
* **Severity**: **Nit**
* **Claim**: Standard modules `os` and `numpy` (as `np`) are imported inline inside `on_test_end`, despite already being imported at the top of the file.
* **Evidence**:
  The `on_test_end` method has inline imports for `os` and `np`. These packages are already imported at the top of the module on lines 11 and 12.
* **Recommendation**:
  Remove the duplicate imports from `on_test_end`. Move `matplotlib` and `plt` imports to the top of the file to centralize all imports.

---

## 3. Ponytail Ledger (Ranked Cleanup Opportunities)

Ranked list of remaining over-engineering and dead code that could be deleted or simplified.

### 1. Consolidate Done-Set Parsing in Sweep Scripts
* **Est. Lines Saved**: ~80 lines across 5 scripts.
* **Rationale**: The ~20-line inline Python snippet that parses `hparams.yaml` to detect completed folds is copy-pasted in all 5 sweep scripts. Since `src/run.py` already implements a `runs_with_checkpoint` method that does exactly this, we can replace the duplicated python blocks with a simple one-liner calling `run.py` or consolidate them into a single utility script (e.g. `src/utils/check_completed.py`).
* **Rung**: 2 (Already provided elsewhere in the repo).

### 2. Standardize YAML Parsing in Sweep Done-Set Logic
* **Est. Lines Saved**: ~15 lines.
* **Rationale**: The inline Python scripts in the sweeps parse `hparams.yaml` line-by-line using custom string matching. This duplication should be replaced by a clean call to PyYAML (`yaml.safe_load`), which is already a core dependency.
* **Rung**: 4 (Use native/installed dependency instead of hand-rolling).

### 3. Centralize Hostname Logging
* **Est. Lines Saved**: ~5 lines.
* **Rationale**: `socket` and `socket.gethostname()` are only imported in `paths.py` to get the hostname for diagnostics. If hostname is logged primarily through other system outputs or logs, it can be deleted, or at least simplified.
* **Rung**: 3 (Use standard library, minimize wrappers).

---

## 4. Provenance Audit

* **Baseline Integrity**: **100% OK**
* **Verification**: All student-added or changed lines in the audited refactor are correctly enclosed inside `# CHANGED(baseline): start ... end` blocks or marked with a single-line `# CHANGED(baseline):` comment.
* **Unaltered Baseline**: No unmarked/baseline lines were altered without documentation or provenance markers. The restructuring and cleanup strictly adhered to the constitution.

---

## 5. SSH / Output Logging Observations

Logging in to the DISI cluster (`baldo`) was successful. The run log outputs under the repo root (`sweep_full.log`, `controls_sweep.log`, and `contrastive_sweep.log`) were analyzed. The following telemetry noise could be quieted to comply with the logging policy:

1. **Quiet PyTorch Lightning Model Summary during Sweeps**:
   PyTorch Lightning prints a full model parameter summary table at the start of every training run. When running a sweep of 30+ folds, this table is repeated 30+ times, producing hundreds of lines of identical model architecture output in the sweep log. Passing `enable_model_summary=False` to the `Trainer` in `main.py` when running in a non-TTY environment would make sweep logs much cleaner and easier to scan.
2. **Quiet PyTorch Lightning Bootstrapping Messages**:
   Standard messages like `GPU available: True (cuda), used: True` and TPU/IPU/HPU warnings are printed on every single fold. Setting the PyTorch Lightning logger level to `WARNING` in `main.py` would silence these repeat messages.
3. **Mute CUDA Tensor Core Warning**:
   The warning regarding CUDA Tensor Cores (`You are using a CUDA device ('NVIDIA L40S') that has Tensor Cores... set torch.set_float32_matmul_precision...`) is printed on every run. Setting `torch.set_float32_matmul_precision('medium')` at the beginning of `main.py` and `checkpoint_test.py` would resolve this and eliminate the warning noise.
4. **Silence pkg_resources Deprecation Warning**:
   The `pkg_resources is deprecated` UserWarning from `setuptools` is printed on every Python startup. Adding a filter warning block (`warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")`) would quiet this down.

---

## 6. Out-of-Scope Suggestions

These suggestions are outside the scope of the refactor but represent significant optimization opportunities:

1. **CPU Bottle-neck in EEG Normalization**:
   In [preprocessing_eegmusic_dataset.py:260-268](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/src/datasets/preprocessing_eegmusic_dataset.py#L260-L268), `normalize_EEG_4` fits a CPU-bound scikit-learn `RobustScaler()` object *dynamically* for every single channel of every EEG window during `__getitem__`. Instantiating, fitting, and transforming via scikit-learn on every step is highly CPU-bound and creates a major training speed bottleneck.
   *Recommendation*: Re-implement `RobustScaler` natively in PyTorch using `torch.median` and `torch.quantile` to run operations efficiently on tensor inputs, or pre-compute scaling factors offline to eliminate the fitting step during training.
2. **Minor out-of-scope script change**:
   A minor table header cleanup was observed in [aggregate_lso.py:61](file:///Users/andrea/Library/Mobile%20Documents/com~apple~CloudDocs/Tesi/akami/aggregate_lso.py#L61) (`"folds"` to `"# of runs"`). This is behavior-preserving and is safe to merge.
