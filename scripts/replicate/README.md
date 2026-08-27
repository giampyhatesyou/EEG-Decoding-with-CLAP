# `scripts/replicate/` — one script per experiment

One script for every experiment in the project, carrying **the exact commands**
taken from the provenance files and from the pre-registered criteria. Each script
states in its header: what it measures in one sentence, how long it takes, whether
it spends held-out looks, and **the expected reference numbers** — so whoever runs
it knows immediately whether it replicated.

The table below is navigation only. **The detail lives in each script's header**,
which is the source, not a summary of one.

Everything here can also be driven from the launcher, which adds a per-run log
file under `runs/logs/`:

```bash
python src/run.py exp --list          # this table, generated from the headers
python src/run.py exp exp13           # one experiment (a unique prefix is enough)
python src/run.py exp cheap           # the four that spend no held-out looks
python src/run.py exp all             # all 26, in order, canaries first
python src/run.py canaries --levels 0 # the regression gates on their own
```

---

## The three environment traps that make everything fail

These were paid for, not hypothetical.

1. **The interpreter: `/opt/miniconda3/bin/python`, never bare `python`.**
   Anaconda base (`/opt/anaconda3/bin/python`) gives the same results with floats
   differing at 2e-5, so **the md5 canaries fail** — it really happened, at the
   start of Exp. 7. The conda env `attention` has no `h5py` and cannot even run
   `madeeg_reconstruction.py`.
   Every script here uses `PY=${PY:-/opt/miniconda3/bin/python}`.
   **Declared exception:** `exp18`/`exp19_matchmismatch.sh` use `/opt/anaconda3`
   because they need `torch` — and there is no md5 gate there to match.
   Scripts that cross the Chapter 1 canary have a **second** interpreter,
   `$PY_TORCH`, which looks for the conda env `attention` (torch 2.2.2, numpy<2)
   and falls back to `/opt/anaconda3` only as a last resort.
2. **`--spatial mono` requires `--train_on raw_solos --test_eeg raw`** (asserted in
   the code: mono duos do not exist in the preprocessed release). It is the trap
   that made Exp. 9's secondaries S1/S2 **unexecutable** — declared as such, not
   substituted.
3. **`--eeg_clean` only applies with `--train_on raw_solos`** (asserted: the
   preprocessed release was already cleaned by the dataset authors).

Two more cost time rather than correctness: **16 GB of RAM** — the `--target_fs 256`
runs (Exp. 7 C5/C7, arm A, AXIS 0) are ~6 GB of design matrix each; and the shell
matters — in `zsh` a `$VAR` holding several flags does not word-split, which is why
this directory contains scripts and bash arrays rather than one-liners with
variables.

## Common variables (sensible defaults, overridable from the environment)

| variable | default | what it does |
|---|---|---|
| `PY` | `/opt/miniconda3/bin/python` | the interpreter. Do not change it without re-reading trap 1 |
| `PY_TORCH` | *searched* | the env with `torch` for the `clip_loss.py` canary. Empty = search; if not found the script **declares it did not cross it**, it does not pretend |
| `MADEEG_DIR` | `$HOME/madeeg` | the MAD-EEG dataset, **not in the repo** (4.7 GB) |
| `OUT_DIR` | `<repo>/runs/replicate` | where the replication's provenance and comparisons land |
| `TAG` | `repl_` | prefix for `--training_date`: **pinned records are never overwritten** (method rule 9) |
| `ARCHIVE` | `<repo>/../_baldo_archive_2026-07-18/results` | the step C/D records, which live outside the repo |
| `LEVELS` | `0 1 2` | `canaries.sh` only: which canary levels to cross |

---

## The navigation table

`null` = the value the number is judged against. `threshold` = the criterion
**written before** the run. `CPU?` = runs on CPU · **+MAD** needs the dataset ·
**+CKPT** needs the paper checkpoints · **GPU** is a step this directory prints
rather than launches.

### Chapter 1 — the Akama arm

| experiment | script | null | threshold | verdict | provenance | canary | CPU? |
|---|---|---|---|---|---|---|---|
| **within-split** + negative controls | `cap1_within_split.sh` | 0.25 (4 fixed slots, assumed from the design) | none — it is a reproduction, the paper fixed the criterion (0.865) | reproduced: MACRO 0.875 / GLOBAL **0.8650 exact**; `labels` 0.225, `audio_pair` 0.383 (**not** bit-reproducible) | `RESULTS.md` within and control sections ← `results_manifest.tsv` | `clip_loss.py` · `run.py --selftest` · `report.py` verifies every pin | yes, +CKPT · the audio_only/eeg_only/clap rows need a GPU |
| **leave-song-out** | `cap1_leave_song_out.sh` | 0.25 | none (reproduction) | closed 20/20 folds — **0.268**, i.e. at chance. eeg_only 0.247 · audio_only_raw 0.181 · contrastive_raw 0.142 | `RESULTS.md` song_out ← manifest (80 pins) | `clip_loss.py` · `run.py --selftest` · pin ↔ `hparams.yaml` | yes (re-reads the pins) · redoing the folds needs a GPU |
| **leave-subject-out** | `cap1_leave_subject_out.sh` | 0.25 | none (reproduction) | 0.943 (6 clap folds) · 0.781 (3 raw folds). Akama Table 2: sub3 0.6642 · sub7 0.8750 · sub2 0.8664 | `RESULTS.md` subject_out ← manifest · `scripts/replicate.sh` header | as above | yes, +CKPT · redoing the folds needs a GPU |

### Chapter 2 — MAD-EEG, the linear family

| experiment | script | null | threshold | verdict | provenance | canary | CPU? |
|---|---|---|---|---|---|---|---|
| **ridge anchors** (A1/A2/A3) | `ridge_anchors.sh` | 0.500 on the duo, by construction | **88/154 = 0.5714** | none crosses it: A1 **0.5584** (project maximum, p 0.0853) · A2 0.5065 · A3 0.4805 | `RESULTS.md` madeeg_duo ← manifest, pins `madeeg_ridge_{duo,solos}_repro2026-08` · A2 in the run archive | md5 of the two pinned records + the ridge `--self_test` (threshold 0.90 in the code) | yes, +MAD |
| **the four axes + AXIS 0** | `axes_paper_protocol.sh` | none — `inner_val_r` is reconstruction, not a decision | none: it is configuration, not a test | one axis gains: **ICA +0.0091 (+16%)**. AXIS 2 (0.1100) is a **metric artefact**: fair test 36/70, p 0.29 | runs/results/ax_* · `madeeg_repro_hparams` | every axis defaults to the pre-existing behaviour and is **asserted**; the three anchors stay bit-identical (→ `ridge_anchors.sh`) | yes, +MAD, **16 GB RAM** |
| **Exp. 4** — mono duos (the only confirmatory look) | `exp04_duo_mono.sh` | 0.500 by construction | **86/150 = 0.5733** (recomputed before the run) | **does not replicate** — 79/150 = 0.5267, p 0.2839, power 0.534 | `RESULTS.md` madeeg_duo, pin `madeeg_exp4_mono_confirm` | `--check_alignment --spatial mono` (r=1.0000 over 272 comparisons) **and** the same on stereo, which **must fail** · md5 of the pinned record | yes, +MAD (raw release) |
| **Exp. 5** — side from band power | `exp05_side_bandpower.sh` | 0.5 theoretical, **0.5034 operational** (75/149, constant guesser) | **86/149 = 0.5772** | **archived without a number.** No file implements it; three thresholds survive (30/47 · 27/42 · 53/89), reused by Exp. 11/12 and 6[H] | **none, and none should exist** | n/a | yes (executes nothing) |
| **Exp. 6 [G]** — own-vs-other, 4 configurations | `exp06_ovo_gate.sh` | **0.500 exact by symmetry** (188 pairs × 2 directions) | McNemar: 80/139 base · **76/131** with ICA | **NO-GO for CCA**: 66/131 vs 65, p 0.5000. "CCA adds nothing", not "CCA loses" | runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/ | **md5 `2eaa9262…` + 208/376** rerun first · `--mcnemar` verifies 376/376 shared comparisons | yes, +MAD (raw release) |
| **Exp. 6 [H]** — paired lateralised alpha | `exp06_alpha_paired.sh` | **0.500 exact by exchangeability** | **28/44 = 0.6364** primary · **27/42** mono control | below threshold: 23/44 (p 0.4402) · mono 24/42, **higher than the primary**. Power 0.371 | runs/results/alpha_real_{stereo,mono}/ · `alpha_ctrl2_a*_inject/` | lateralised injection **44/44 in the run's exact configuration** (threshold 0.90 in the code), crossed before the real number | yes, +MAD (raw release) |
| Exp. 7 — sweep over 8 configurations | `exp07_ovo_config_sweep.sh` | 0.500 exact | Bonferroni McNemar α = 0.05/8 = 0.00625 | none passes. REF **208/376 = 0.5532** · max C3 213/376 | `docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt` · runs/results/ovo_sweep_* | md5 `2eaa9262…` | yes, +MAD |
| Exp. 8 — spectral flux on own-vs-other | `exp08_flux_ovo.sh` | 0.500 exact | α = 0.025 | **positive**: F1 **243/376** (p 0.0015) · F2 238/376 | `docs/provenance/2026-08-11_exp8_flux_mcnemar.txt` | md5 `2eaa9262…` | yes, +MAD |
| Exp. 9 — flux on **attention** | `exp09_flux_attention.sh` | 0.500 on the duo | absolute bar **90/154** at α 0.025 | the transfer fails: 76/154 and 77/154 | `docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt` | mel anchors **74/154 and 86/154 decision-identical** (zero discordant) | yes, +MAD |
| Exp. 11 — spectral register, stereo duos | `exp11_spectral_register_stereo.sh` | 0.500 exact by symmetry | **30/47** (written 2026-08-09) | 24/47 = 0.5106 | runs/results/exp11_* · `docs/provenance/2026-08-17_exp11_*_RESULT_*` | positive control **47/47**, hard exit inside the driver | yes, +MAD |
| Exp. 12 — the same on mono duos | `exp12_spectral_register_mono.sh` | 0.500 | **27/42** mono · **53/89** pooled | 22/42 · 46/89 | runs/results/exp12_* · `docs/provenance/2026-08-17_exp12_*_RESULT_*` | positive control **42/42**, hard exit | yes, +MAD (raw release too) |
| arm A — protocol parity with the paper | `armA_paper_protocol.sh` | 0.500 on the duo | none: it is a protocol probe | F1 **0.5267 / 0.4800 / 0.5400** at n=150 — the paper's 79 **does not reproduce** | runs/results/armA_paper_* · `docs/provenance/2026-08-17_armA_paper_*_RESULT_*` | none declared (the reproduction of the three F1 values is the control) | yes, +MAD, **16 GB RAM** |
| arm D — per-window leakage audit | `armD_leakage_audit.sh` | **0.5136 by construction** on windows, **0.500 exact** on trials — the null pairs with the unit of the accuracy next to it | interpretation rule written in the code **before** the run: leakage shown if pseudo window-CV > 0.60 **and** pseudo trial-CV in [0.40, 0.60] | leakage demonstrated: pseudo window-CV **0.6209** vs trial-CV **0.4847** (true labels 0.3803 vs 0.0895, null 0.2277) | runs/results/armD_leakage_audit/ · `docs/provenance/2026-08-17_armD_leakage_audit_RESULT_*` | the pseudo-label arm **is** the audit's positive control; v1 kept as `*_v1_unbalanced*` | yes, +MAD |

### Chapter 2 — separability and front ends

| experiment | script | null | threshold | verdict | provenance | canary | CPU? |
|---|---|---|---|---|---|---|---|
| Exp. 13 — audio-only separability | `exp13_stem_separability.sh` | cross-song floor (0.0256 for C3) | two pre-registered criteria | **GO**, the only candidate passing both: C3 MFCC-13 (0.0533 vs 0.0256, 32/36) | `docs/provenance/2026-08-12_exp13_stem_separability.txt` | mel-8 **0.1772/0.1442** · flux **0.2874/0.2417** · 30/36 (±0.0010), **hard exit** | yes, +MAD (~10 s measured) |
| Exp. 14 — MFCC tracked by the EEG | `exp14_mfcc_tracking.sh` | 0.500 exact | bar **≥ 208/376** | both track: T1 214/376 · T2 212/376 | `docs/provenance/2026-08-12_exp14_mfcc_tracking.txt` | md5 `2eaa9262…` · identity with C3 at **0.000e+00** · two synthetic positives 1.0000 | yes, +MAD |
| Exp. 15 — attentional differential | `exp15_mfcc_differential.sh` | D = 0 | **MDD 0.2467** | not conclusive for power: D(mel) +0.1001 vs D(MFCC) −0.1119, p 0.9833 | `docs/provenance/2026-08-12_exp15_mfcc_differential.txt` | decision-identical anchors **and** underlying correlations at 0.000e+00 | yes, +MAD |
| Exp. 16A — CLAP in the gate (criterion retracted) | `exp16a_clap_separability.sh` | — | **criterion retracted**, falsified by 512-dim noise | on record. **Do not rerun it to judge CLAP** — use Exp. 17. It still runs the negative control (0.922) | `docs/provenance/2026-08-12_exp16A_{clap_separability,negative_control}.txt` | md5 `f3ebe89b…` | yes (stage 2) · stage 1 needs the checkpoint |
| Exp. 16B — `band_power` as a CCA view | `exp16b_ccaviews.sh` | 0.500 exact | bar **≥ 208/376** | 202/376 = 0.5372, bar not crossed | `docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt` | md5 `2eaa9262…` · single-view CCA re-measured 209/376 with 0 discordant | yes, +MAD |
| Exp. 17 — CLAP with the invariant criterion | `exp17_clap_separability.sh` | 512-dim noise as the floor (0.922) | criterion 2, invariant to dimensionality | **NO-GO**: CLAP 3.837 · mel-8 3.131 · MFCC-13 2.084 | `docs/provenance/2026-08-12_exp17_clap_separability.txt` + log and manifest | 512-dim noise **in the same run, before** the CLAP row · md5 `a3c7533c…`/`f3ebe89b…` | yes (stage 2, 24 s) · extraction needs the checkpoint |

### Chapter 2 — the contrastive arm (GPU)

| experiment | script | null | threshold | verdict | provenance | canary | CPU? |
|---|---|---|---|---|---|---|---|
| **step C** — CLAP↔EEG, batch negatives | `stepC_contrastive_clap.sh` | **0.500 by construction** | **88/154 = 0.5714** | **below chance and explained**: 58/154 = 0.3766, 95% CI [0.300, 0.458] excludes 0.50. The sign is **not** flipped | `RESULTS.md` madeeg_duo, pin `madeeg_clap_kfold` · run archive | `clip_loss.py` · `--loss` asserted · adapter `_self_check` · **`--check_rule` → 58/154 exact** | gates yes · training needs a GPU |
| **step D** — within-mixture loss | `stepD_within_mixture.sh` | **0.500 by construction** | **88/154 = 0.5714** (the same, not recomputed) | primary 70/154 = 0.4545 **and** the secondary **falsified in the opposite direction**: prior-following rises 0.7177 → 0.8145 | run archive · `docs/provenance/2026-08-11_diagnose_step{C,D}.txt`. **Not pinned** in the manifest | `clip_loss.py` line 2 **is** step D's objective (0.951610 vs 2.160834) · a mistyped `--loss` **breaks** · separate directories | gates yes · training needs a GPU |
| Exp. 18 — match-mismatch on the solos | `exp18_matchmismatch.sh` | S1 **0.5000** · S2 0.5063 (held-out **0.6061**) | 0.70 on the held-out set | S1 **278/536 = 0.5187**, gate not passed, so S2 was not run | `docs/provenance/2026-08-12_exp18_matchmismatch.txt` · `2026-08-13_exp18_S1_RESULT_*` | Chapter 1 canary · sampling tested pair by pair · 20/20 overfit | yes, with `/opt/anaconda3` (needs torch) · training needs a GPU |
| Exp. 19 — epoch budget + repaired S2 null | `exp19_matchmismatch.sh` | S2 **0.6061 → 0.5000** after `--balance_pairs` | 0.70 (unchanged) | S1 **244/536 = 0.4552** · S2 **113/188 = 0.6011** (exploratory, p 0.0034). Both below the gate; the train loss falls 0.6962 → 0.3160 while the held-out set stays flat | `docs/provenance/2026-08-13_exp19_S{1,2}.txt` and `_RESULT_*` | the four gates · Chapter 1 canary identical before **and** after the diff | yes, with `/opt/anaconda3` · training needs a GPU |

### Cross-cutting

| what | script | verdict | canary | CPU? |
|---|---|---|---|---|
| **all canaries, in order** | `canaries.sh` | printed by the run; **exit ≠ 0** if a crossable canary fails | **is** the canary file | L0 yes · L1 needs torch · L2 +MAD · L3 printed |
| exp13 · exp07 · exp08 · exp14 in sequence | `run_all_cheap.sh` | — | those four | yes, +MAD (~25–50 min) |

**How to read the "looks" column** (which lives in each script's header, not here):
*NO* means the ledger of looks declares the experiment free (audio only, or
own-vs-other on held-out solo segments). *YES* means it decides on the duos. All
309 duos have been spent since 2026-07-30: rerunning them opens no new material,
but every duo number is **exploratory by construction** and has to be labelled so
in the title. Spending looks: `ridge_anchors` · `axes_paper_protocol` · `exp04` ·
`exp06_alpha_paired` · `exp09` · `exp11` · `exp12` · `exp15` · `armA` · `armD` ·
`stepC` · `stepD`.
**No script in this directory touches the trios** (92 stereo + 93 mono, the only
remaining holdout, one shot, stereo threshold already written at 38/90).

---

## From scratch: environment, data, order

### 1. The environments (there are **three**, and they are not interchangeable)

| for | interpreter | what it must contain |
|---|---|---|
| `torch` canaries (`clip_loss.py`), Chapter 1 | conda **`attention`** — `eeg_attention` on the cluster | `torch==2.2.2`, `numpy<2`, `pytorch_lightning==1.9.5` → `pip install -r requirements.txt` |
| MAD-EEG CPU analysis (every `madeeg_*.py`) | **`/opt/miniconda3/bin/python`** | numpy · scipy · sklearn · **h5py** · mne · librosa · pandas · pyyaml |
| Exp. 16A/17/18/19 (needs `torch` **and** `laion_clap`) | **`/opt/anaconda3/bin/python`** | torch + `laion_clap` 1.1.6 |

Why three and not one: the `attention` env **has no `h5py`** and cannot execute
`madeeg_reconstruction.py`; `/opt/miniconda3` has no `torch`. It is not a
convenience, it is the reason the scripts have `$PY` **and** `$PY_TORCH`.

```bash
# Chapter 1 env — 3.9 is the version the root README declares; requirements.txt
# pins torch 2.2.2 + numpy 1.26.4 + pytorch_lightning 1.9.5, which is the real constraint.
conda create -n attention python=3.9 && conda activate attention
pip install -r requirements.txt
python src/run.py --selftest        # must say: 23 protocol flags ... PASS
python src/modules/clip_loss.py     # must say: 0.628491 / 1.2994 / 4.8198 / 0.951610 vs 2.160834
```

### 2. Where the data goes

- **MAD-EEG is not in the repo** (4.7 GB): `MADEEG_DIR=~/madeeg bash scripts/madeeg_setup.sh`.
  `madeeg_setup.sh` fetches **only the preprocessed release**: the `raw_solos`
  path (own-vs-other, Exp. 4/6/9, arm A, the axes) **also** needs
  `madeeg_raw.hdf5` + `madeeg_raw.yaml` + `madeeg_sequences_raw.yaml` + `stimuli/`.
  The script notices and says so; it does not guess.
- **Paper checkpoints** (Chapter 1): `bash scripts/setup_checkpoints.sh`, which
  unpacks them from `archive/*.7z`. The `.ckpt` files are gitignored.
- **CLAP checkpoint** `630k-audioset-best.pt` (1.74 GiB), sha256
  `8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`: downloading
  it is a deliberate decision, not something a script does on its own.
- **`runs/` is gitignored.** From a clean clone the run records **do not exist**:
  the scripts that re-read them say so rather than pretending. The one exception is
  `!docs/provenance/*.log`.

### 3. The order things run in

```bash
# 0. ALWAYS first, after any diff. Level 0 needs nothing.
python src/run.py canaries --levels 0
PY_TORCH=~/miniconda3/envs/attention/bin/python python src/run.py canaries --levels "0 1"

# 1. Chapter 1, from the released checkpoints (no training)
bash scripts/setup_checkpoints.sh
python src/run.py exp cap1_within_split        # ~25 min CPU
python src/run.py exp cap1_leave_song_out      # re-reads the 80 pins, seconds
python src/run.py exp cap1_leave_subject_out   # ~20 min CPU

# 2. Chapter 2, free (no held-out looks): audio only and own-vs-other
python src/run.py exp exp13                    # ~10 s — the separability gate
python src/run.py exp exp06_ovo                # the 208/376 reference + its md5
python src/run.py exp exp07
python src/run.py exp exp08
python src/run.py exp exp14
python src/run.py exp exp16b
python src/run.py exp exp17                    # stage 2

# 3. Chapter 2 that SPENDS LOOKS (duos already spent, but the caveat goes in the title)
python src/run.py exp ridge_anchors
python src/run.py exp axes_paper_protocol      # 16 GB RAM
python src/run.py exp exp04
python src/run.py exp exp06_alpha
python src/run.py exp exp09
python src/run.py exp exp11
python src/run.py exp exp12
python src/run.py exp exp15
python src/run.py exp armA
python src/run.py exp armD

# 4. Canary level 2 (reruns the reference and the anchors from scratch)
python src/run.py canaries --levels 2

# 5. The contrastive arm: gates on CPU, training printed
python src/run.py exp stepC
python src/run.py exp stepD
python src/run.py exp exp18
python src/run.py exp exp19

# ...or all of the above, in this order, one log file per experiment:
python src/run.py exp all
```

### 4. What needs a GPU

Every script here **prints its GPU command and stops**. The complete list:

| printed by | what it produces |
|---|---|
| `canaries.sh` L3-a · `cap1_within_split.sh` | `PHASES="within sanity" bash scripts/replicate.sh` after `setup_checkpoints.sh` → GLOBAL **0.8650 exact** end to end (CPU, but needs the checkpoints) |
| `cap1_within_split.sh` | `run.py train --model {audio_only,eeg_only,clap}` + `main.py --objective classify_audio --audio_repr clap` → the `within` rows with no released checkpoint, including the **0.996 leakage control** |
| `cap1_leave_song_out.sh` | `run.py sweep --cv song` (4 models × 20 songs), ~80 min/fold |
| `cap1_leave_subject_out.sh` | `run.py sweep --cv subject` |
| `canaries.sh` L3-b · stepC/stepD | `madeeg_contrastive.py --self_test [--loss within_mixture] --kfold 5` → the synthetic controls **148/154** and **142/154** (threshold 0.90 in the code) |
| stepC/stepD | `madeeg_contrastive.py --kfold 5 [--loss within_mixture]` → the real numbers 58/154 and 70/154 |
| `exp18`/`exp19_matchmismatch.sh` | the S1/S2 match-mismatch training commands |
| `exp16a`/`exp17_clap_separability.sh` | CLAP extraction (stage 1) → the embeddings plus the manifest with the checkpoint's sha256 |

After a GPU step the diagnostics go back to CPU: the two
`madeeg_diagnose.py --madeeg_dir ~/madeeg --records …` commands for steps C and D
are printed at the end of `stepD_within_mixture.sh` and reproduce
`docs/provenance/2026-08-11_diagnose_step{C,D}.txt`.

---

## Rules these scripts follow, and that whoever edits them does not renegotiate

Full text with the incident behind each one: [`docs/METHOD_RULES.md`](../../docs/METHOD_RULES.md).

- **The canary first.** Where one exists it is crossed before any real number and
  the script stops if it fails (rule 3). Exp. 6/7/8/14/16B check the md5
  `2eaa926244de340d31907c6deeb04b0c` of the default path; Exp. 9 and 15 check that
  the mel anchors are **decision-identical** trial by trial (74/154 and 86/154,
  zero discordant); Exp. 11/12/13 have their gates inside the drivers, which exit
  with an error before computing the real number.
- **A control never writes over a result** (rule 9): `--training_date` values are
  prefixed with `$TAG` and provenance goes to `$OUT_DIR`. Two declared exceptions:
  the drivers `madeeg_exp1{4,5}_*.py` do not accept `--log_dir` and rewrite
  `runs/results/exp1{4,5}_*` — the pinned provenance files stay intact because
  `--out` is passed. It is written at the top of those two scripts.
- **A canary that was not crossed is not a canary that passed.** Scripts that look
  for `$PY_TORCH` and do not find it print `NOT CROSSED` and say so explicitly.
  There is no fourth category between passed, failed and not crossed.
- **No plausible invocations** (rule 10): anything not reconstructible with
  certainty is commented `TO BE CONFIRMED:` with the reason. The open cases today
  are: the **same/diff-melody split of arm A** (no file in the repo produces it,
  the script was not saved); the **runtimes in minutes** of much of Chapter 2 (the
  2026-08-11 timestamps are identical and give no duration); and the **exact cause**
  of the 5 missing trials on Exp. 4's mono arm (the 155/150 count is written down,
  the mechanism is not).
- **No GPU** (rule 11). No script here needs one to run: where it is needed, the
  command is **printed**, not launched.

## What this directory does not cover

- **Exp. 10** — retired **before** the run (insensitive by algebra). The flag
  `--score_rule ortho` exists and is inert: **do not produce numbers with it**.
  The toy that shows it is `docs/provenance/2026-08-11_exp10_retired_toy_sweep.py`
  and `canaries.sh` L0-d crosses it.
- **C1-8 (`--eeg_repr spectra`)** — 0.9350 / 0.4975 / 0.4713: never aggregated,
  never pinned, **no canary**. It has no script and should not have one until those
  numbers enter the manifest.
- **Arm A's same/diff-melody split** — no file in the repo produces it. The numbers
  (0.5169/0.5410 and 0.5730/0.4918) have their provenance file
  (`docs/provenance/2026-08-11_armA_same_diff_melody.txt`) **but not the code**.
  It has to be rewritten, not guessed.
