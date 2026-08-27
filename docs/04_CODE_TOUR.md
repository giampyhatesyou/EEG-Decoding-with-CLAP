# 4 · Code tour

*Reading path: [1 README](../README.md) → [2 REPO_MAP](02_REPO_MAP.md) → [3 OVERVIEW](03_OVERVIEW.md) → **4 you are here** → [5 replicate](../scripts/replicate/README.md) → [6 provenance](provenance/README.md) → [7 METHOD_RULES](07_METHOD_RULES.md).*

How this repository is put together, from the outside in. Read
[`03_OVERVIEW.md`](03_OVERVIEW.md) first if you want to know *what the experiments are
asking*; this file is about *where the code that asks it lives*.

---

## 1. Three levels, and one rule

```
src/run.py                    order, logging, no science
  └─ scripts/replicate/*.sh   THE EXACT COMMAND + the canaries + the expected numbers
       └─ src/madeeg_*.py     the computation
```

The rule that holds it together: **the command of an experiment lives in exactly
one place**, its script. `run.py` does not know it — it knows only the order to
run things in and what to call the log file. That is why `run.py exp --list` has
no hand-written table: it reads title, held-out-look cost and GPU requirement
back out of the script headers, so the listing cannot drift from the source.

Chapter 1 is the exception and predates the rest: `run.py` drives it directly,
because it owns the fixed Akama protocol (`PROTOCOL`, the 23 flags) and builds
the `main.py` / `checkpoint_test.py` command line itself.

## 2. One experiment, end to end

```bash
python src/run.py exp exp06_ovo
```

1. `run.py` resolves the prefix to `scripts/replicate/exp06_ovo_gate.sh`, opens
   `runs/logs/<timestamp>_exp06_ovo_gate.log` and tees the script's merged
   stdout/stderr into it.
2. The script pins `PY=/opt/miniconda3/bin/python`, `MADEEG_DIR=~/madeeg`,
   `TAG=repl_`, and defines `BASE=(...)` — the invariants of the pre-registered
   criterion. Those flags **are the identity of the number**.
3. **First thing it does is the canary.** It reruns the `ridge + notch_ica`
   configuration and compares the md5 of `madeeg_ownvsother.csv` against
   `2eaa9262…`. If it disagrees, `exit 1`, and no candidate is computed.
4. Then the other three configurations, then `madeeg_diagnose.py --mcnemar` to
   pair them, then the exact paired p values.
5. Every run writes to `runs/results/repl_<name>/`. The `repl_` prefix is what
   stops a replication from overwriting a pinned record.

That shape — *canary, then the runs, then the paired test, then a comparison
against the reference numbers in the header* — is the same in all 28 scripts.

## 3. `madeeg_reconstruction.py` — the workhorse

1687 lines, and roughly 80 % of Chapter 2. It is not one program: it is a
`main()` with six mutually exclusive modes, selected by flag.

| flag | what it does | key function |
|---|---|---|
| `--inspect N` | prints the schema of the first N trials and exits | — |
| `--check_alignment` | recovery gate: EEG/audio durations, declared onsets, the borrowed-stem mechanism. PASS/FAIL, exits | `check_alignment()` |
| `--self_test` | positive control: EEG replaced by a **synthetic** mixture of the attended source at the model's lags, then the real ridge | `synth_attended_eeg()` |
| `--alpha_li` | sign test on lateralised alpha. No decoder, no audio, no training | delegates to `models/alpha_lateralization.py` |
| `--own_vs_other` | discrimination on held-out **solo** segments, n = 376, null 0.500 exact | `mutual_pairs()` |
| *(none)* | the normal path: k-fold, the duo attention decision | the `for subj in meta` loop |

`--inner_val_only` is a modifier of the last one: it fits the decoder exactly as
a full run would and reports `inner_val_r` only, building no test trial — so no
accuracy exists to be read by accident.

### The data path

```
read_f64()        MAD-EEG's HDF5 uses a non-standard float layout; h5py alone fails
process_eeg()     band-pass 1-8 Hz -> RobustScaler + clamp -> downsample to 64 Hz
source_repr()     audio -> mel / envelope / mag / flux / mfcc      (this is --target)
build_trial()     assembles the EEG and the sources present in THAT mixture
lagged_design()   design matrix over lags 0-250 ms
fit_pool()        -> fit_ridge_pool()  or  fit_cca_pool()
native_score()    band_pearson between the reconstruction and each candidate source
```

**The decision is always the same line, conceptually**: argmax, over the sources
*present in that trial*, of the mean per-band Pearson correlation. That is why
chance is `1/n_present` (0.500 on a duo) and not 1/9 — the choice is internal to
the trial, so stimulus identity cannot win it.

### Why it can run without torch

`_load_by_file()` imports repo modules **by path, never as a package**, because
`utils/__init__.py` and `models/__init__.py` pull in torch. That is what makes
this the one MAD-EEG binary that runs in the `h5py` environment on CPU, with no
cluster and no GPU.

### The ancestor, kept on purpose: `stimulus_reconstruction.py`

`stimulus_reconstruction.py` is the **Chapter 1** backward model: the same idea as
`madeeg_reconstruction.py`, written first, on the Akama dataset. It reuses that
dataset's class only to get identical cross-validation splits, and it imports
nothing from the contrastive path. **No experiment in the register runs it and
it produces no reported number** — the question it was written to ask moved to
MAD-EEG, where the within-song comparison it relies on is not confounded by the
fixed song-to-target mapping. It is kept because it is where the Chapter 2
workhorse comes from, and because deleting the ancestor of a method is not the
same as simplifying it. Its own header states the rationale and the limitation.

## 4. The other drivers split into two families

### (a) Real drivers — they compute

| file | experiments | what it actually does |
|---|---|---|
| `madeeg_stem_separability.py` | 13 / 16A / 17 | **never opens the EEG**. Reads `soli` and the metadata only, and correlates stem against stem under six representations. Hence 10 s and zero held-out looks |
| `madeeg_spectral_attention.py` | 11 / 12 | Morlet 1-40 Hz -> shrinkage LDA -> LOSO on twin pairs. `--source raw` is Exp. 12 |
| `madeeg_leakage_audit.py` | arm D | 165 lines: `band_logvar()` + `cv_acc()` at two levels (window vs trial). A claim about the **protocol**, not about the EEG |
| `madeeg_contrastive.py` | steps C/D, 18, 19 | the only one that wants a GPU |

### (b) Orchestrators — they judge

`madeeg_exp14_tracking.py`, `madeeg_exp15_differential.py` and
`madeeg_exp16b_ccaviews.py` compute nothing. All three share `run_recon()`:

```python
cmd = [PY, RECON, "--madeeg_dir", madeeg_dir, "--training_date", tag] + extra
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0: sys.exit(f"run {tag} failed -- STOP")
```

and then read the result back **from the summary file**, not from stdout
(`ovo_result()`, `selftest_acc()`). The reason is methodological: an experiment
cannot end up using a different version of the computation from the reference it
compares itself against, and its gates run inside the driver before the real
number is computed. If you open `exp14` looking for the tracking arithmetic, it
is not there — it is in `madeeg_reconstruction.py`.

## 5. The contrastive arm

```
madeeg_contrastive.py
  ├─ datasets/madeeg_contrastive_dataset.py   steps C/D   (PREPROCESSED release, duo/trio)
  ├─ datasets/madeeg_solo_matchmismatch.py    Exp. 18/19  (RAW release, the solos)
  ├─ models/clap_encoder.py                   FROZEN CLAP + trainable head
  ├─ models/sample_cnn2d_eeg.py               EEG encoder (unchanged from Chapter 1)
  └─ modules/clip_loss.py                     InfoNCE — FROZEN at bb016fd
```

Two dataset classes because **the preprocessed release contains no solo trial at
all** (154 duo + 92 trio). The solos exist only in the raw release, as segments
of the continuous record cut at the onsets `madeeg_sequences_raw.yaml` declares.
It is a different I/O path, not a different experiment: everything downstream is
the code steps C and D already ran.

`--loss` has four values, and it changes **only where the negative comes from**:

| value | the negative is | experiment |
|---|---|---|
| `batch` | whatever landed in the batch | step C — 58/154 |
| `within_mixture` | the other stem of the same duo | step D — 70/154 |
| `temporal_offset` | the same piece at another instant | Exp. 18/19 S1 |
| `cross_instrument` | same piece, different instrument | Exp. 19 S2 |

**The neatest thing in the codebase is here**: `madeeg_contrastive.py` does
`from madeeg_diagnose import trial_decision`. The decision rule has **one
implementation**, and it lives in the diagnostic file, which does not import
torch. That is why `--check_rule` can rerun step C's 154 decisions offline and
recover exactly 58/154.

### The two audio towers, and what each one costs

The `--audio_repr` switch in `main.py` is the whole architectural difference
between the two Chapter 1 models, and it is worth stating in parameters because
the comparison in `RESULTS.md` is otherwise easy to misread.

| variant | audio side | trainable | frozen |
|---|---|---:|---:|
| `raw` (Akama baseline) | **four independent** `SampleCNN2DEEG`, one per stem | **580,600** | — |
| `clap` (the extension) | **one shared** frozen LAION-CLAP tower + one `Linear(512,256)-GELU-Linear(256,100)` head, referenced from all four slots | **273,148** | ~158 M |

Measured, not inferred: `SampleCNN2DEEG` is **116,120** parameters whatever it is
built on, because nothing about the input reaches its shape -- `out_dim` is accepted
and never used, the convolutional stack ends in `adaptive_avg_pool2d`, and the head is
a hardcoded `128 -> 100 -> 100`. Its other argument, `kernal_size`, IS the Conv2d
kernel and does move the count; all five construction sites, in `main.py` and
`checkpoint_test.py`, pass `3`. The same instance runs on a `(4, 768)` EEG window
and on a `(4, 177147)` audio clip with the same weights. So `raw` is
5 x 116,120 and `clap` is 116,120 + 157,028, and the reading is that **CLAP
reaches a higher within-split accuracy with less than half the trainable weight
on the audio side** -- one small head against four convolutional networks. The
frozen ~158 M is resident regardless: `CLAP_Module` also loads a RoBERTa text
tower (~124.6 M) that the forward pass never calls.

Two limits travel with every CLAP figure, and both are declared rather than fixed:

* **The input is out of CLAP's regime.** LAION-CLAP was pre-trained on 10-30 s
  clips; the Chapter 1 pipeline pads each 3 s clip to 3^11 samples at 44.1 kHz
  (~4.016 s), so CLAP sees roughly a second of silence on each side. Removing the
  padding means changing the dataset code, which is held fixed so the baseline
  stays reproducible. The same gap, four orders of magnitude worse, is what
  Exp. 17 measures in Chapter 2 at a 62.5 ms window.
* **Only the head moves.** If the bottleneck were the rigidity of the
  representation rather than the head, the next step would be parameter-efficient
  fine-tuning; nothing here measures which of the two it is.

`madeeg_diagnose.py` is the post-mortem: `--records` (prior-following),
`--mcnemar` (paired), `--paired`, `--check_rule`, `--demo`. It is the file that
turned step C's failure into the chapter's mechanism.

## 6. The modules under `models/`

- **`cca_multiview.py`** — features and algebra, **no I/O**: it does not know
  what a subject or a fold is. The entrypoint owns the data path and the
  decision rule. Its self-check is canary L0-c, and it also proves the
  `position` view is **inert**: it is constant in time, so a within-trial
  Pearson centres it out.
- **`alpha_lateralization.py`** — the laterality index plus
  `inject_lateralized_alpha()`, which is the positive control. No training.
- **`clap_encoder.py`** — CLAP frozen (`requires_grad=False`, `.eval()`); only
  the projection head moves.
- **`spectra_eeg.py`** — Chapter 1 only, an alternative EEG front end. It does
  not enter Chapter 2.

## 7. Where the numbers end up

`resolve_log_dir()` in `utils/paths.py` sends everything to
`runs/results/<training_date>/`. The file names are a fixed convention:

```
madeeg_summary.txt             duo decision              madeeg_records.csv
madeeg_ownvsother_summary.txt  solo discrimination       madeeg_ownvsother.csv
madeeg_selftest_summary.txt    positive control          (directory <tag>_selftest/)
madeeg_alpha_summary.txt       alpha test                madeeg_alpha_{li,pairs}.csv
madeeg_innerval_summary.txt    --inner_val_only
madeeg_matchmismatch_*         Exp. 18/19
madeeg_contrastive_*           steps C/D
```

`runs/` is **gitignored**, which creates **two degrees of provenance**, and the
distinction matters when citing:

- **degree 1** — `docs/provenance/*`: the integral output of the run, in the repo.
- **degree 2** — a replication script's header: *the value the run must produce*,
  verified once by its author. To check it, you have to rerun.

Chapter 1 has a third route: `results_manifest.tsv` -> `sweeps/report.py` ->
`RESULTS.md`, where `report.py` verifies **every pin** against its run's
`hparams.yaml` and refuses to write the table at all if one disagrees.

## 8. How to read an experiment you have not seen before

1. `python src/run.py exp --list` — find the name.
2. `head -60 scripts/replicate/<name>.sh` — null, threshold written in advance,
   verdict, expected numbers, canary, provenance file. **The header is the
   source, not a summary of one.**
3. See which driver it invokes. If it is `madeeg_reconstruction.py`, the flags in
   `BASE=(...)` are the identity of the number: change one and it is a different
   experiment.
4. `cat docs/provenance/<file>` for the real output — [`provenance/README.md`](provenance/README.md)
   is the index from number to file.
5. [`07_METHOD_RULES.md`](07_METHOD_RULES.md) if a comment cites `method rule N`.
