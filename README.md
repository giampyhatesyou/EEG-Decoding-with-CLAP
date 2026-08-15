# EEG-based Auditory Attention Decoding in Polyphonic Music

Master's thesis code. Two arms, two datasets, two chapters — and they answer
different questions, so their numbers never share a table.

- **Chapter 1 — the negative/methodological arm (Akama et al. 2025, consumer EEG).**
  Reproduces the published contrastive EEG↔audio baseline and then shows what it
  actually learned: within-split **0.865**, leave-song-out **0.268** (chance 0.25),
  leave-subject-out **0.943**. The model generalises across *people* but not across
  *songs*, and the audio alone already identifies the target at **0.996** — in that
  dataset song ≡ target, so "which instrument is attended" is aliased with "which song".
- **Chapter 2 — the positive/identifiable arm (MAD-EEG, 20-channel lab EEG).**
  The same mixture is attended with **different** targets across trials, so attention
  is identifiable and stimulus identity cannot win. Everything here is a
  pre-registered test with a written threshold, and most verdicts are honest
  negatives: the linear family peaks at **0.5584** on the duo (threshold 0.5714), and
  the contrastive arm lands **below** chance at **0.3766** for a diagnosed reason
  (prior-following, not noise).

> **What was stale in this file, and is fixed here (15 Aug 2026).** The previous
> version described *only* Chapter 1. It listed none of the MAD-EEG arm — 10 driver
> scripts (~6 000 lines) under `src/` plus their dataset adapters and models, the 28
> scripts in `scripts/replicate/`, and the 20 pinned files in `docs/provenance/` —
> and its `src/` tree was missing
> `stimulus_reconstruction.py`, `models/{spectra_eeg,cca_multiview,alpha_lateralization}.py`,
> `modules/supervised_classification.py` and both MAD-EEG dataset adapters. It also
> claimed `python src/run.py replicate` reproduces "every number this project
> reports": it reproduces **Chapter 1's**. Chapter 2 needs the MAD-EEG dataset and
> `scripts/replicate/`. `docs/METHODOLOGY.md` and `docs/CHANGES_FROM_BASELINE.md`
> (both 26 Jul) still cover Chapter 1 only — `docs/MADEEG.md` is the Chapter 2 note.

## Reference papers

> Akama T. et al. *Decoding Selective Auditory Attention to Musical Elements in
> Ecologically Valid Music Listening*. Sony CSL, 2025. — `docs/paper.pdf`.
> Upstream code: <https://github.com/JURIUENO11/Music_attention> (its README is kept
> as `docs/LEGACY.md`).

> Cantisani G. et al. *EEG-based decoding of auditory attention to a target
> instrument in polyphonic music*. WASPAA 2019. — `docs/cantisani_waspaa2019.pdf`.
> The MAD-EEG arm reproduces this protocol; its 79 F1 on duets does **not** reproduce
> from any package faithful to the declared methods (see `armA_paper_protocol.sh`).

## Repository layout

```
.
├── results_manifest.tsv   THE pinned list of folds behind every reported number
├── RESULTS.md             Generated from it by sweeps/report.py — do not edit by hand
├── requirements.txt       Chapter 1 stack, pinned (torch 2.2.2, numpy 1.26.4, PL 1.9.5)
├── configs/baseline.yaml  Config template + the audio_repr / objective / cv_mode switches
│                          (⚠️ loose defaults: the REAL protocol is $PROTO / PROTOCOL)
├── src/
│   ├── run.py                     Launcher: replicate | train | sweep | report | --selftest
│   ├── main.py                    Chapter 1 training entry point
│   ├── checkpoint_test.py         Chapter 1 evaluation (writes the breakdown)
│   ├── stimulus_reconstruction.py Chapter 1 linear reconstruction probe
│   ├── madeeg_reconstruction.py   ── CH.2 WORKHORSE (1687 l): ridge/shrinkage/CCA stimulus
│   │                              reconstruction, own-vs-other, alpha LI, the four axes,
│   │                              arm A, Exp. 4/6/7/8/9/10
│   ├── madeeg_diagnose.py         CH.2 post-mortem: prior-following, McNemar, --check_rule
│   ├── madeeg_contrastive.py      CH.2 steps A/B/C/D + Exp. 18/19 (GPU)
│   ├── madeeg_spectral_attention.py  CH.2 Exp. 11/12 — band power + LDA on the register
│   ├── madeeg_stem_separability.py   CH.2 Exp. 13/16A/17 — audio-only separability gate
│   ├── madeeg_leakage_audit.py       CH.2 arm D — window-CV leakage audit
│   ├── madeeg_exp14_tracking.py      CH.2 Exp. 14 — MFCC/mel-64 tracking
│   ├── madeeg_exp15_differential.py  CH.2 Exp. 15 — attentional differential
│   ├── madeeg_exp16a_clap_extract.py CH.2 CLAP embedding extraction (stage 1)
│   ├── madeeg_exp16b_ccaviews.py     CH.2 Exp. 16B — band_power as a CCA view
│   ├── datasets/
│   │   ├── preprocessing_eegmusic_dataset.py  Ch.1 dataset + CV routing
│   │   ├── madeeg_contrastive_dataset.py      Ch.2 contrastive adapter (steps C/D)
│   │   └── madeeg_solo_matchmismatch.py       Ch.2 match-mismatch sampler (Exp. 18/19)
│   ├── models/
│   │   ├── sample_cnn2d_eeg.py     Akama 2D-CNN EEG / raw-audio encoder
│   │   ├── clap_encoder.py         Frozen LAION-CLAP + projection head (extension)
│   │   ├── spectra_eeg.py          Ch.1 spectral EEG front end (--eeg_repr spectra)
│   │   ├── cca_multiview.py        Ch.2 model 9 — regularised multi-view CCA
│   │   ├── alpha_lateralization.py Ch.2 Exp. 6 [H] — alpha lateralisation index
│   │   └── model.py                Base nn.Module the encoders subclass
│   ├── modules/
│   │   ├── clip_loss.py            InfoNCE — 🔒 FROZEN AT bb016fd, see below
│   │   ├── contrastive_learning.py LightningModule: loss, audit hooks, breakdown
│   │   └── supervised_classification.py  classify_eeg / classify_audio controls
│   ├── preprocessing/ · utils/     Transforms; config loader, paths, logging
├── scripts/
│   ├── replicate.sh          Chapter 1 from the released checkpoints (PHASES=...)
│   ├── setup_checkpoints.sh  Unpack the paper checkpoints from archive/*.7z
│   ├── madeeg_setup.sh       Fetch MAD-EEG (⚠️ preprocessed release only)
│   ├── train.sh              Train one model on one split
│   └── replicate/            ── ONE SCRIPT PER EXPERIMENT (28) + its own README
├── sweeps/
│   ├── sweep_common.sh       Conda env, $PROTO (the 23 fixed flags), resume logic
│   ├── sweep_{song,subject}_out.sh   Resumable CV sweeps
│   └── report.py             Renders RESULTS.md; verifies every pin, or writes nothing
├── docs/
│   ├── provenance/           20 PINNED result files — the evidence behind Chapter 2
│   ├── MADEEG.md             The Chapter 2 note (why MAD-EEG is the positive arm)
│   ├── METHODOLOGY.md · CHANGES_FROM_BASELINE.md · LEGACY.md   (Chapter 1)
│   └── paper.pdf · cantisani_waspaa2019.pdf · model_architecture.png
├── checkpoints/ · archive/   Paper weights (.ckpt gitignored; .7z tracked)
├── dataset/                  Chapter 1 EEG + per-stem audio
└── runs/                     Everything a run emits — GITIGNORED
    ├── results/              Per-run CSVs, summaries, hparams, checkpoints
    ├── replicate/            Where the replication scripts write their provenance
    └── logs/                 stdout of the launchers
```

**MAD-EEG is not in this repo** (4.7 GB). It lives at `--madeeg_dir ~/madeeg`.

## Where the results live

`results_manifest.tsv` lists every fold behind a reported number — model,
evaluation, held-out id, run directory, code vintage. `sweeps/report.py` reads
**only** that file and renders `RESULTS.md` (**107 pinned folds** today).

Nothing is discovered by scanning the disk, so a number cannot change because a new
run directory appeared. `report.py` also checks each pin against the run's own
`hparams.yaml` / `madeeg_summary.txt` and **refuses to write the table** if one
disagrees — it has already rejected two pins written by its own author. (This
replaced an aggregator that globbed and de-duplicated by a partial key: two runs of
the same fold collided and whichever the filesystem returned last silently won,
which moved a reported number from 0.142 to 0.154 with no new evidence behind it.)

Chapter 2's evidence lives in **`docs/provenance/`**: the integral output of the run
that produced each number, pinned and md5-checked. `runs/` is gitignored, so from a
clean clone the run directories do **not** exist — the replication scripts say so
rather than pretending.

## How to replicate an experiment

```bash
# Chapter 1 — from the authors' RELEASED checkpoints. No training, no GPU.
pip install -r requirements.txt        # conda env `attention`: torch 2.2.2, numpy<2
bash scripts/setup_checkpoints.sh
python src/run.py replicate            # ~45 min CPU  (== bash scripts/replicate.sh)

# Chapter 2 — one script per experiment, each with the expected numbers in its header.
MADEEG_DIR=~/madeeg bash scripts/madeeg_setup.sh
bash scripts/replicate/exp13_stem_separability.sh      # ~10 s, no EEG opened
bash scripts/replicate/exp06_ovo_gate.sh               # the 208/376 reference + its md5

# Before and after ANY diff: the regression gates, in one command.
LEVELS=0 bash scripts/replicate/canaries.sh
```

`scripts/replicate/README.md` is the map: one row per experiment with its null,
its pre-registered threshold, its verdict, its vault contract, its provenance file,
its canary, and whether it runs on CPU. Read it before running anything.

The launcher surface is five commands; anything else is a one-off that belongs in a
scratch shell:

```bash
python src/run.py                      # interactive launcher
python src/run.py replicate            # Chapter 1 from released checkpoints
python src/run.py train --model clap   # --model baseline|clap|audio_only|eeg_only
python src/run.py sweep --cv song      # resumable leave-one-song-out sweep (GPU)
python src/run.py report               # re-render RESULTS.md from the manifest
```

## Which flags change a number

This is the part that costs money when it is wrong. Every entry below is a flag whose
value **is part of the identity of a reported number**: change it and you have a
different experiment, not a different run of the same one.

### Chapter 1

| flag | values | what moves |
|---|---|---|
| `--cv_mode` | `within` · `leave_song_out` · `leave_subject_out` | **0.865** vs **0.268** vs **0.943**. This single flag *is* the chapter's result |
| `--shuffle_test_mode` | `none` · `labels` · `audio_pair` | 0.865 → 0.225 → 0.383. ⚠️ the two controls are **not** bit-reproducible (`checkpoint_test` forces `shuffle=True`); cite them as "at chance", never as exact digits |
| `--audio_repr` | `raw` · `clap` | within 0.875 (raw) vs 0.946 (clap, 8 seeds) |
| `--objective` | `contrastive` · `classify_eeg` · `classify_audio` | `classify_audio` is the **leakage control**: 0.996 with CLAP audio, 0.967 with raw. It must be cited *next to* the 0.865, not after it |
| `--eeg_repr` | `raw` · `spectra` | ⚠️ the `spectra` numbers (0.9350 / 0.4975 / 0.4713) are **not in the manifest and have no canary**. Do not quote them as pinned |
| `$PROTO` / `PROTOCOL` | 23 flags | the real hyperparameters — **not** `configs/baseline.yaml`, whose defaults are a loose template. `python src/run.py --selftest` asserts the two copies still agree, flag by flag |

### Chapter 2 (`madeeg_reconstruction.py` unless noted)

| flag | values | what moves |
|---|---|---|
| `--train_on` | `duos_kfold` · `raw_solos` | 0.5584 vs 0.4805 on the duo decision |
| `--test_eeg` | `preprocessed` · `raw` | **the trap that bit.** With `raw_solos`, the documented command *without* `--test_eeg raw` gives **0.4870**, not the archived **0.4805**. Two numbers one digit apart for an unwritten flag |
| `--spatial` | `stereo` · `mono` · `both` | which duo set is tested: 154 stereo vs 150 decidable mono. Asserts `--train_on raw_solos --test_eeg raw` (mono duos do not exist in the preprocessed release) |
| `--eeg_clean` | `none` · `notch` · `notch_ica` | the only axis that gains: `inner_val_r` 0.0582 → 0.0673 (+16%), own-vs-other 202/376 → 208/376. Asserts `raw_solos` |
| `--filters` | `pooled` · `per_instrument` | `inner_val_r` 0.0582 → 0.1100, but ⚠️ that is a **metric artefact**, not a decoder gain (fair test 36/70, p 0.29). Also drops n from 154 to 150. **Forbidden** with `--own_vs_other`, by assert |
| `--estimator` | `ridge` · `shrinkage` · `cca` | `inner_val_r` 0.0582 → 0.0349 (shrinkage); own-vs-other ridge 208/376 vs CCA 209/376 (the Exp. 6 NO-GO) |
| `--target` | `mel` · `envelope` · `mag` · `flux` · `flux_mel` · `mfcc` · `mfcc_c0` | own-vs-other mel 208/376 vs flux **243/376**; duo accuracy mel 0.5584 vs envelope 0.5065 |
| `--n_mels` · `--target_fs` | 8 @ 64 (ours) · 24 @ 256 (published) | `inner_val_r` 0.0582 → 0.0468, accuracy 0.4805 → 0.4870. The published hyperparameters make it **worse**, and the two halves worsen independently |
| `--band_low/high` | 1–8 Hz (ours) | ⚠️ **our choice, not the paper's** — Cantisani et al. do not band-pass at all. Declared inside every summary file |
| `--own_vs_other` | flag | switches from the duo decision to held-out **solo** discrimination: n = 376, null 0.500 *exact by symmetry*. Spends no looks |
| `--score_rule` | `plain` · `ortho` | **inert by algebra** — Exp. 10 was retired before any run. The flag exists and must not be used to produce numbers |
| `--seed` | 42 | changes the folds and the pairings. Same seed or the paired tests compare different comparisons |
| `--loss` (`madeeg_contrastive.py`) | `batch` · `within_mixture` · `temporal_offset` · `cross_instrument` | step C **0.3766** vs step D **0.4545**. A mistyped value **breaks** rather than falling back — and `clip_loss.py`'s self-check proves the flag really changes the objective (0.951610 vs 2.160834) |
| `--balance_pairs` (Exp. 19) | flag | S2's held-out null **0.6061 → 0.5000**, at a measured cost of 853 → 188 pairs |
| the **interpreter** | not a flag | `/opt/anaconda3` gives the same 376 decisions with floats different at 2e-5 ⟹ the md5 canaries fail. It has happened. Use `/opt/miniconda3/bin/python` |

## The canaries

Regression gates that protect numbers already reported. If one goes red, **stop and
report** — do not update the expected value.

| value | protects | how to cross it |
|---|---|---|
| `0.628491` · `1.2994` · `4.8198` | **every number in Chapter 1** (the InfoNCE arithmetic) | `python src/modules/clip_loss.py` — entirely synthetic, no dataset, no GPU |
| `0.951610` vs `2.160834` | step D's objective (within-mixture ≠ batch) | second line of the same self-check |
| md5 `2eaa926244de340d31907c6deeb04b0c` + 208/376 | the whole default own-vs-other path — the base of Exp. 7, 8, 14, 16B, 18 | rerun the reference, compare the md5 of `madeeg_ownvsother.csv` |
| every pin vs its own `hparams.yaml` | that a number cannot change because a directory landed on disk | `python sweeps/report.py --check` |
| `--check_rule` → 58/154 exact | the contrastive arm's decision rule (one implementation) | `python src/madeeg_diagnose.py --check_rule <records.csv>` |

`bash scripts/replicate/canaries.sh` crosses all of them in order and exits non-zero
if a crossable one fails. `LEVELS` picks the level: 0 needs nothing, 1 needs `torch`,
2 needs MAD-EEG, 3 is printed and **launched by hand** (GPU).

> 🔒 **`src/modules/clip_loss.py` is frozen at `bb016fd` on purpose.** It carries the
> canaries that protect every Chapter 1 number. Do not touch it.

## Hardware and environments

Three interpreters, and they are not interchangeable — the env with `torch` has no
`h5py` and cannot run `madeeg_reconstruction.py`; the one with `h5py` has no `torch`.

| for | interpreter | why |
|---|---|---|
| Chapter 1, `clip_loss.py` canary | conda `attention` (`eeg_attention` on the cluster) | torch 2.2.2 + numpy<2 + PL 1.9.5 (`requirements.txt`) |
| Chapter 2 CPU analysis (`madeeg_*.py`) | `/opt/miniconda3/bin/python` | numpy · scipy · sklearn · **h5py** · mne · librosa |
| Exp. 16A/17/18/19 | `/opt/anaconda3/bin/python` | needs `torch` **and** `laion_clap` 1.1.6 |

- Chapter 1 training requires a CUDA GPU (tested: NVIDIA L40S, SLURM). Everything
  reported here was produced from the released checkpoints on CPU, or on the cluster.
- Chapter 2 is CPU-only except the contrastive arm (steps C/D, Exp. 18/19).
- `conda activate` is a no-op in non-interactive shells (`nohup`, `sbatch`): pass the
  interpreter **by path**. `sweep_common.sh` does exactly that.
- 16 GB RAM: `--target_fs 256` runs are ~6 GB of design matrix each.

## License

CC-BY-SA 4.0 (inherited from the upstream repo). See `LICENSE`.
