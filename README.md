# EEG-based Auditory Attention Decoding in Polyphonic Music

Code and evidence for a master's thesis on decoding, from EEG, **which instrument a
listener is attending to** in real polyphonic music.

Two arms, two datasets, two chapters. They answer different questions, so their
numbers never share a table.

- **Chapter 1 — the methodological arm** (Akama et al. 2025, consumer EEG).
  Two moves, and the second is what started the investigation. First the published
  contrastive EEG↔audio baseline is reproduced: **0.875 macro / 0.865 global**,
  chance 0.25, matching the paper. Then its four learned audio encoders are replaced
  by a frozen LAION-CLAP tower, and the within-split accuracy goes **well past 90 %**
  — **0.946 macro / 0.935 global** over 8 seeds. *That* number is what had to be
  explained, and the rest of the chapter explains it: **the same CLAP architecture
  scores 0.943 leave-subject-out and 0.268 leave-song-out** (chance 0.25). It
  generalises across *people* and collapses across *songs* — and the audio alone,
  with no EEG at all, already identifies the target at **0.996**. In that dataset
  song is aliased with target, so "which instrument is attended" cannot be separated
  from "which song is playing". The learned-audio baseline behaves the same way, one
  step lower (0.781 subject-out, 0.142 song-out): the confound is in the dataset, not
  in the representation.

  *Two caveats travel with those four numbers, and neither is hidden in a script.*
  **(a)** 0.943 and 0.268 are the same *architecture*, not the same pipeline: 5 of the
  6 subject-out folds are the pre-refactor **2026-05** revision, which is why
  `RESULTS.md` carries a `vintage` column — the contrast holds, it just has to be
  quoted with the vintage ([`cap1_leave_subject_out.sh`](scripts/replicate/cap1_leave_subject_out.sh),
  "TWO MANDATORY CAVEATS"). **(b)** The thesis quotes only the **global 0.865** for the
  baseline row: a second reading of that same fold, 0.865/0.852, is still open, and
  0.865 is the figure common to both ([`cap1_within_split.sh`](scripts/replicate/cap1_within_split.sh),
  "OPEN NUMBER CONFLICT").
- **Chapter 2 — the identifiable arm** (MAD-EEG, 20-channel lab EEG).
  The same mixture is attended with **different** targets across trials, so attention
  is identifiable and stimulus identity cannot win. Everything here is a
  pre-registered test with a written threshold, and most verdicts are honest
  negatives: the linear family peaks at **0.5584** on the duo (threshold 0.5714), and
  the contrastive arm lands **below** chance at **0.3766** for a diagnosed reason —
  prior-following, not noise.

Every reported number has a null computed from the data, a positive control crossed
before the number was looked at, and a threshold written before the run.

## The documentation, in the order it is meant to be read

Seven documents, numbered. Read 1 to 4 once, in order — that is about twenty
minutes and it is the whole picture. Then 5, 6 and 7 are reference: you open them
when you need a specific number, a specific file, or a specific rule.

| # | document | what it answers | when |
|:-:|---|---|---|
| **1** | **this README** | what the project found, how to install it, how to run it, which flags change a number | first |
| **2** | [`docs/02_REPO_MAP.md`](docs/02_REPO_MAP.md) | **where everything is** — every file in the repository, one line each | when you do not know where to look |
| **3** | [`docs/03_OVERVIEW.md`](docs/03_OVERVIEW.md) | the science: the question, the four tasks and their four nulls, what each change was optimising and what it bought | before touching any code |
| **4** | [`docs/04_CODE_TOUR.md`](docs/04_CODE_TOUR.md) | how the code is put together, from the outside in: the three levels, one experiment end to end, the workhorse | before changing any code |
| **5** | [`scripts/replicate/README.md`](scripts/replicate/README.md) | the experiment table: for every row, its null, its pre-registered threshold, its verdict, its canary, its provenance | before running or citing an experiment |
| **6** | [`docs/provenance/README.md`](docs/provenance/README.md) | which file on disk proves which number | before quoting a number |
| **7** | [`docs/07_METHOD_RULES.md`](docs/07_METHOD_RULES.md) | the twelve rules every number obeys, each with the incident that caused it | when a comment cites `method rule N` |

**The numbering is in the filenames**, so `ls docs/` shows the order. Documents 1, 5
and 6 keep the name `README.md` in the directory they describe, because that is the
file GitHub renders when you browse that directory — their number is in their title.
Anything in `docs/` without a number is reference, not part of the route.

Reference material sits outside that path: [`docs/MADEEG.md`](docs/MADEEG.md) (the
Chapter 2 dataset), [`docs/CHANGES_FROM_BASELINE.md`](docs/CHANGES_FROM_BASELINE.md)
(every diff from the upstream repository) and [`docs/LEGACY.md`](docs/LEGACY.md)
(the upstream README, verbatim).

## Reference papers

> Akama T. et al. *Decoding Selective Auditory Attention to Musical Elements in
> Ecologically Valid Music Listening*. Sony CSL, 2025. — `docs/paper.pdf`.
> Upstream code: <https://github.com/JURIUENO11/Music_attention> (its README is kept
> as `docs/LEGACY.md`).

> Cantisani G. et al. *EEG-based decoding of auditory attention to a target
> instrument in polyphonic music*. WASPAA 2019. — `docs/cantisani_waspaa2019.pdf`.
> The MAD-EEG arm reproduces this protocol; its 79 F1 on duets does **not** reproduce
> from any package faithful to the declared methods (see
> `scripts/replicate/armA_paper_protocol.sh`).

## Quick start

```bash
pip install -r requirements.txt        # conda env `attention`: torch 2.2.2, numpy<2

# Chapter 1 — from the authors' RELEASED checkpoints. No training, no GPU.
bash scripts/setup_checkpoints.sh
python src/run.py replicate            # ~45 min CPU

# Chapter 2 — needs MAD-EEG (4.7 GB, not in this repo)
MADEEG_DIR=~/madeeg bash scripts/madeeg_setup.sh
python src/run.py exp --list           # every experiment and what it costs
python src/run.py exp exp13            # ~10 s, opens no EEG at all
python src/run.py exp all              # everything, in order, one log file each

# Before and after ANY change: the regression gates.
python src/run.py canaries --levels 0
```

## The launcher

`src/run.py` is the single entry point. Chapter 1 is driven from it directly;
Chapter 2 goes through `scripts/replicate/`, one script per experiment, which stay
the source of the exact commands, the expected numbers and the canaries. Every
non-interactive run is teed to its own timestamped file under `runs/logs/`.

```bash
python src/run.py                       # interactive launcher (Chapter 1)
python src/run.py replicate             # Chapter 1 from the released checkpoints
python src/run.py train --model clap    # --model baseline|clap|audio_only|eeg_only
python src/run.py test                  # evaluate a checkpoint (interactive)
python src/run.py sweep --cv song       # resumable leave-one-song-out sweep (GPU)
python src/run.py report                # re-render RESULTS.md from the manifest

python src/run.py exp --list            # the 26 experiments, in run order
python src/run.py exp exp13             # one of them (a unique prefix is enough)
python src/run.py exp cheap             # every experiment that spends no held-out look
python src/run.py exp all               # all of them, canaries first, one log each
python src/run.py canaries --levels 0   # 0 needs nothing, 1 needs torch, 2 needs MAD-EEG

python src/run.py --selftest            # assert the protocol has not drifted
```

`scripts/replicate/README.md` is the map: one row per experiment with its null, its
pre-registered threshold, its verdict, its provenance file, its canary, and whether
it runs on CPU. Read it before running anything.

Experiments that need a GPU **print their command and stop**; nothing here starts a
training run implicitly.

## Repository layout

```
.
├── README.md · RESULTS.md · results_manifest.tsv   what was found, and the pins behind it
├── docs/        the seven documents above, the two papers, and provenance/ (the evidence)
├── src/         everything that computes: run.py is the entry point, the rest are drivers
├── scripts/     everything you invoke, incl. replicate/ — one script per experiment
├── sweeps/      the resumable cross-validation sweeps and the reporting script
├── configs/     the config template and the tracklist
├── checkpoints/ · archive/ · dataset/   released weights and Chapter 1 data
└── runs/        GITIGNORED — everything a run emits: results/, replicate/, logs/
```

**Every file, with one line on what it does: [`docs/02_REPO_MAP.md`](docs/02_REPO_MAP.md)** —
that is document 2 and it is the index to the whole repository.

**MAD-EEG is not in this repo** (4.7 GB). It lives at `--madeeg_dir ~/madeeg`.

## Where the results live

`results_manifest.tsv` lists every fold behind a reported number — model,
evaluation, held-out id, run directory, code vintage. `sweeps/report.py` reads
**only** that file and renders `RESULTS.md` (**107 pinned folds** today).

Nothing is discovered by scanning the disk, so a number cannot change because a new
run directory appeared. `report.py` also checks each pin against the run's own
`hparams.yaml` / `madeeg_summary.txt` and **refuses to write the table** if one
disagrees — it has already rejected two pins written by its own author. This
replaced an aggregator that globbed and de-duplicated by a partial key: two runs of
the same fold collided, whichever the filesystem returned last silently won, and a
reported number moved from 0.142 to 0.154 with no new evidence behind it.

Chapter 2's evidence lives in **`docs/provenance/`**: the integral output of the run
that produced each number, pinned and md5-checked. `runs/` is gitignored, so from a
clean clone the run directories do **not** exist — the replication scripts say so
rather than pretending.

## Which flags change a number

This is the part that costs money when it is wrong. Every entry below is a flag whose
value **is part of the identity of a reported number**: change it and you have a
different experiment, not a different run of the same one.

### Chapter 1

| flag | values | what moves |
|---|---|---|
| `--cv_mode` | `within` · `leave_song_out` · `leave_subject_out` | **compare within one model family or the contrast is meaningless.** CLAP: 0.946 macro within · **0.268** song-out · 0.943 subject-out. Raw: 0.875 within · 0.142 song-out · 0.781 subject-out. Both survive holding out a listener and both collapse on holding out a song — this flag *is* the chapter's result |
| `--shuffle_test_mode` | `none` · `labels` · `audio_pair` | 0.865 → 0.225 → 0.383. The two controls are **not** bit-reproducible (`checkpoint_test` forces `shuffle=True`); cite them as "at chance", never as exact digits |
| `--audio_repr` | `raw` · `clap` | within 0.875 (raw) vs 0.946 (clap, 8 seeds) |
| `--objective` | `contrastive` · `classify_eeg` · `classify_audio` | `classify_audio` is the **leakage control**, and it is quoted next to the contrastive model *of its own audio branch*: 0.996 against CLAP's 0.946, 0.967 against raw's 0.875. Either way the audio alone beats the EEG model, which is the point |
| `--eeg_repr` | `raw` · `spectra` | the `spectra` numbers (0.9350 / 0.4975 / 0.4713) are **not in the manifest and have no canary**. Do not quote them as pinned |
| `$PROTO` / `PROTOCOL` | 23 flags | the real hyperparameters — **not** `configs/baseline.yaml`, whose defaults are a loose template. `python src/run.py --selftest` asserts the two copies still agree, flag by flag |

### Chapter 2 (`madeeg_reconstruction.py` unless noted)

| flag | values | what moves |
|---|---|---|
| `--train_on` | `duos_kfold` · `raw_solos` | 0.5584 vs 0.4805 on the duo decision |
| `--test_eeg` | `preprocessed` · `raw` | **the trap that bit.** With `raw_solos`, the documented command *without* `--test_eeg raw` gives **0.4870**, not the archived **0.4805**. Two numbers one digit apart for an unwritten flag |
| `--spatial` | `stereo` · `mono` · `both` | which duo set is tested: 154 stereo vs 150 decidable mono. Asserts `--train_on raw_solos --test_eeg raw` (mono duos do not exist in the preprocessed release) |
| `--eeg_clean` | `none` · `notch` · `notch_ica` | the only axis that gains: `inner_val_r` 0.0582 → 0.0673 (+16%), own-vs-other 202/376 → 208/376. Asserts `raw_solos` |
| `--filters` | `pooled` · `per_instrument` | `inner_val_r` 0.0582 → 0.1100, but that is a **metric artefact**, not a decoder gain (fair test 36/70, p 0.29). Also drops n from 154 to 150. **Forbidden** with `--own_vs_other`, by assert |
| `--estimator` | `ridge` · `shrinkage` · `cca` | `inner_val_r` 0.0582 → 0.0349 (shrinkage); own-vs-other ridge 208/376 vs CCA 209/376 (the Exp. 6 NO-GO) |
| `--target` | `mel` · `envelope` · `mag` · `flux` · `flux_mel` · `mfcc` · `mfcc_c0` | own-vs-other mel 208/376 vs flux **243/376**; duo accuracy mel 0.5584 vs envelope 0.5065 |
| `--n_mels` · `--target_fs` | 8 @ 64 (ours) · 24 @ 256 (published) | `inner_val_r` 0.0582 → 0.0468, accuracy 0.4805 → 0.4870. The published hyperparameters make it **worse**, and the two halves worsen independently |
| `--band_low/high` | 1–8 Hz (ours) | **our choice, not the paper's** — Cantisani et al. do not band-pass at all. Declared inside every summary file |
| `--own_vs_other` | flag | switches from the duo decision to held-out **solo** discrimination: n = 376, null 0.500 *exact by symmetry*. Spends no looks |
| `--score_rule` | `plain` · `ortho` | **inert by algebra** — Exp. 10 was retired before any run. The flag exists and must not be used to produce numbers |
| `--seed` | 42 | changes the folds and the pairings. Same seed, or the paired tests compare different comparisons |
| `--loss` (`madeeg_contrastive.py`) | `batch` · `within_mixture` · `temporal_offset` · `cross_instrument` | step C **0.3766** vs step D **0.4545**. A mistyped value **breaks** rather than falling back — and `clip_loss.py`'s self-check proves the flag really changes the objective (0.951610 vs 2.160834) |
| `--balance_pairs` (Exp. 19) | flag | S2's held-out null **0.6061 → 0.5000**, at a measured cost of 853 → 188 pairs |
| the **interpreter** | not a flag | `/opt/anaconda3` gives the same 376 decisions with floats different at 2e-5, so the md5 canaries fail. It has happened. Use `/opt/miniconda3/bin/python` |

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

`python src/run.py canaries` crosses all of them in order and exits non-zero if a
crossable one fails. `--levels` picks the level: 0 needs nothing, 1 needs `torch`,
2 needs MAD-EEG, 3 is printed and launched by hand (GPU).

> **`src/modules/clip_loss.py` is frozen at `bb016fd` on purpose.** It carries the
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
