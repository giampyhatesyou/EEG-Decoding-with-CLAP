# 2 · Repository map — every file, one line each

*Reading path: [1 README](../README.md) → **2 you are here** → [3 OVERVIEW](03_OVERVIEW.md) → [4 CODE_TOUR](04_CODE_TOUR.md) → [5 replicate](../scripts/replicate/README.md) → [6 provenance](provenance/README.md) → [7 METHOD_RULES](07_METHOD_RULES.md).*

Open this when you know *what* you want and not *where* it is. Every tracked file
outside `docs/provenance/`, `archive/` and `dataset/` is listed; those three are
bulk directories and are described at the bottom instead.

The one rule that explains the layout: **`src/` computes, `scripts/` invokes,
`docs/` explains, and nothing generated is tracked.** Anything a run produces
lands in `runs/`, which is gitignored — the only run outputs in the repository are
the copies under `docs/provenance/`.

---

## Root — what a number is, and how to install

| file | what it is |
|---|---|
| `README.md` | The entry point: what the project found, how to install and run it, which flags change a number, the canaries, the three environments. |
| `RESULTS.md` | **Generated, never edited by hand.** The Chapter 1 tables, rendered from the manifest by `sweeps/report.py`. |
| `results_manifest.tsv` | **The one source for every Chapter 1 number**: one row per pinned fold, 107 of them. `report.py` reads only this and refuses to write if a pin disagrees with its run's own `hparams.yaml`. |
| `requirements.txt` | The Chapter 1 stack, pinned, with the reason for each pin written next to it (torch 2.2.2, numpy 1.26.4, Lightning 1.9.5, torchmetrics 0.11.4). |
| `.gitignore` | Notable for one line: `/runs/` — the whole run tree is excluded, which is why `docs/provenance/` exists. |
| `LICENSE` | CC-BY-SA 4.0, inherited from the upstream repository. |

## `docs/` — the explanations, in reading order

The four documents on the reading path carry their number in the filename, so
`ls docs/` shows the order. Documents **1**, **5** and **6** are called `README.md`
on purpose: they sit in the directory they describe and GitHub renders them when
you browse it, which is worth more than a prefix. Unnumbered files here are
reference, not part of the path.

> **Renamed on 27 August 2026.** Three of them were `OVERVIEW.md`, `CODE_TOUR.md`
> and `METHOD_RULES.md`; `02_REPO_MAP.md` is new that day and has no earlier name, so
> its history starts there. Notes and reports dated before then use the old names;
> `git log --follow` traces the three renamed files across the rename.

| file | what it is |
|---|---|
| `02_REPO_MAP.md` | **2** · this file. |
| `03_OVERVIEW.md` | **3** · the science: the question, the four tasks with their four nulls, both chapters, the genealogy of every change with what it was optimising and what it bought. Read before any code. |
| `04_CODE_TOUR.md` | **4** · the code from the outside in: the three levels, one experiment traced end to end, the workhorse and its six modes, real drivers versus orchestrators, the two audio towers and what each costs. |
| `07_METHOD_RULES.md` | **7** · the twelve rules every number obeys, each with the incident that produced it. Source comments cite these as `method rule N`. |
| `MADEEG.md` | Reference · the Chapter 2 dataset: schema, what is in which release, the loader, the traps. |
| `CHANGES_FROM_BASELINE.md` | Reference · every difference from the upstream Akama repository, file by file, including what is byte-identical. |
| `LEGACY.md` | Reference · the **upstream** README, kept verbatim. Its internal links point at the upstream layout. |
| `paper.pdf` · `cantisani_waspaa2019.pdf` | The two papers the two arms build on. |
| `model_architecture.png` | The upstream architecture figure. |
| `provenance/` | **6** · the run outputs behind every Chapter 2 number, plus `README.md`, which maps each number to the file that produced it. |

## `src/` — the code that computes

### Chapter 1 — the Akama arm (needs `torch`)

| file | what it does |
|---|---|
| `run.py` | **The single entry point.** `replicate · train · test · sweep · report · exp · canaries`, an interactive launcher, and `PROTOCOL` — the 23 fixed Akama flags, kept in sync with `sweeps/sweep_common.sh` by `--selftest`. Every non-interactive run is teed to `runs/logs/`. |
| `main.py` | Training entry point: builds the encoders according to `--audio_repr` / `--eeg_repr` / `--objective` and hands them to the Lightning module. |
| `checkpoint_test.py` | Evaluation from a checkpoint; writes the per-class breakdown the manifest pins. Forces `shuffle=True` under `--shuffle_test_mode`, which is why two negative controls are not bit-reproducible. |
| `stimulus_reconstruction.py` | The Chapter 1 backward model — the ancestor of the Chapter 2 workhorse. **No experiment runs it and it produces no reported number**; kept because it is where the method comes from. |
| `modules/clip_loss.py` | The InfoNCE arithmetic behind **every** Chapter 1 number. **Frozen at `bb016fd`**: it carries the canaries `0.628491 / 1.2994 / 4.8198` and step D's `0.951610` vs `2.160834`. Do not touch. |
| `modules/contrastive_learning.py` | The LightningModule: loss wiring, audit hooks, the evaluation matrix and the breakdown. |
| `modules/supervised_classification.py` | The two diagnostic controls (`classify_eeg`, `classify_audio`) in a **separate** module, so the contrastive path stays byte-identical. Its docstring carries the reading of both controls. |
| `models/sample_cnn2d_eeg.py` | The Akama 2D-CNN, used both as the EEG encoder and as each raw audio encoder. 116,120 trainable parameters, independent of input length. |
| `models/clap_encoder.py` | Frozen LAION-CLAP tower plus the trainable `Linear(512,256)-GELU-Linear(256,100)` head (157,028 parameters); resamples 44.1 → 48 kHz and embeds under `no_grad`. |
| `models/spectra_eeg.py` | Alternative EEG front end (raw branch + band power). Exploratory: **no pin, no canary, nothing quotable**, and never run on MAD-EEG. |
| `models/model.py` | The 11-line `nn.Module` base class the encoders subclass. |
| `datasets/preprocessing_eegmusic_dataset.py` | The Chapter 1 dataset: sliding windows over the EEG and the four stems, plus the `cv_mode` / `cv_held_out_id` routing that defines within / leave-song-out / leave-subject-out. |
| `preprocessing/transform.py` | One-off conversion of the raw experiment exports into the dataset the class above reads (phase assignment, task ids, noise rejection). Not part of any run. |

### Chapter 2 — the MAD-EEG arm (mostly no `torch`)

| file | what it does |
|---|---|
| `madeeg_reconstruction.py` | **The workhorse, ~1,700 lines.** Ridge / shrinkage / CCA stimulus reconstruction, own-vs-other, the alpha index, the four axes, arm A, and Experiments 4 · 6 · 7 · 8 · 9. Every flag here is part of the identity of a number. |
| `madeeg_diagnose.py` | The post-mortem, and it imports no torch: prior-following, paired McNemar, `--check_rule` (which reproduces step C's 58/154 offline), `--demo`. It owns the **single** implementation of the decision rule. |
| `madeeg_contrastive.py` | The contrastive arm on MAD-EEG: steps A–D and Experiments 18/19, behind `--loss {batch,within_mixture,temporal_offset,cross_instrument}`. GPU. |
| `madeeg_spectral_attention.py` | Experiments 11 and 12: band power plus LDA on the attended spectral register. Its positive control runs first and the script exits if it fails. |
| `madeeg_stem_separability.py` | The audio-only separability gate: Experiment 13, the noise-512 negative control, and Experiment 17. Two of its outputs carry pinned md5s. |
| `madeeg_leakage_audit.py` | Arm D: the window-versus-trial cross-validation audit on balanced pseudo-labels. The interpretation rule is written in the code before the run. |
| `madeeg_exp14_tracking.py` | Experiment 14 — orchestrator: runs `madeeg_reconstruction.py` as a subprocess and judges MFCC-13 and mel-64 against the 208/376 bar. |
| `madeeg_same_diff_melody.py` | The same/diff-melody split of an arm A run, from its records plus the session metadata. Descriptive, no new looks; exits non-zero if the split is not 89/61. |
| `madeeg_exp15_differential.py` | Experiment 15 — orchestrator: the attended-minus-unattended differential under MFCC, with its minimum detectable difference. |
| `madeeg_exp16a_clap_extract.py` | Stage 1 of Experiments 16A/17: CLAP embedding extraction, writing a manifest with the checkpoint checksum and library versions. |
| `madeeg_exp16b_ccaviews.py` | Experiment 16B — orchestrator: band power as a second view inside the multi-view CCA. |
| `models/cca_multiview.py` | Regularised multi-view CCA: features and algebra only, no I/O. Its self-check is canary L0-c. |
| `models/alpha_lateralization.py` | The alpha laterality index and `inject_lateralized_alpha()`, which is the positive control for Experiment 6[H]. Nothing is fitted. |
| `datasets/madeeg_contrastive_dataset.py` | MAD-EEG for steps C/D: the **preprocessed** release, duo and trio trials. |
| `datasets/madeeg_solo_matchmismatch.py` | Match-mismatch sampling over the **solos**, which exist only in the raw release. Carries the sampling test and the `--balance_pairs` postcondition. |

### `src/utils/` — small, boring, shared

| file | what it does |
|---|---|
| `paths.py` | Where every artefact goes (`runs/results/<training_date>/`), plus `EEG_*` environment overrides and DataLoader worker sizing. |
| `logger.py` | The logger; computes its directory relative to the project root, not the cwd. |
| `yaml_config_hook.py` | Loads `configs/baseline.yaml` into argparse defaults. |
| `file_helpers.py` | Pickle / CSV / JSON readers and writers. |
| `time_helper.py` | Timezone-aware timestamp parsing, 15 lines. |
| `__init__.py` (in `utils/`, `models/`, `modules/`, `datasets/`, `preprocessing/`) | Package re-exports. They pull in the training dependencies, which is **why `madeeg_reconstruction.py` imports its two repo modules by file path instead of as packages** — that is what lets it run in an environment with no torch. |

## `scripts/` — the things you invoke

| file | what it does |
|---|---|
| `replicate.sh` | Reproduces **every Chapter 1 number** from the released checkpoints, in one command. `PHASES="within loso sanity report"` selects the phases; it ends by re-rendering `RESULTS.md`. |
| `train.sh` | Trains one model on one split. Takes the protocol from `sweep_common.sh` so it cannot drift. |
| `setup_checkpoints.sh` | Unpacks the paper checkpoints from `archive/*.7z`. Run once, before `replicate.sh`. |
| `madeeg_setup.sh` | Fetches the MAD-EEG preprocessed release. Says what it did **not** fetch rather than guessing. |
| `replicate/` | **5** · one script per experiment, listed just below. Each header states what it measures, its null, its pre-registered threshold, its verdict, its cost, whether it spends a held-out look, its GPU need, its expected numbers, its canary and its provenance file. **The headers are the source; `run.py exp --list` reads the table back out of them.** |

### `scripts/replicate/` — the 26 experiments, plus two batches

Names only here, because the numbers belong to doc **5** and to each header. Run
order is the one below; `python src/run.py exp --list` prints it live, with what
each costs.

| script | experiment |
|---|---|
| `cap1_within_split.sh` | Chapter 1, within split: the baseline reproduction **and** the CLAP model that took it past 90 %. |
| `cap1_leave_song_out.sh` | Chapter 1, leave-song-out: the collapse, on both audio branches. |
| `cap1_leave_subject_out.sh` | Chapter 1, leave-subject-out: the same model survives unseen listeners. |
| `exp13_stem_separability.sh` | Exp. 13 — audio-only separability gate. Opens no EEG file; ~10 s. |
| `exp06_ovo_gate.sh` | Exp. 6[G] — own-vs-other in four configurations; produces the project's reference 208/376 and its md5. |
| `exp07_ovo_config_sweep.sh` | Exp. 7 — eight configurations against that reference. |
| `exp08_flux_ovo.sh` | Exp. 8 — spectral flux on own-vs-other: **the only pre-registered positive**. |
| `exp14_mfcc_tracking.sh` | Exp. 14 — MFCC-13 and mel-64 on the tracking side. |
| `exp16b_ccaviews.sh` | Exp. 16B — band power as a second CCA view. |
| `exp17_clap_separability.sh` | Exp. 17 — CLAP in the gate, criterion remade dimensionality-invariant. |
| `exp16a_clap_separability.sh` | Exp. 16A — its CLAP stage **was never executed** (it would have to download a 1.74 GiB checkpoint), but its noise-512 negative control ran and **falsified the Exp. 13 criterion**. Kept for the record; to judge CLAP use Exp. 17. |
| `ridge_anchors.sh` | The three linear anchors on the duo decision. |
| `axes_paper_protocol.sh` | The four axes of the published protocol, plus axis 0. |
| `exp04_duo_mono.sh` | Exp. 4 — the mono half of the duos, the only confirmatory look. |
| `exp06_alpha_paired.sh` | Exp. 6[H] — paired lateralised alpha, with its injection control. |
| `exp09_flux_attention.sh` | Exp. 9 — does the flux gain transfer to attention? Plus the audio-only diagnostic that explains why not. |
| `exp11_spectral_register_stereo.sh` · `exp12_spectral_register_mono.sh` | Exp. 11 and 12 — the attended spectral register, stereo then fresh mono. |
| `exp15_mfcc_differential.sh` | Exp. 15 — the attentional differential under MFCC, with its detectability floor. |
| `armA_paper_protocol.sh` | Arm A — protocol parity with Cantisani et al.: the published 79 does not reproduce. |
| `armA_same_diff_melody.sh` | Arm A — the same/diff-melody split of those runs: where the published advantage lives in frequency. Rewritten 2026-09-01 after the original was lost. |
| `armD_leakage_audit.sh` | Arm D — the window-versus-trial leakage audit. |
| `stepC_contrastive_clap.sh` · `stepD_within_mixture.sh` | Steps C and D — the contrastive arm and the within-mixture loss. GPU steps are printed, not launched. |
| `exp18_matchmismatch.sh` · `exp19_matchmismatch.sh` | Exp. 18 and 19 — match-mismatch on the solos, then the same at twelve times the compute budget. |
| `exp05_side_bandpower.sh` | Exp. 5 — **archived without a number**: never run as written, kept so its absence cannot read as an oversight. |
| `canaries.sh` | Every regression gate in the project, in four levels. The first thing to run after any diff. |
| `run_all_cheap.sh` | The six experiments that spend no held-out look, cheapest first (~45–70 min). |
| `README.md` | Doc **5**: the navigation table — null, threshold, verdict, provenance and canary for every row. |

## `sweeps/` — the cross-validation machinery

| file | what it does |
|---|---|
| `sweep_common.sh` | Everything the two sweeps must agree on, in one place: the conda environment, `$PROTO` (the 23 fixed flags), where artefacts go, and the resume logic. `run.py --selftest` asserts it still matches `PROTOCOL`. |
| `sweep_song_out.sh` · `sweep_subject_out.sh` | The two resumable sweeps, 4 models × 20 songs and 4 models × subjects. The pair is the chapter's result: the same model survives unseen people and collapses on unseen songs. |
| `report.py` | Renders `RESULTS.md` from the manifest and **verifies every pin against its run's own `hparams.yaml`**, writing nothing if one disagrees. `--check` verifies without writing. It replaced an aggregator that globbed the disk and once moved a reported number by itself. |

## `configs/` and the bulk directories

| path | what it is |
|---|---|
| `configs/baseline.yaml` | The config template and the `audio_repr` / `objective` / `cv_mode` switches, each documented. **Loose defaults: the real protocol is `$PROTO` / `PROTOCOL`, not this file.** |
| `configs/tracklist.csv` | The 30 songs of the Akama dataset with genre, source and licence. |
| `checkpoints/` | The paper checkpoints (`.7z` tracked in `archive/`, the unpacked `.ckpt` gitignored) plus a README on their provenance. |
| `archive/` | The upstream release artefacts, including the checkpoint archives `setup_checkpoints.sh` unpacks. |
| `dataset/` | The Chapter 1 EEG and per-stem audio. |
| `docs/provenance/` | 61 run-output files plus their index. See doc **6**. |
| `runs/` | **Gitignored.** `results/` (per-run CSVs, summaries, `hparams.yaml`), `replicate/` (what the replication scripts write), `logs/` (one file per launcher run). From a clean clone this directory does not exist. |

---

## Three things that are not where you would guess

1. **The protocol lives in two files on purpose.** `PROTOCOL` in `src/run.py` and `$PROTO` in `sweeps/sweep_common.sh` are the same 23 flags for Python and for shell. They are not allowed to drift: `python src/run.py --selftest` compares them flag by flag and is canary L0-a.
2. **`madeeg_contrastive.py` imports its decision rule from `madeeg_diagnose.py`**, the file that has no torch. That is why the rule has one implementation and why `--check_rule` can rerun step C's 154 decisions offline.
3. **Four drivers compute nothing.** `madeeg_exp14/15/16b` and parts of `exp16a` are orchestrators: they run `madeeg_reconstruction.py` as a subprocess and read its summary back. Looking for the arithmetic inside them is a waste of time — see doc **4** §4.
