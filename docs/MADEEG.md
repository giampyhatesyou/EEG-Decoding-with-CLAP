# MAD-EEG arm (positive / identifiable AAD)

The original (Akama) dataset cannot support attention decoding: song == target (1:1,
identical across subjects), so "which instrument is attended" is aliased with "which
song", and the 4-channel Muse EEG carries no decodable envelope tracking (linear
reconstruction r ≈ 0 even within the training distribution; CNN under test). That arm is
the **negative / methodological** chapter.

MAD-EEG is the **positive** arm: the same mixture is attended with **different** target
instruments across trials, so attention is **identifiable** and the within-trial decision
(reconstruction vs the sources of the *same* mixture) cannot be won by stimulus identity.

## Dataset (Cantisani et al., SMM 2019 — Zenodo 4537751, CC-BY-SA)

- 20-channel research EEG @256 Hz (ch `F3..O2`, incl. occipital `O1/Oz/O2` and parietal —
  so the SpectraEEG band-power idea finally gets a fair montage), 8 subjects
  (`0001..0009` minus `0006`).
- ~6 s excerpts ×4 reps (~28 s/trial, measured).
- Files needed: `madeeg_preprocessed.hdf5` (3.7 GB) + `madeeg_preprocessed.yaml` (154 kB).
  The isolated sources (`soli`) are inside the HDF5, so `stimuli.zip` is NOT required.

**Trial counts — measured 2026-07-26 by counting both metadata files.** The two numbers
that circulate for this dataset are *both right*; they describe **different files**, and
confusing them costs you half the data or invents data you do not have.

| | `madeeg_sequences_raw.yaml` (what was **recorded**) | `madeeg_preprocessed.yaml` (what we **use**) |
|---|---|---|
| per subject | **78** — 14 solo, 40 duo, 24 trio (subject 0007: 53) | **32** — 0 solo, 20 duo, 12 trio (0007: 22) |
| total | **599** | **246** — 154 duo + 92 trio |
| spatial conditions | `stereo_lcr` **and** `mono` | `stereo_lcr` only, relabelled `spatial: stereo` |

The preprocessed release is **exactly the `stereo_lcr` half**: all 246 of its keys map onto a
raw `stereo_lcr` key by substituting `_stereo_` → `_stereo_lcr_`, 246/246, no exceptions. The
`mono` half (155 duo + 93 trio) and all 105 solos exist **only** in `madeeg_raw.hdf5`, which
`madeeg_setup.sh` does not fetch — the authors' preprocessing was never applied to them, so
using them means reproducing that preprocessing ourselves. `build_solo_trials()` already does
this for the solos (it reads the raw HDF5 + sequences + `stimuli/`), which is the precedent
if we ever want the `mono` half too.

So the dataset is **complete as downloaded**; it is the *published preprocessing* that covers
one spatial condition. Anything below refers to the preprocessed 246 unless stated.

| | measured (preprocessed) |
|---|---|
| duo trials | **154** — 20 per subject, except 0007 with 14 |
| unique duo mixtures | **18**, each presented with **2 different attended targets** (36 `stim` ids) |
| instruments attended in duos | 9 — pop `Vx Gt Bs Dr`, classical `Co Fl Ob Fh Bo` (`Ob`/`Bo`, not `Ba`) |
| genre split (duo trials) | 94 classical / 60 pop |
| trio trials | **92** — ~11–12 per subject |
| trial length | **not uniform** — 19.7 s ×49, 20.0 s ×25, 24.0 s ×130, 28.0 s ×42. (`--inspect 1` shows one trial, which is where the earlier "28 s" came from.) `soli` always has 3 rows; the third is empty for a duo. EEG and audio durations agree on all 246 trials |

The `stim` id encodes everything: `classique_morceau1_duo_CoFl_theme1_stereo_Co` = classical
piece 1, duo of Co+Fl, attended **Co**. The `_Fl` twin exists too — **all 18 duo mixtures
appear with two different targets, 18 of 18**. That is the identifiability property this
whole arm rests on, and it is visible in the data rather than only asserted here.

The trios are *not* as clean: of the 10 trio mixtures, 8 appear with all three targets, one
with two, and one — `pop_mixtape_trio_BsDrVx_theme1_stereo`, 2 trials — with a **single**
target. For those 2 trials stimulus identity predicts the answer, i.e. the Akama confound in
miniature. Two trials out of 246 will not move a number, but they should be excluded or
declared rather than discovered later. The duo design has no such case.

**Training budget** — counted by `src/datasets/madeeg_contrastive_dataset.py`, tiling each
trial with 3 s windows at 1 s stride (1 to 26 windows per trial, since trials differ in
length):

| | trials | windows | EEG |
|---|---:|---:|---:|
| duos only | 154 | **3276** | 60.1 min |
| duos + trios | 246 | **5235** | 96.1 min |
| per subject (duos + trios) | ~31 | ~650 | ~12 min |

Small. With the CLAP backbone frozen only the projection head and the EEG encoder train,
which is what makes this arguable at all — but it is also why the linear anchor may well
win, and that has to be reported as such rather than explained away.

Roughly doubling this is *possible* but not free: the `mono` half (155 duo + 93 trio) sits
unprocessed in `madeeg_raw.hdf5`. Pooling two spatial conditions is a methodological choice,
not a free data top-up — spatial separation is itself known to help auditory attention, so
the condition would have to be modelled or reported, never silently merged.

### Verified HDF5/metadata schema (from `tutorial-MAD-EEG.ipynb`)

```
data[subj][stim]['response'] : EEG (20, M) @ 256 Hz
data[subj][stim]['soli']     : isolated sources (3, N) @ 44100 (3rd row empty if duo),
                               ordered as metadata[subj][stim]['instruments']
data[subj][stim]['stimulus'] : stereo mixture (2, N) @ 44100
metadata[subj][stim] = {target, instruments[list], ensemble(duo/trio), spatial,
                        genre, song, theme, eeg_info{sfreq:256}, wav_info{sfreq:44100}}
```

## Workflow

```bash
# 1. download (baldo/edu02), ~3.7 GB
MADEEG_DIR=~/madeeg bash madeeg_setup.sh

# 2. verify the schema on the real files before any run
cd src && python madeeg_reconstruction.py --madeeg_dir ~/madeeg --inspect 3

# 3. linear backward-model AAD baseline on duos (the reproducible anchor)
python madeeg_reconstruction.py --madeeg_dir ~/madeeg --ensemble duo \
    --training_date madeeg_ridge_duo --log_dir ../results
```

## Method (`src/madeeg_reconstruction.py`)

Per subject, k-fold (default 5) over duo (and/or trio) trials. Ridge backward model
reconstructs the **attended** source envelope from EEG (band-pass 1–8 Hz, downsample to
64 Hz, lags 0–250 ms). At test: argmax of Pearson r between the reconstruction and each
**present** source envelope. Chance = 1/n_present (0.50 duo, 0.33 trio).

- **Target to beat:** the paper's linear stimulus-reconstruction reaches r_attended >
  r_unattended in **>78 %** of duet tests. Recovering that number validates the loader.
- Method validated on synthetic data (EEG that truly tracks the attended source is
  recovered at ~100 %), so a chance result on real data means genuine absence of signal,
  not a bug.

## Next (extension, after the linear anchor holds)

Bring the repo's contrastive / CLAP / SpectraEEG framework here, with the **correct**
negatives = the *competing stems of the same mixture* (within-stimulus contrast), and the
within-trial attended-vs-unattended evaluation. Honest risk: 8 subjects is small → linear
may beat deep; report it as such.
