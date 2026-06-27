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
- 78 stimuli/subject: 14 solo, 40 **duo**, 24 trio; ~6 s excerpts ×4 reps (~24 s/trial).
- Instruments: pop (Vx/Gt/Bass/Dr) + classical (Co/Fl/FH/Ba). Genres pop + classique.
- Files needed: `madeeg_preprocessed.hdf5` (3.7 GB) + `madeeg_preprocessed.yaml` (154 kB).
  The isolated sources (`soli`) are inside the HDF5, so `stimuli.zip` is NOT required.

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
