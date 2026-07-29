# CHANGED(baseline): NEW FILE — POSITIVE thesis arm. Stimulus-reconstruction (backward
#   model) AAD on the MAD-EEG dataset (Cantisani et al., SMM 2019; Zenodo 4537751).
#   Unlike the original (Akama) dataset, MAD-EEG presents the SAME mixture with DIFFERENT
#   attended instruments across trials -> within-stimulus contrast -> attention decoding
#   is IDENTIFIABLE. The decision is internal to one trial (reconstruction vs the present
#   sources of the SAME mixture), so it cannot be won by stimulus identity.
#
#   Self-contained (does not import the original-dataset pipeline). Deps: h5py, pyyaml,
#   scipy, sklearn, numpy, librosa (mel target) — all already in the env.
#
# Schema (verified from tutorial-MAD-EEG.ipynb + the real files):
#   data[subj][stim]['response'] : EEG (20, M) @ 256 Hz  (ch F3..O2, incl. occipital)
#   data[subj][stim]['soli']     : isolated sources (3, N) @ 44100 (3rd row empty if duo),
#                                  ordered as metadata[subj][stim]['instruments']
#   metadata[subj][stim] : {target, instruments[list], ensemble(duo/trio), spatial, ...}
#   NOTE: the preprocessed set contains only duo (154) + trio (92) trials — NO solos.
#   HDF5 arrays use a non-standard float layout -> read via read_f64 (NATIVE_DOUBLE).
#
# Method: per subject, k-fold over duo (and/or trio) trials. Backward model reconstructs
#   the ATTENDED source representation from EEG (band-pass 1-8 Hz, downsample to 64 Hz,
#   lags 0-250 ms). Representation = broadband envelope (--target envelope) OR log-mel
#   spectrogram (--target mel, default; multi-output ridge). At test: argmax over present
#   sources of the (mean per-band) Pearson r between reconstruction and source rep.
#   Chance = 1/n_present (0.50 duo). Target to beat (paper, duets): ~78%.
#   For music, the mel target matters: instruments can share rhythm (same envelope) but
#   differ spectrally — the broadband envelope discards exactly that.

import os
import argparse
import numpy as np
import h5py
import yaml
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt, resample
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import f1_score

EEG_FS = 256
AUDIO_FS = 44100


def read_f64(dset):
    """Read an HDF5 dataset into float64, forcing HDF5's own type conversion.
    MAD-EEG stores arrays in a non-standard float layout that h5py cannot auto-map
    ('Insufficient precision...'); NATIVE_DOUBLE as the memory type converts on read."""
    out = np.empty(dset.shape, dtype=np.float64)
    dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, out, h5py.h5t.NATIVE_DOUBLE)
    return out


# --------------------------------------------------------------------------- #
# backward-model primitives
# --------------------------------------------------------------------------- #
def bandpass(x, fs, low, high):
    if not high or high <= 0:
        return x
    nyq = 0.5 * fs
    b, a = butter(4, [max(low, 0.01) / nyq, min(high, 0.99 * nyq) / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def band_pearson(A, B):
    """Mean per-band Pearson r between two (T, n_bands) representations."""
    return float(np.mean([pearson(A[:, b], B[:, b]) for b in range(A.shape[1])]))


def ridge_fit(X, Y, lam, shrinkage=False):
    """Closed-form ridge, multi-output: W = (X^T X + lam I)^-1 X^T Y. Y may be (T,) or (T,k).

    AXIS 3 -- `shrinkage=True` swaps the penalty for the paper's normalized reverse
    correlation (WASPAA 2019 Sec. 3.2): G(f) = C_RR^-1 C_RS with the autocorrelation
    shrunk as C'_RR = (1-lam) C_RR + lam * nu * I, where nu is the average eigenvalue of
    C_RR -- i.e. trace/d, so the penalty is scaled by the data instead of being absolute.
    lam is then a smoothing parameter in [0, 1] and the paper's grid search published 0.1.

    Multi-output stays faithful to "each feature f is reconstructed independently of the
    others": C_RR does not depend on f, so solving the k right-hand sides jointly gives
    exactly the k independent per-feature solutions. What the joint form would couple is
    the CHOICE of lam (one band_pearson averaged over bands) -- and under shrinkage there
    is nothing to choose, lam is fixed at the published value.
    """
    d = X.shape[1]
    C = X.T @ X
    if shrinkage:
        nu = np.trace(C) / d
        return np.linalg.solve((1.0 - lam) * C + lam * nu * np.eye(d), X.T @ Y)
    return np.linalg.solve(C + lam * np.eye(d), X.T @ Y)


def lagged_design(eeg, n_lags):
    c, T = eeg.shape
    padded = np.pad(eeg, ((0, 0), (0, n_lags)), mode="edge")
    return np.concatenate([padded[:, tau:tau + T] for tau in range(n_lags + 1)], axis=0).T


def _inner_split(n, rng):
    idx = np.arange(n); rng.shuffle(idx)
    nv = max(1, n // 5)
    return idx[nv:].tolist(), idx[:nv].tolist()


# --------------------------------------------------------------------------- #
# MAD-EEG feature extraction
# --------------------------------------------------------------------------- #
def process_eeg(resp, target_fs, band, clamp):
    """EEG (20, M)@256 -> band-pass -> per-channel RobustScaler+clamp -> resample to target_fs."""
    x = np.asarray(resp, dtype=np.float64)
    x = bandpass(x, EEG_FS, band[0], band[1])
    for i in range(x.shape[0]):
        x[i] = RobustScaler().fit(x[i].reshape(-1, 1)).transform(x[i].reshape(-1, 1)).reshape(-1)
    x = np.clip(x, -clamp, clamp)
    L = int(round(x.shape[1] * target_fs / EEG_FS))
    return resample(x, L, axis=1), L


def n_repr_bands(kind, n_mels, target_fs, src_fs=AUDIO_FS):
    """Rows of the representation source_repr() returns. For "mag" the STFT geometry fixes
    it (n_fft = 2*hop -> hop+1 bins), so it cannot be chosen with --n_mels."""
    if kind == "mag":
        return max(1, int(round(src_fs / target_fs))) + 1
    return 1 if (kind == "envelope" or n_mels <= 1) else n_mels


def source_repr(src, src_fs, out_len, target_fs, band, compression, n_mels, kind="mel"):
    """Isolated source (1-D @44100) -> representation (n_bands, out_len), band-matched to EEG.
    n_mels<=1 -> broadband Hilbert envelope; else -> log-mel spectrogram (per-band temporal
    series). Each band is band-passed to the EEG band and z-scored.

    AXIS 4 -- kind="mag" is the paper's third descriptor, the linear magnitude spectrogram:
    same STFT geometry the paper states (hop = fs_audio/fs_eeg, window = 2*hop) but with
    linear frequency bins instead of a mel filterbank, so MEL-vs-MAG isolates the
    perceptual warping alone. The log and the per-band band-pass + z-score are kept
    identical to the mel branch, so the two are comparable inside THIS pipeline; the paper
    does not say whether it logs its spectrograms.
    """
    x = np.asarray(src, dtype=np.float64)
    if not np.any(x):
        return np.zeros((n_repr_bands(kind, n_mels, target_fs, src_fs), out_len))
    if kind == "mag":
        import librosa
        hop = max(1, int(round(src_fs / target_fs)))
        S = np.abs(librosa.stft(y=x, n_fft=2 * hop, hop_length=hop))     # (hop+1, frames)
        rep = np.log(S + 1e-6)
        if rep.shape[1] != out_len:
            rep = resample(rep, out_len, axis=1)
    elif n_mels <= 1:
        env = np.abs(hilbert(x))
        if compression and compression != 1.0:
            env = np.power(np.maximum(env, 0.0), compression)
        nyq = 0.5 * src_fs
        b, a = butter(4, min(20.0, 0.99 * nyq) / nyq, btype="low")
        env = filtfilt(b, a, env)
        rep = resample(env, out_len)[None, :]                       # (1, out_len)
    else:
        import librosa
        hop = max(1, int(round(src_fs / target_fs)))
        n_fft = int(2 ** np.ceil(np.log2(2 * hop)))
        S = librosa.feature.melspectrogram(y=x, sr=int(src_fs), n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        rep = np.log(S + 1e-6)                                       # (n_mels, frames)
        if rep.shape[1] != out_len:
            rep = resample(rep, out_len, axis=1)
    out = np.empty((rep.shape[0], out_len))
    for bnd in range(rep.shape[0]):
        r = bandpass(rep[bnd], target_fs, band[0], band[1])
        out[bnd] = (r - r.mean()) / (r.std() + 1e-8)
    return out


def build_trial(data, meta, subj, stim, target_fs, band, clamp, compression, n_mels, kind="mel"):
    m = meta[subj][stim]
    instruments = list(m["instruments"])
    n_present = len(instruments)
    target_idx = instruments.index(m["target"])
    eeg, L = process_eeg(read_f64(data[subj][stim]["response"]), target_fs, band, clamp)
    soli = read_f64(data[subj][stim]["soli"])
    reps = np.stack([source_repr(soli[i], m["wav_info"]["sfreq"], L, target_fs, band, compression, n_mels, kind)
                     for i in range(n_present)], axis=0)             # (n_present, n_bands, L)
    return eeg, reps, target_idx, n_present, m["ensemble"]


def _proc_eeg_segment(seg, target_fs, clamp):
    """Per-channel RobustScaler + clamp + resample. seg is already band-passed."""
    x = seg.astype(np.float64).copy()
    for i in range(x.shape[0]):
        x[i] = RobustScaler().fit(x[i].reshape(-1, 1)).transform(x[i].reshape(-1, 1)).reshape(-1)
    x = np.clip(x, -clamp, clamp)
    L = int(round(x.shape[1] * target_fs / EEG_FS))
    return resample(x, L, axis=1), L


# FIFF channel kinds, as stored in madeeg_raw.yaml -> chs[i]['kind'].
_FIFF_KIND = {2: "eeg", 3: "stim", 202: "eog", 302: "emg", 402: "ecg", 502: "misc"}


def clean_continuous(x, raw_chs, chs_meta, mode, subj, seed=42, diag=None):
    """AXIS 1 -- the EEG preprocessing the paper describes and our raw path never had.

    WASPAA 2019 Sec. 2, verbatim: "the 50 Hz power-line interference was removed using a
    notch filter and EOG/ECG artifacts were detected and removed using independent
    component analysis (ICA)". That is how the authors' own `preprocessed` release was
    made, so the raw path is the only place this is missing.

      mode="notch"      50 Hz notch only -- separates the notch from the ICA, so a change
                        can be attributed to one or the other
      mode="notch_ica"  notch, then remove the ICs that track the reference EOG/ECG

    TRAP, read off madeeg_raw.yaml and not guessable from the labels: the channel NAMED
    "ECG" is typed MISC (kind 502). The actual references are AUX1 (kind 402 = ECG) and
    AUX3 (kind 202 = EOG). Typing by name hands ICA no reference at all, and the cleaning
    then runs, reports success and removes nothing. Types come from `kind`, never the name.

    Positive control, written into `diag` and from there into the run summary: the
    correlation between the frontal EEG channels and the EOG reference, inside the 1-8 Hz
    analysis band, measured before and after. Ocular artefacts dominate that band, so if
    the ICA did what it claims this number has to fall.
    """
    import mne
    mne.set_log_level("ERROR")
    types = [_FIFF_KIND.get(c["kind"], "misc") for c in chs_meta]
    raw = mne.io.RawArray(x.copy(), mne.create_info(list(raw_chs), EEG_FS, types), verbose=False)
    raw.notch_filter(50.0, picks=["eeg", "eog", "ecg"], verbose=False)
    if mode == "notch":
        if diag is not None:
            diag[subj] = "notch only"
        return raw.get_data()

    eeg_pick = [i for i, t in enumerate(types) if t == "eeg"]
    eog_pick = [i for i, t in enumerate(types) if t == "eog"]
    frontal = [i for i in eeg_pick if raw_chs[i].startswith("F")]

    def eog_coupling(arr):
        """Mean |r| between the frontal EEG channels and the EOG reference, 1-8 Hz."""
        if not eog_pick or not frontal:
            return float("nan")
        ref = bandpass(arr[eog_pick[0]], EEG_FS, 1.0, 8.0)
        return float(np.mean([abs(pearson(bandpass(arr[i], EEG_FS, 1.0, 8.0), ref)) for i in frontal]))

    before = eog_coupling(raw.get_data())
    # ICA is unstable on slow drifts, so it is fitted on a 1 Hz high-passed copy and
    # applied to the notched data -- mne's documented recipe, not a choice of ours.
    fit_raw = raw.copy().filter(l_freq=1.0, h_freq=None, picks=["eeg", "eog", "ecg"], verbose=False)
    ica = mne.preprocessing.ICA(n_components=0.999, method="fastica",
                                random_state=seed, max_iter="auto", verbose=False)
    ica.fit(fit_raw, picks="eeg", decim=3, verbose=False)
    bad_eog, sc_eog = ica.find_bads_eog(fit_raw, verbose=False)
    try:
        bad_ecg, sc_ecg = ica.find_bads_ecg(fit_raw, method="correlation", verbose=False)
    except (RuntimeError, ValueError):                      # no usable ECG reference
        bad_ecg, sc_ecg = [], np.zeros(ica.n_components_)
    ica.exclude = sorted(set(bad_eog) | set(bad_ecg))
    ica.apply(raw, verbose=False)
    after = eog_coupling(raw.get_data())
    if diag is not None:
        diag[subj] = (f"ICs={ica.n_components_} excluded={len(ica.exclude)} "
                      f"(eog={sorted(bad_eog)} max|r|={np.max(np.abs(sc_eog)):.2f} · "
                      f"ecg={sorted(bad_ecg)} max|r|={np.max(np.abs(sc_ecg)):.2f})  "
                      f"frontal-EOG coupling 1-8 Hz: {before:.3f} -> {after:.3f}")
    return raw.get_data()


def raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band, clean="none", raw_info=None, seed=42,
                       diag=None):
    """Read the continuous raw EEG for a subject, reorder the 20 EEG channels to the
    preprocessed order, band-pass once. Returns (20, N).

    AXIS 1 -- `clean` inserts the paper's EEG preprocessing BEFORE the band-pass, on the
    full 30-channel record (the reference EOG/ECG channels are needed and are dropped by
    the reorder). "none" is the current behaviour and the default."""
    x = read_f64(raw_f[subj])
    if clean != "none":
        x = clean_continuous(x, raw_chs, raw_info[subj]["chs"], clean, subj, seed, diag)
    idx = [raw_chs.index(c) for c in pre_chs]
    return bandpass(x[idx], EEG_FS, band[0], band[1])


def solo_instrument(key):
    """`classique_morceau1_solo_Co_theme2_mono` -> `Co`, the instrument played in that solo.
    Used by AXIS 2 to group the training solos into one decoder per instrument."""
    parts = key.split("_")
    return parts[parts.index("solo") + 1]


def build_solo_trials(eeg_full_bp, sq, stimuli_dir, subj, target_fs, band, clamp, compression, n_bands,
                      kind="mel"):
    """Paper protocol training pairs: each solo stimulus repetition is cut from the
    band-passed continuous raw EEG at its n_ech sample index and paired with the solo
    source audio (from stimuli/). Returns trials
    [eeg(20,L), reps(1,n_bands,L), 0, 1, 'solo', instrument]."""
    import librosa
    N = eeg_full_bp.shape[1]
    trials = []
    for k in sq[subj]:
        if "solo" not in k:
            continue
        n_ech = sq[subj][k].get("n_ech", []); wavs = sq[subj][k].get("wav_files", [])
        for i in range(min(len(n_ech), len(wavs))):
            wpath = os.path.join(stimuli_dir, wavs[i])
            if not os.path.exists(wpath):
                continue
            audio, sr = librosa.load(wpath, sr=None, mono=True)
            seg_len = int(round(len(audio) / sr * EEG_FS))
            a = int(n_ech[i]); b = a + seg_len
            if b > N or seg_len < EEG_FS:
                continue
            eeg_proc, L = _proc_eeg_segment(eeg_full_bp[:, a:b], target_fs, clamp)
            rep = source_repr(audio, sr, L, target_fs, band, compression, n_bands, kind)
            trials.append([eeg_proc, rep[None, ...], 0, 1, "solo", solo_instrument(k)])
    return trials


def build_duo_trial_raweeg(eeg_full_bp, soli, n_ech, target_idx, n_present, src_fs,
                           target_fs, band, clamp, compression, n_bands, kind="mel"):
    """Test trial with RAW EEG (same pipeline as the solos -> no preprocessing mismatch).
    The 4 repetitions are cut from the raw EEG at their n_ech onsets (each rep length =
    soli_len/4 in EEG samples) and concatenated to match the 4-rep preprocessed 'soli'
    audio, which provides the isolated sources for the AAD decision."""
    rep_len = int(round((soli.shape[1] / max(1, len(n_ech))) / src_fs * EEG_FS))
    segs = []
    for o in n_ech:
        o = int(o)
        if o + rep_len <= eeg_full_bp.shape[1]:
            segs.append(eeg_full_bp[:, o:o + rep_len])
    if not segs:
        return None
    eeg_proc, L = _proc_eeg_segment(np.concatenate(segs, axis=1), target_fs, clamp)
    reps = np.stack([source_repr(soli[i], src_fs, L, target_fs, band, compression, n_bands, kind)
                     for i in range(n_present)], axis=0)
    return [eeg_proc, reps, target_idx, n_present, "duo"]


# --------------------------------------------------------------------------- #
# decoder (ridge) + AAD decision
# --------------------------------------------------------------------------- #
def fit_ridge_pool(train_trials, n_lags, lam_grid, rng, shrinkage=False):
    """Backward model on the ATTENDED source representation, pooled over training trials.
    Multi-output (one column per band). Returns (predict_fn, lambda, inner_val_r).

    Under AXIS 3 the caller passes the single published lambda as the whole grid, so the
    loop stops being a search and becomes an evaluation -- inner_val_r keeps its meaning
    (in-distribution reconstruction quality) and stays comparable across axes."""
    Xs = [lagged_design(t[0], n_lags) for t in train_trials]
    Ys = [t[1][t[2]].T for t in train_trials]                       # (T, n_bands) attended rep
    tr_idx, val_idx = _inner_split(len(train_trials), rng)
    sc = StandardScaler().fit(np.concatenate([Xs[i] for i in tr_idx], 0))
    Xtr = sc.transform(np.concatenate([Xs[i] for i in tr_idx], 0))
    Ytr = np.concatenate([Ys[i] for i in tr_idx], 0)
    best_lam, best_r = lam_grid[0], -np.inf
    for lam in lam_grid:
        W = ridge_fit(Xtr, Ytr, lam, shrinkage)
        r = float(np.mean([band_pearson(sc.transform(Xs[i]) @ W, Ys[i]) for i in val_idx]))
        if r > best_r:
            best_r, best_lam = r, lam
    sc = StandardScaler().fit(np.concatenate(Xs, 0))
    W = ridge_fit(sc.transform(np.concatenate(Xs, 0)), np.concatenate(Ys, 0), best_lam, shrinkage)

    def predict(eeg):
        return sc.transform(lagged_design(eeg, n_lags)) @ W         # (T, n_bands)
    return predict, best_lam, best_r


def kfold_indices(n, k, rng):
    idx = np.arange(n); rng.shuffle(idx)
    return [idx[i::k] for i in range(k)]


def synth_attended_eeg(att, lags_op, W_op, rng, snr):
    """SELF-TEST ONLY. Synthetic EEG (n_ch, L) that is a FIXED linear mixture of the ATTENDED
    source representation `att` (n_bands, L): each channel = a band-weighted mix (a row of
    W_op) delayed by a fixed per-channel lag in [0, n_lags] samples, plus gaussian noise
    (signal/noise std ratio = `snr`). The forward operator (lags_op, W_op) is shared across
    trials, so a backward ridge model can recover `att` -> r(attended) >> r(unattended).
    Never called by the real reconstruction path (only under --self_test)."""
    n_ch, L = W_op.shape[0], att.shape[1]
    mix = W_op @ att                                     # (n_ch, L) band mixture per channel
    eeg = np.empty((n_ch, L))
    for ch in range(n_ch):
        lag = int(lags_op[ch])
        sh = np.roll(mix[ch], lag); sh[:lag] = 0.0       # causal delay: EEG lags the stimulus
        eeg[ch] = sh / (sh.std() + 1e-8) + rng.standard_normal(L) / snr
    return eeg


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="MAD-EEG stimulus-reconstruction AAD")
    ap.add_argument("--madeeg_dir", required=True, help="dir with madeeg_preprocessed.hdf5 + .yaml")
    ap.add_argument("--log_dir", default="", help="run-output dir; default <repo>/runs/results")
    ap.add_argument("--training_date", default="madeeg_recon")
    ap.add_argument("--inspect", type=int, default=0, help="print schema for first N trials and exit")
    ap.add_argument("--self_test", action="store_true",
                    help="positive control: replace EEG with a synthetic mixture of the ATTENDED "
                         "source at the model lags + noise, run the real ridge+AAD, print PASS/FAIL, exit")
    ap.add_argument("--self_test_snr", type=float, default=4.0,
                    help="--self_test only: signal/noise std ratio of the synthetic EEG")
    ap.add_argument("--ensemble", default="duo", choices=["duo", "trio", "both"])
    ap.add_argument("--target", default="mel", choices=["mel", "envelope", "mag"],
                    help="reconstruction target; 'mag' is AXIS 4, the paper's linear magnitude "
                         "spectrogram (bins fixed by the STFT geometry, --n_mels ignored)")
    ap.add_argument("--n_mels", type=int, default=8)
    # --- the four axes of the published protocol. Every default is the CURRENT behaviour,
    #     so the three reference numbers stay reproducible bit-for-bit. -------------------
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "shrinkage"],
                    help="AXIS 3: 'shrinkage' = the paper's normalized reverse correlation, "
                         "C'_RR = (1-l)C_RR + l*nu*I at the published l (--shrinkage_lambda), "
                         "instead of ridge with lambda picked on an inner split")
    ap.add_argument("--shrinkage_lambda", type=float, default=0.1,
                    help="AXIS 3: the paper's published smoothing parameter (grid search over "
                         "[0.1, 1] found 0.1). Only read when --estimator shrinkage")
    ap.add_argument("--filters", default="pooled", choices=["pooled", "per_instrument"],
                    help="AXIS 2: 'per_instrument' learns one decoder per (subject, instrument) "
                         "from that instrument's solos, as in the paper's Fig. 1 caption, and "
                         "reconstructs each candidate source with ITS OWN decoder. raw_solos only")
    ap.add_argument("--eeg_clean", default="none", choices=["none", "notch", "notch_ica"],
                    help="AXIS 1: the paper's EEG preprocessing on the raw record -- 50 Hz notch, "
                         "then ICA removal of the EOG/ECG components. raw_solos only (the "
                         "preprocessed release already carries the authors' own notch+ICA)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--target_fs", type=float, default=64.0)
    ap.add_argument("--band_low", type=float, default=1.0)
    ap.add_argument("--band_high", type=float, default=8.0)
    ap.add_argument("--lags_ms", type=float, default=250.0)
    ap.add_argument("--clamp", type=float, default=20.0)
    ap.add_argument("--compression", type=float, default=0.3)
    ap.add_argument("--train_on", default="duos_kfold", choices=["duos_kfold", "raw_solos"],
                    help="duos_kfold = k-fold on duo trials; raw_solos = paper protocol (train on raw solo trials, test on duos)")
    ap.add_argument("--raw_dir", default=None, help="dir with madeeg_raw.hdf5 + madeeg_raw.yaml + madeeg_sequences_raw.yaml (default: madeeg_dir)")
    ap.add_argument("--stimuli_dir", default=None, help="dir with the unzipped solo wavs (default: madeeg_dir/stimuli)")
    ap.add_argument("--test_eeg", default="preprocessed", choices=["preprocessed", "raw"],
                    help="raw_solos only: 'raw' rebuilds the test-duo EEG from raw too (no preprocessing mismatch)")
    args = ap.parse_args()
    if not args.log_dir:  # same run-output dir as main.py / checkpoint_test.py
        # Load paths.py by file rather than as `utils.paths`: the package __init__ pulls in
        # the training deps, and this script is sklearn/scipy only -- it must stay runnable
        # from the repo root, not just from src/.
        import importlib.util
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "paths.py")
        _spec = importlib.util.spec_from_file_location("_paths", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        args.log_dir = _mod.resolve_log_dir()

    data = h5py.File(os.path.join(args.madeeg_dir, "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.load(open(os.path.join(args.madeeg_dir, "madeeg_preprocessed.yaml")), Loader=yaml.FullLoader)
    band = (args.band_low, args.band_high)
    rng = np.random.RandomState(args.seed)
    n_lags = int(round(args.lags_ms / 1000.0 * args.target_fs))
    n_bands = n_repr_bands(args.target, args.n_mels, args.target_fs)
    shrink = args.estimator == "shrinkage"
    # AXIS 2 and AXIS 1 only exist on the raw path: `duos_kfold` has no solos to fit a
    # per-instrument decoder on, and the preprocessed release already carries the authors'
    # notch+ICA. Assert rather than silently ignoring -- a flag that is quietly dropped
    # produces a run that looks like the new arm and is the old one.
    if args.train_on != "raw_solos":
        assert args.filters == "pooled", "--filters per_instrument needs --train_on raw_solos"
        assert args.eeg_clean == "none", "--eeg_clean needs --train_on raw_solos (the preprocessed release is already cleaned)"
    ica_diag = {}

    keep = {"duo"} if args.ensemble == "duo" else ({"trio"} if args.ensemble == "trio" else {"duo", "trio"})
    trials = [(s, k) for s in meta for k in meta[s] if meta[s][k].get("ensemble") in keep]
    print(f"[madeeg] subjects={list(meta.keys())} ensemble={args.ensemble} n_trials={len(trials)} "
          f"target={args.target} n_bands={n_bands} target_fs={args.target_fs} lags={n_lags} band={band} "
          f"estimator={args.estimator} filters={args.filters} eeg_clean={args.eeg_clean}")

    if args.inspect:
        for s, k in trials[:args.inspect]:
            m = meta[s][k]
            print(f"\n  subj={s} stim={k}\n    ensemble={m['ensemble']} instruments={m['instruments']} "
                  f"target={m['target']} spatial={m['spatial']}")
            print(f"    response={data[s][k]['response'].shape}@{m['eeg_info']['sfreq']}Hz "
                  f"soli={data[s][k]['soli'].shape}@{m['wav_info']['sfreq']}Hz")
        return

    # AXIS 3: under shrinkage there is no search left -- the paper published the value, so
    # the "grid" is that single value and inner_val_r becomes its validation score.
    lam_grid = [args.shrinkage_lambda] if shrink else [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]

    # raw-solo training resources (paper protocol)
    raw_f = sq = raw_info = pre_chs = stim_dir = None
    if args.train_on == "raw_solos":
        raw_dir = args.raw_dir or args.madeeg_dir
        stim_dir = args.stimuli_dir or os.path.join(args.madeeg_dir, "stimuli")
        raw_f = h5py.File(os.path.join(raw_dir, "madeeg_raw.hdf5"), "r")
        sq = yaml.load(open(os.path.join(raw_dir, "madeeg_sequences_raw.yaml")), Loader=yaml.FullLoader)
        raw_info = yaml.load(open(os.path.join(raw_dir, "madeeg_raw.yaml")), Loader=yaml.FullLoader)
        s0 = next(iter(meta)); k0 = next(iter(meta[s0]))
        pre_chs = list(meta[s0][k0]["eeg_info"]["ch_names"])
        print(f"[madeeg] train_on=raw_solos raw_dir={raw_dir} stimuli={stim_dir} pre_chs={len(pre_chs)}")

    def decide(subj, st, test_built, predict, records, inner_val_r=float("nan"), fold=-1):
        """`predict` is one callable (pooled) or, under AXIS 2, a dict instrument->callable,
        in which case every candidate source is reconstructed by ITS OWN decoder."""
        for i in range(len(test_built)):
            eeg, reps, tgt, npr, ens = test_built[i]
            m = meta.get(subj, {}).get(st[i])                     # absent under --self_test
            instr = list(m["instruments"]) if m else [str(j) for j in range(npr)]
            if callable(predict):
                shat = predict(eeg)
                sims = [band_pearson(shat, reps[j].T) for j in range(npr)]
            else:
                sims = [band_pearson(predict[instr[j]](eeg), reps[j].T) for j in range(npr)]
            pred = int(np.argmax(sims))
            records.append(dict(subject=subj, stim=st[i], ensemble=ens, target_idx=tgt, n_present=npr,
                                pred=pred, correct=int(pred == tgt),
                                r_attended=sims[tgt], r_best_unattended=max(s for j, s in enumerate(sims) if j != tgt),
                                inner_val_r=inner_val_r, fold=fold,
                                target_instr=instr[tgt], pred_instr=instr[pred]))

    if args.self_test:
        # Positive control (see synth_attended_eeg): swap every trial's EEG for a synthetic
        # linear mixture of ITS attended source at lags 0..n_lags + noise, then run the REAL
        # pipeline (fit_ridge_pool + decide) unchanged. Passing proves that chance-level AAD on
        # the real EEG is a genuine absence of decodable signal, not a loader/decoder bug.
        lags_op = W_op = None
        built, stims = [], []
        for s, k in trials:
            eeg, reps, tgt, npr, ens = build_trial(data, meta, s, k, args.target_fs, band,
                                                    args.clamp, args.compression, n_bands, args.target)
            if W_op is None:                       # one fixed forward operator, shared by all trials
                n_ch = eeg.shape[0]
                lags_op = rng.randint(0, n_lags + 1, size=n_ch)
                W_op = rng.standard_normal((n_ch, reps.shape[1]))   # rows of the real repr, not the nominal count
            built.append([synth_attended_eeg(reps[tgt], lags_op, W_op, rng, args.self_test_snr),
                          reps, tgt, npr, ens])
            stims.append(k)
        st_records = []
        for fold in kfold_indices(len(built), args.cv_folds, rng):
            te = set(fold.tolist()); tr = [i for i in range(len(built)) if i not in te]
            predict, _, val_r = fit_ridge_pool([built[i] for i in tr], n_lags, lam_grid, rng, shrink)
            decide("synthetic", [stims[i] for i in fold], [built[i] for i in fold], predict, st_records, val_r)
        rec = pd.DataFrame(st_records)
        acc, ra, ru = rec.correct.mean(), rec.r_attended.mean(), rec.r_best_unattended.mean()
        chance = float(np.mean(1.0 / rec.n_present))
        ok = acc >= 0.90 and ra > ru
        print("\n== MAD-EEG SELF-TEST (synthetic attended-signal positive control) ==")
        print(f"n_trials={len(rec)} chance={chance:.3f} snr={args.self_test_snr} "
              f"n_ch={n_ch} n_bands={n_bands} lags={n_lags}")
        print(f"AAD accuracy={acc:.4f}  mean r(attended)={ra:.4f}  mean r(best unattended)={ru:.4f}")
        print(f"[{'PASS' if ok else 'FAIL'}] " + (
              "ridge recovers the attended source (r_att >> r_unatt, AAD>=0.90) -> pipeline is "
              "wired correctly, so chance-level AAD on real EEG is genuine signal absence."
              if ok else
              "ridge did NOT recover a signal it should trivially decode -> fix the loader/decoder "
              "before interpreting the real-data result."))
        return

    records = []
    for subj in meta:
        st = [k for k in meta[subj] if meta[subj][k].get("ensemble") in keep]   # test set = duos/trios
        if not st:
            continue
        if args.train_on == "raw_solos":
            raw_chs = list(raw_info[subj]["ch_names"])
            eeg_full_bp = raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band,
                                             args.eeg_clean, raw_info, args.seed, ica_diag)
            train_trials = build_solo_trials(eeg_full_bp, sq, stim_dir, subj, args.target_fs, band,
                                             args.clamp, args.compression, n_bands, args.target)
            if len(train_trials) < 3:
                print(f"  subj {subj}: only {len(train_trials)} solo segments -> skip"); continue
            if args.filters == "per_instrument":
                # AXIS 2: one decoder per (subject, instrument), fitted on that instrument's
                # solos only. Sorted so the rng is consumed in a fixed order.
                groups = {}
                for t in train_trials:
                    groups.setdefault(t[5], []).append(t)
                predict, vrs = {}, []
                for instrument in sorted(groups):
                    p, _, vr = fit_ridge_pool(groups[instrument], n_lags, lam_grid, rng, shrink)
                    predict[instrument] = p; vrs.append(vr)
                val_r = float(np.mean(vrs))
            else:
                predict, lam, val_r = fit_ridge_pool(train_trials, n_lags, lam_grid, rng, shrink)
            # build test duos: raw EEG (no mismatch) or preprocessed EEG
            if args.test_eeg == "raw":
                rawmap = {k.replace("_lcr", ""): k for k in sq[subj]}
                test_built, st_used = [], []
                for k in st:
                    rk = rawmap.get(k)
                    if rk is None:
                        continue
                    m = meta[subj][k]; instr = list(m["instruments"])
                    t = build_duo_trial_raweeg(eeg_full_bp, read_f64(data[subj][k]["soli"]),
                                               sq[subj][rk]["n_ech"], instr.index(m["target"]), len(instr),
                                               m["wav_info"]["sfreq"], args.target_fs, band, args.clamp,
                                               args.compression, n_bands, args.target)
                    if t is not None:
                        test_built.append(t); st_used.append(k)
            else:
                st_used = st
                test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands, args.target) for k in st]
            if not callable(predict):
                # AXIS 2: a trial is undecidable when one of its sources has no solo for this
                # subject (subject 0007 recorded 7 solos, not 14). Dropped and COUNTED, never
                # silently reconstructed with somebody else's filter.
                keepable = [i for i, k in enumerate(st_used)
                            if all(x in predict for x in meta[subj][k]["instruments"])]
                dropped = len(st_used) - len(keepable)
                if dropped:
                    print(f"  subj {subj}: {dropped} trial(s) dropped -- no solo decoder for one of their sources")
                test_built = [test_built[i] for i in keepable]; st_used = [st_used[i] for i in keepable]
            decide(subj, st_used, test_built, predict, records, val_r)
            print(f"  subj {subj}: solos={len(train_trials)} test={len(test_built)}/{len(st)} test_eeg={args.test_eeg} "
                  f"inner_val_r={val_r:.3f} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)
        else:
            test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands, args.target) for k in st]
            if len(st) < args.cv_folds:
                print(f"  subj {subj}: only {len(st)} trials -> skip"); continue
            folds = kfold_indices(len(test_built), args.cv_folds, rng)
            for f in range(args.cv_folds):
                te = set(folds[f].tolist()); tr = [i for i in range(len(test_built)) if i not in te]
                predict, lam, val_r = fit_ridge_pool([test_built[i] for i in tr], n_lags, lam_grid, rng, shrink)
                decide(subj, [st[i] for i in folds[f]], [test_built[i] for i in folds[f]], predict, records, val_r, fold=f)
            print(f"  subj {subj}: trials={len(st)} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)

    rec = pd.DataFrame(records)
    out = os.path.join(args.log_dir, args.training_date)
    os.makedirs(out, exist_ok=True)
    rec.to_csv(os.path.join(out, "madeeg_records.csv"), index=False)
    chance = float(np.mean(1.0 / rec.n_present))
    per_subj = rec.groupby("subject").correct.mean()
    # TRAP A -- the paper reports F1, we reported accuracy, and they are not the same number.
    # The decision is "which of the present instruments was attended", so the label space is
    # the instrument. In single-label multiclass, MICRO-F1 is identically the accuracy; MACRO
    # averages over the 9 instruments and so weights a rare instrument as much as a common
    # one. The paper does NOT say which averaging it used, so all three are printed and the
    # ambiguity is stated instead of being resolved by assumption.
    f1s = {a: f1_score(rec.target_instr, rec.pred_instr, average=a, zero_division=0)
           for a in ("micro", "macro", "weighted")}
    lines = ["== MAD-EEG stimulus-reconstruction AAD ==",
             f"train_on={args.train_on} ensemble={args.ensemble} target={args.target} n_bands={n_bands} "
             f"target_fs={args.target_fs} lags={n_lags} band={band} folds={args.cv_folds} "
             f"estimator={args.estimator}{'(lambda=%g)' % args.shrinkage_lambda if shrink else ''} "
             f"filters={args.filters} eeg_clean={args.eeg_clean} test_eeg={args.test_eeg}",
             f"n_trials={len(rec)}  chance={chance:.3f}",
             f"OVERALL AAD accuracy: {rec.correct.mean():.4f}",
             "per-subject: " + " ".join(f"{s}={v:.2f}" for s, v in per_subj.items()),
             f"F1 over the attended-instrument label: micro={f1s['micro']:.4f} (= accuracy) "
             f"macro={f1s['macro']:.4f} weighted={f1s['weighted']:.4f}",
             "  ^ the paper (WASPAA 2019 Table 1) reports F1 without naming the averaging;"
             " its duets column is AE 58 / MAG 74 / MEL 79",
             f"mean r(attended)={rec.r_attended.mean():.4f}  mean r(best unattended)={rec.r_best_unattended.mean():.4f}",
             f"mean inner_val_r (in-distribution reconstruction)={rec.inner_val_r.mean():.4f}"]
    if ica_diag:
        lines.append("EEG cleaning (AXIS 1) per subject -- positive control is the drop in "
                     "frontal-EOG coupling:")
        lines += [f"  {s}: {v}" for s, v in sorted(ica_diag.items())]
    txt = "\n".join(lines)
    open(os.path.join(out, "madeeg_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[madeeg] wrote -> {out}")


if __name__ == "__main__":
    main()
