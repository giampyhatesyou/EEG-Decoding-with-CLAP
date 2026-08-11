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
import collections
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


def _load_by_file(name, *parts):
    """Import a repo module by PATH, not as a package.

    Both `utils/__init__.py` and `models/__init__.py` pull in the training deps (torch,
    pytz, ...). This script is numpy/scipy/sklearn/h5py only -- that is what makes it the
    one MAD-EEG binary an agent may run without a GPU -- so its two repo imports go through
    the file, never through the package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CCA = None


def _cca_mod():
    """The multi-view/CCA module, loaded on first use so `--estimator ridge` imports nothing new."""
    global _CCA
    if _CCA is None:
        _CCA = _load_by_file("_cca_multiview", "models", "cca_multiview.py")
    return _CCA


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
    if kind == "flux":
        return 1
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
    elif kind == "flux":
        # CHANGED(baseline): Exp. 8 -- onset-strength regressor. Weineck, Wen & Henry
        # (eLife 2022, 75515): neural synchronization to natural music is strongest to the
        # SPECTRAL FLUX, not the amplitude envelope; flux carries rhythm communicated by
        # pitch changes with no amplitude change. Everything downstream (band-pass +
        # z-score) is identical to the mel branch, so the manipulation is single-variable.
        import librosa
        hop = max(1, int(round(src_fs / target_fs)))
        flux = librosa.onset.onset_strength(y=x, sr=int(src_fs), hop_length=hop)
        rep = flux[None, :].astype(np.float64)
        if rep.shape[1] != out_len:
            rep = resample(rep, out_len, axis=1)
    elif kind == "flux_mel":
        # CHANGED(baseline): Exp. 8 -- per-band spectral flux, capacity-matched to the mel
        # reference: same STFT geometry and n_mels as the mel branch, then the half-wave
        # rectified first difference per band (the standard per-band flux).
        import librosa
        hop = max(1, int(round(src_fs / target_fs)))
        n_fft = int(2 ** np.ceil(np.log2(2 * hop)))
        S = librosa.feature.melspectrogram(y=x, sr=int(src_fs), n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        logS = np.log(S + 1e-6)
        rep = np.maximum(np.diff(logS, axis=1, prepend=logS[:, :1]), 0.0)
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


def mixture_id(key):
    """`classique_morceau1_duo_CoFl_theme2_mono_Co` -> ('classique','morceau1','CoFl','theme2').

    The MIXTURE: genre, song, instrument pair, theme -- everything except the spatial
    render and the attended instrument. Two trials sharing a mixture id were rendered from
    the same stems, which is what makes the mono half of the duo recoverable at all
    (see duo_test_recipes)."""
    p = key.split("_")
    return (p[0], p[1], p[3], p[4])


def key_spatial(key):
    """'..._theme1_stereo_lcr_Co' -> 'stereo'; '..._theme2_mono_Co' -> 'mono'."""
    return "stereo" if "_stereo" in key else "mono"


def soli_donors(meta, keep):
    """mixture id -> the (subject, stim) of the preprocessed trial whose `soli` stands for
    that mixture. Sorted, so the donor of a mixture is the same on every machine and every
    run. Only mixtures of the requested ensembles are indexed."""
    donors = {}
    for s in sorted(meta):
        for k in sorted(meta[s]):
            if meta[s][k].get("ensemble") in keep:
                donors.setdefault(mixture_id(k), (s, k))
    return donors


def duo_test_recipes(meta, sq, subj, keep, spatial, donors):
    """Which duo trials to test on, and where each one's isolated sources come from.
    Returns ([(record_key, raw_key, (donor_subj, donor_stim), instruments, target)], dropped).

    spatial="stereo" is the CURRENT behaviour and the default: the 154 `stereo_lcr` duo of
    the preprocessed release, each trial carrying its own `soli`.

    spatial="mono" is the other half of the duo -- 155 trials that exist ONLY in the raw
    release. The preprocessed HDF5 contains no mono trial at all, so there is no `soli` to
    read for them; the isolated sources are borrowed from the preprocessed twin of the SAME
    mixture. That is well defined because `soli` is a property of the MIXTURE, not of the
    subject and not of the attend condition: all 18 duo mixtures have a twin (18/18 ->
    155/155 trials covered) and `wav_info` (gains, panning) is recorded per mixture, the
    same in every twin.

    Mono and stereo differ in the spatial RENDER -- how the same stems were distributed over
    the LCR speaker array. The release ships one audio channel per condition, so the render
    difference is not inspectable as "channels of the mixture"; what matters is that the AAD
    decision reads per-band z-scored correlations, so any per-source gain difference between
    the two renders cancels and the stem waveform -- all the decision sees -- is the same
    object. --check_alignment measures the borrowing instead of assuming it: on the stereo
    half, where the true `soli` exists, the borrowed stems must reproduce it.
    """
    recipes, dropped = [], []
    if spatial in ("stereo", "both"):
        rawmap = {k.replace("_lcr", ""): k for k in sq[subj]}
        for k in meta[subj]:                                  # yaml order: record order
            if meta[subj][k].get("ensemble") not in keep:
                continue
            rk = rawmap.get(k)
            if rk is None:
                dropped.append((k, "no raw sequence")); continue
            m = meta[subj][k]
            recipes.append((k, rk, (subj, k), list(m["instruments"]), m["target"]))
    if spatial in ("mono", "both"):
        for k in sorted(sq[subj]):
            p = k.split("_")
            if p[2] not in keep or key_spatial(k) != "mono":
                continue
            donor = donors.get(mixture_id(k))
            if donor is None:
                dropped.append((k, "no preprocessed twin for this mixture")); continue
            instr = list(meta[donor[0]][donor[1]]["instruments"])
            target = p[-1]
            if target not in instr:
                dropped.append((k, f"target {target} not in {instr}")); continue
            recipes.append((k, k, donor, instr, target))
    return recipes, dropped


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


# Recovery-gate thresholds, declared here and not at the prompt. Both are the tolerances
# this project already uses for the same two questions on the 246 preprocessed trials
# (src/datasets/madeeg_contrastive_dataset.py::_self_check and _check_soli_row_order), so
# they are inherited rather than eyeballed for the occasion.
ALIGN_TOL_S = 0.05        # EEG span vs audio span, per trial
SOLI_MATCH_R = 0.99       # two twins of one mixture must carry the same stem


def check_alignment(meta, sq, data, raw_f, keep, spatial, donors, stim_dir):
    """Recovery gate for the raw-EEG duo test set. Answers, with a count for every claim:
    how many trials get built, whether EEG and audio cover the same span, whether the four
    repetitions fall at the declared onsets, and whether the borrowed-`soli` mechanism the
    mono half depends on actually returns that mixture's stems.

    Every question here has its answer known by construction, so it is a positive control
    and it costs no look at the attention decision: durations, onsets, stem waveforms and
    wav headers only -- no decoder is fitted and no trial is decided.
    """
    import soundfile as sf                                   # ships with librosa
    ok = True
    built = {}
    for subj in sorted(sq):
        recipes, dropped = duo_test_recipes(meta, sq, subj, keep, spatial, donors)
        built[subj] = (recipes, dropped)

    # --- (a) the count, before anything else ------------------------------------------- #
    print(f"\n== RECOVERY GATE -- raw duo test set, spatial={spatial} ==")
    solo_instr = {s: {solo_instrument(k) for k in sq[s] if "solo" in k} for s in sq}
    n_tot = n_dec = 0
    for subj in sorted(built):
        recipes, dropped = built[subj]
        dec = [r for r in recipes if all(i in solo_instr[subj] for i in r[3])]
        n_tot += len(recipes); n_dec += len(dec)
        print(f"  {subj}: built={len(recipes):3d}  not built={len(dropped)}  "
              f"decidable with a per-instrument decoder={len(dec):3d}  "
              f"(solos for {sorted(solo_instr[subj])})")
    print(f"  TOTAL built={n_tot}   with --filters per_instrument={n_dec} "
          f"({n_tot - n_dec} lack a solo decoder for one of their two sources)")

    # --- (b) alignment: spans and the four declared onsets ------------------------------ #
    # The unit that has to line up is the REPETITION: build_duo_trial_raweeg cuts one EEG
    # segment of `soli`/4 at each declared onset and pairs it with one quarter of `soli`. So
    # the governing quantity is |played wav_i - soli/4| per repetition, and ALIGN_TOL_S is
    # applied to it. The 4-repetition SUM is printed alongside because it accumulates a
    # per-file truncation of a few ms into something that crosses the tolerance on its own
    # and would read as a fault where there is none.
    bad_reps = bad_fit = bad_overlap = bad_span = bad_rep_dur = bad_sum_dur = 0
    worst_span = worst_rep = 0.0
    offenders = collections.Counter()
    for subj in sorted(built):
        N = raw_f[subj].shape[1]
        for rec_key, raw_key, (ds_, dk_), instr, target in built[subj][0]:
            n_ech = [int(o) for o in sq[subj][raw_key]["n_ech"]]
            src_fs = meta[ds_][dk_]["wav_info"]["sfreq"]
            soli_len = data[ds_][dk_]["soli"].shape[1]
            if len(n_ech) != 4:
                bad_reps += 1; continue
            rep_len = int(round((soli_len / len(n_ech)) / src_fs * EEG_FS))
            if any(o + rep_len > N for o in n_ech):
                bad_fit += 1                   # a dropped repetition squeezes the audio
            if rep_len > min(np.diff(n_ech)):
                bad_overlap += 1               # a segment would run into the next repetition
            span = abs(len(n_ech) * rep_len / EEG_FS - soli_len / src_fs)
            worst_span = max(worst_span, span)
            if span >= ALIGN_TOL_S:
                bad_span += 1
            durs = [sf.info(os.path.join(stim_dir, w)).duration
                    for w in sq[subj][raw_key]["wav_files"]]
            quarter = soli_len / len(n_ech) / src_fs
            err = max(abs(x - quarter) for x in durs)
            worst_rep = max(worst_rep, err)
            if err >= ALIGN_TOL_S:
                bad_rep_dur += 1; offenders["_".join(mixture_id(rec_key)) + f" [{key_spatial(rec_key)}]"] += 1
            if abs(sum(durs) - soli_len / src_fs) >= ALIGN_TOL_S:
                bad_sum_dur += 1
    print(f"  alignment over {n_tot} trials (tolerance {ALIGN_TOL_S} s, the one this project "
          f"already uses for the 246 preprocessed):")
    print(f"    repetitions != 4: {bad_reps}   onset+len past the end of the record: {bad_fit}   "
          f"segment overlapping the next onset: {bad_overlap}")
    print(f"    EEG span vs `soli` span out of tolerance: {bad_span} (worst {worst_span * 1000:.1f} ms)")
    print(f"    played wav_i vs `soli`/4 PER REPETITION out of tolerance: {bad_rep_dur} "
          f"(worst {worst_rep * 1000:.1f} ms)   [4-rep sum, context only: {bad_sum_dur}]")
    for mix, n in offenders.most_common():
        print(f"      ^ {n} trial(s) on {mix}: the released last repetition is truncated, so its "
              f"EEG segment runs past the audio it is paired with")
    ok &= (bad_reps == 0 and bad_fit == 0 and bad_overlap == 0 and bad_span == 0 and bad_rep_dur == 0)

    # --- (c1) the borrowing mechanism, checked where the truth exists ------------------- #
    # Positive control: on the stereo half every trial carries its OWN `soli`, so the stems
    # fetched through mixture_id from another twin can be compared against them. If mixture_id
    # or the instrument row order were wrong, this is where it shows.
    twins = {}
    for s in sorted(meta):
        for k in sorted(meta[s]):
            if meta[s][k].get("ensemble") in keep:
                twins.setdefault(mixture_id(k), []).append((s, k))
    n_cmp, worst_r, mism = 0, 1.0, 0
    for mix, lst in sorted(twins.items()):
        ds_, dk_ = donors[mix]
        ref = read_f64(data[ds_][dk_]["soli"]); ref_i = list(meta[ds_][dk_]["instruments"])
        for (s, k) in lst:
            if (s, k) == (ds_, dk_):
                continue
            other = read_f64(data[s][k]["soli"]); other_i = list(meta[s][k]["instruments"])
            n = min(ref.shape[1], other.shape[1])
            for inst in sorted(set(ref_i) & set(other_i)):
                r = pearson(ref[ref_i.index(inst), :n], other[other_i.index(inst), :n])
                n_cmp += 1; worst_r = min(worst_r, r)
                if r <= SOLI_MATCH_R:
                    mism += 1
    print(f"  borrowed stems vs the twin's own `soli`: {n_cmp} row comparisons over "
          f"{len(twins)} mixtures, worst r={worst_r:.4f}, below {SOLI_MATCH_R}: {mism}")
    ok &= (mism == 0 and n_cmp > 0)

    # --- (c2) the mono trials are mono, and are matched to their own mixture ------------ #
    name_ok = name_bad = 0
    mono_pairs, chan = [], set()
    for subj in sorted(built):
        for rec_key, raw_key, (ds_, dk_), instr, target in built[subj][0]:
            w0 = sq[subj][raw_key]["wav_files"][0]
            chan.add(sf.info(os.path.join(stim_dir, w0)).channels)
            # the mixture the subject actually heard, read off the played file name
            if mixture_id(w0) == mixture_id(dk_) and key_spatial(w0) == key_spatial(rec_key):
                name_ok += 1
            else:
                name_bad += 1
            if key_spatial(w0) == "mono":
                mono_pairs.append(w0)
    print(f"  played wav vs borrowed mixture: {name_ok} match, {name_bad} mismatch; "
          f"channel counts in the release: {sorted(chan)}")
    ok &= (name_bad == 0)

    if mono_pairs:
        # "mono and stereo differ only in the panning" -- the release ships ONE channel per
        # condition, so this is not inspectable as channels of the mixture. What is
        # inspectable: the two renders of the same mixture are the same length (same stems,
        # same excerpt) and different content (the LCR spread is baked into the render).
        same_len = diff_content = n_pair = 0
        rs = []
        for w in sorted(set(mono_pairs)):
            st = w.replace("_mono_", "_stereo_lcr_")
            p1, p2 = os.path.join(stim_dir, w), os.path.join(stim_dir, st)
            if not os.path.exists(p2):
                continue
            a, _ = sf.read(p1); b, _ = sf.read(p2); n_pair += 1
            if a.shape == b.shape:
                same_len += 1
            n = min(len(a), len(b))
            r = pearson(a[:n], b[:n]); rs.append(r)
            if r < 0.999:
                diff_content += 1
        print(f"  mono vs stereo_lcr render of the same mixture: {n_pair} pairs, "
              f"same length {same_len}, different content {diff_content}, "
              f"median r={np.median(rs):.3f} -- same stems, different spatial render")
        ok &= (n_pair > 0 and same_len == n_pair and diff_content == n_pair)

        # And the mono trials are genuinely other EEG: their onsets are elsewhere in the record.
        overlap = 0
        for subj in sorted(built):
            ster = {int(o) for k in sq[subj] if key_spatial(k) == "stereo"
                    for o in sq[subj][k]["n_ech"]}
            for rec_key, raw_key, _, _, _ in built[subj][0]:
                if key_spatial(rec_key) != "mono":
                    continue
                if any(int(o) in ster for o in sq[subj][raw_key]["n_ech"]):
                    overlap += 1
        print(f"  mono trials whose EEG onsets coincide with a stereo trial's: {overlap} "
              f"(must be 0 -- otherwise the 'mono' half is re-reading stereo EEG)")
        ok &= (overlap == 0)

    print(f"\n[{'PASS' if ok else 'FAIL'}] " + (
        f"{n_tot} trials built for spatial={spatial}, spans and onsets consistent, borrowed "
        "stems reproduce the twins' own stems -> the loading is sound. Says NOTHING about "
        "decoding accuracy." if ok else
        "the raw duo test set does not load consistently -- fix it before decoding anything."))
    return ok


def duo_eeg_segments(eeg_full_bp, soli_len, n_ech, src_fs):
    """The trial's EEG: one segment of `soli_len`/len(n_ech) at each declared onset,
    concatenated. Returns None if no onset fits inside the record.

    Shared by the reconstruction test-trial builder and by the alpha laterality index, so the
    two can never drift apart on WHICH SAMPLES ARE THIS TRIAL -- a divergence there would be
    invisible in every summary and would make the two analyses answer about different data."""
    rep_len = int(round((soli_len / max(1, len(n_ech))) / src_fs * EEG_FS))
    segs = [eeg_full_bp[:, int(o):int(o) + rep_len] for o in n_ech
            if int(o) + rep_len <= eeg_full_bp.shape[1]]
    return np.concatenate(segs, axis=1) if segs else None


def build_duo_trial_raweeg(eeg_full_bp, soli, n_ech, target_idx, n_present, src_fs,
                           target_fs, band, clamp, compression, n_bands, kind="mel"):
    """Test trial with RAW EEG (same pipeline as the solos -> no preprocessing mismatch).
    The 4 repetitions are cut from the raw EEG at their n_ech onsets (each rep length =
    soli_len/4 in EEG samples) and concatenated to match the 4-rep preprocessed 'soli'
    audio, which provides the isolated sources for the AAD decision."""
    seg = duo_eeg_segments(eeg_full_bp, soli.shape[1], n_ech, src_fs)
    if seg is None:
        return None
    eeg_proc, L = _proc_eeg_segment(seg, target_fs, clamp)
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


def source_positions(meta, subj, stim, n_present):
    """Side of each present source, from `wav_info.panning`: -1 left, +1 right, 0 centre.

    MODEL 9 only. The mapping is the dataset's own: 0.2 = left, 0.8 = right, 0.5 = centre.

    MONO IS ZERO BY CONSTRUCTION, and it is read off the trial KEY, never off the metadata
    the trial borrows. The mono half has no entry of its own in the preprocessed yaml and
    takes its `soli` from the stereo twin -- whose panning is +-1. Taking the donor's panning
    would hand the mono control exactly the spatial information the control exists to
    withhold. The dataset tutorial states the fact: "in the mono case, the panning will be
    0.5 for all instruments".

    An absent entry (the --self_test subject, which is synthetic) also gives 0: the synthetic
    EEG carries no spatial structure to correlate with."""
    if key_spatial(stim) == "mono":
        return np.zeros(n_present)
    m = meta.get(subj, {}).get(stim)
    if m is None:
        return np.zeros(n_present)
    pan = [float(p) for p in m["wav_info"]["panning"][:n_present]]
    return np.array([0.0 if abs(p - 0.5) < 1e-9 else (-1.0 if p < 0.5 else 1.0) for p in pan])


def fit_cca_pool(train_trials, train_pos, n_lags, rng, cfg):
    """MODEL 9 -- regularized multi-view CCA on the pooled (EEG, ATTENDED source) pairs.

    Returns (model, inner_val_r, inner_val_rho), and the two numbers are NOT the same kind
    of thing:

      inner_val_r    band_pearson between the BACK-PROJECTED reconstruction and the attended
                     representation, on the inner validation split -- the same function, the
                     same target and the same split logic `fit_ridge_pool` uses. This is the
                     one number comparable with the ridge's 0.0582.
      inner_val_rho  the sum of the canonical correlations on that split. A canonical
                     correlation is >= a Pearson BY CONSTRUCTION, so it is not comparable
                     with any ridge number and must never be reported as if it were.

    The inner split is drawn with the same `_inner_split(len, rng)` the ridge uses, so the
    two estimators validate on the same kind of held-out solos."""
    cca = _cca_mod()
    Xs, Ys, Rs, cols = [], [], [], None
    bx = by = None
    for t, pos in zip(train_trials, train_pos):
        X, bx = cca.eeg_features(t[0], cfg["ch_names"], cfg["views"], n_lags,
                                 cfg["band"], cfg["fs"], lagged_design)
        Y, by, cols = cca.stim_features(t[1][t[2]], pos, cfg["stim_views"])
        Xs.append(X); Ys.append(Y); Rs.append(t[1][t[2]].T)
    cfg["blocks_x"], cfg["blocks_y"] = bx, by          # for the run summary, set on first fit
    tr_idx, val_idx = _inner_split(len(train_trials), rng)
    inner = cca.fit(np.concatenate([Xs[i] for i in tr_idx], 0),
                    np.concatenate([Ys[i] for i in tr_idx], 0), cfg["k"], cfg["reg"])
    val_r = float(np.mean([band_pearson(inner.reconstruct(Xs[i])[:, cols], Rs[i]) for i in val_idx]))
    val_rho = float(np.mean([inner.rho(Xs[i], Ys[i]) for i in val_idx]))
    model = cca.fit(np.concatenate(Xs, 0), np.concatenate(Ys, 0), cfg["k"], cfg["reg"])
    return model, val_r, val_rho


def fit_pool(train_trials, train_pos, n_lags, lam_grid, rng, shrink, cca_cfg):
    """Estimator dispatch. Returns (model, inner_val_r, inner_val_rho).

    With `cca_cfg is None` this calls `fit_ridge_pool` with the arguments it always had and
    returns its two numbers unchanged: the ridge and shrinkage paths run the same code they
    ran before model 9 existed, and consume the rng in the same order."""
    if cca_cfg is None:
        predict, _lam, val_r = fit_ridge_pool(train_trials, n_lags, lam_grid, rng, shrink)
        return predict, val_r, float("nan")
    return fit_cca_pool(train_trials, train_pos, n_lags, rng, cca_cfg)


def native_score(model, eeg, rep, n_lags, cca_cfg, ch_names, band, fs, n_keep):
    """The estimator's OWN similarity between one EEG segment and one source representation.

    ridge / shrinkage -> band_pearson of the reconstruction.   CCA -> rho.
    🔴 The two are NEVER compared with each other. Only the ACCURACIES they produce are, and
    those are dimensionless. Comparing rho against a band_pearson is the metric trap in its
    third costume (AXIS 2 on 2026-07-29, the step-1 gate this morning).

    `n_keep` truncates both sides to the same number of samples, so `own` and `other` are
    always scored on equally long windows: two solo repetitions have different durations and
    a longer window is not a fairer one."""
    if cca_cfg is None:
        return band_pearson(model(eeg)[:n_keep], rep.T[:n_keep])
    cca = _cca_mod()
    X, _ = cca.eeg_features(eeg, ch_names, cca_cfg["views"], n_lags, band, fs, lagged_design)
    Y, _, _ = cca.stim_features(rep, 0.0, cca_cfg["stim_views"])   # solos are mono -> position 0
    return model.rho(X[:n_keep], Y[:n_keep])


def mutual_pairs(trials, held_out):
    """Disjoint MUTUAL pairs of held-out segments with DIFFERENT instruments.

    Different instrument is not a refinement, it is required: `build_solo_trials` makes one
    trial per stimulus REPETITION, so two segments of the same instrument carry the SAME
    audio and 'other' would literally be 'own'.

    Mutual and disjoint because each pair is then scored in BOTH directions, which is what
    makes the null exactly 0.5 by symmetry: a decoder with nothing but a fixed instrument
    preference gets exactly one of the two right, however strong the preference."""
    out, used = [], set()
    order = sorted(held_out)
    for a in order:
        if a in used:
            continue
        for b in order:
            if b <= a or b in used or trials[b][5] == trials[a][5]:
                continue
            out.append((a, b)); used.add(a); used.add(b)
            break
    return out


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
    ap.add_argument("--self_test_pool", type=int, default=0,
                    help="--self_test only: cap the number of training trials pooled per fold "
                         "(0 = all, the default and the behaviour every archived PASS was run "
                         "with). At --target_fs 256 the full pool is a ~7.7 GB design matrix "
                         "and does not fit in 16 GB; a cap makes the control frugal enough to "
                         "actually cross. The count used is printed and written to the summary")
    ap.add_argument("--check_alignment", action="store_true",
                    help="recovery gate for the raw-EEG duo test set (honours --spatial): "
                         "counts the trials built, checks EEG/audio durations and the declared "
                         "onsets, and validates the borrowed-`soli` mechanism against the "
                         "stereo half where the true `soli` exists. Prints PASS/FAIL and exits. "
                         "Touches no accuracy: durations, onsets and audio only")
    ap.add_argument("--alpha_li", action="store_true",
                    help="THE PAIRED ALPHA TEST (vault: Exp. 6, 10/8). No decoder, no audio, no "
                         "training: the alpha laterality index on F3/F4, C3/C4, P3/P4, O1/O2, "
                         "scored on TWIN PAIRS (same subject, same mixture, both targets), where "
                         "every constant of subject/channel/session cancels in the difference. "
                         "Null 0.5 exactly, by exchangeability. Honours --spatial: mono is the "
                         "control and must stay at chance. THIS IS A LOOK at the duos")
    ap.add_argument("--alpha_inject", type=float, default=0.0,
                    help="--alpha_li POSITIVE CONTROL: inject a lateralized alpha modulation of "
                         "this amplitude (x the channel's own alpha std) into the REAL EEG, with "
                         "the side taken from the TRUE panning, and check the same code recovers "
                         "it. Threshold 0.90, declared in the code. Writes to a separate "
                         "directory with the caveat inside the summary. 0 = off = real data")
    ap.add_argument("--own_vs_other", action="store_true",
                    help="THE REPLACEMENT GATE (vault: Exp. 6, 10/8). On held-out SOLO segments, "
                         "does the estimator score a segment's EEG higher against its own audio "
                         "than against another solo's? Discrimination -- the quantity de "
                         "Cheveigne's claim is about -- on TRAINING material, so no duo is "
                         "loaded and no attention decision is taken. Each estimator uses its OWN "
                         "similarity and only the accuracies are compared. Null 0.5, exact by "
                         "symmetry. Pair two runs with madeeg_diagnose.py --mcnemar")
    ap.add_argument("--inner_val_only", action="store_true",
                    help="fit the decoder exactly as a full run would and report inner_val_r "
                         "ONLY: no test trial is built and none is decided, so no accuracy "
                         "exists to be read by accident. The vault's ledger of looks declares "
                         "inner_val_r free because it is scored on an inner split of the "
                         "TRAINING material and touches no attention decision; this flag makes "
                         "that structural instead of a matter of discipline. Use it for every "
                         "configuration step that is not the one pre-registered look")
    ap.add_argument("--ensemble", default="duo", choices=["duo", "trio", "both"])
    ap.add_argument("--target", default="mel", choices=["mel", "envelope", "mag", "flux", "flux_mel"],
                    help="reconstruction target; 'mag' is AXIS 4, the paper's linear magnitude "
                         "spectrogram (bins fixed by the STFT geometry, --n_mels ignored)")
    ap.add_argument("--n_mels", type=int, default=8)
    # --- the four axes of the published protocol. Every default is the CURRENT behaviour,
    #     so the three reference numbers stay reproducible bit-for-bit. -------------------
    ap.add_argument("--estimator", default="ridge", choices=["ridge", "shrinkage", "cca"],
                    help="AXIS 3: 'shrinkage' = the paper's normalized reverse correlation, "
                         "C'_RR = (1-l)C_RR + l*nu*I at the published l (--shrinkage_lambda), "
                         "instead of ridge with lambda picked on an inner split. "
                         "MODEL 9: 'cca' = regularized multi-view canonical correlation "
                         "(de Cheveigne et al. 2018); the decision stays argmax over the "
                         "present sources, of rho(s) instead of band_pearson")
    # --- MODEL 9, the multi-view CCA. Every default reproduces the ridge's data path exactly
    #     (one EEG view = the ridge's own design matrix, one stimulus view = the ridge's own
    #     target), so `--estimator cca` with no other flag is the like-for-like comparison. --
    ap.add_argument("--cca_views", default="eeg_lagged",
                    help="MODEL 9: comma-separated EEG views -- eeg_lagged (the ridge's design "
                         "matrix, unchanged), band_power (per-channel log power envelope per "
                         "declared band), lateralization (log P(right)-log P(left) on F3/F4, "
                         "C3/C4, P3/P4, O1/O2, alpha band). Bands outside --band_low/--band_high "
                         "are zero by construction: they are dropped and the drop is declared")
    ap.add_argument("--cca_stim_views", default="source_repr",
                    help="MODEL 9: comma-separated stimulus views -- source_repr (the log-mel "
                         "of the candidate source, unchanged), position (its side from "
                         "wav_info.panning, -1/0/+1), position_env (position x the source's own "
                         "envelope). NOTE, measured and not assumed: `position` is constant in "
                         "time, so a within-trial Pearson centres it out and it CANNOT move the "
                         "decision; position_env is the time-varying form, still identically 0 "
                         "for every source in mono. src/models/cca_multiview.py pins both")
    ap.add_argument("--cca_components", type=int, default=5,
                    help="MODEL 9: k, the number of canonical components summed into rho(s)")
    ap.add_argument("--cca_reg", type=float, default=1e-3,
                    help="MODEL 9: CCA regularization in [0,1], C' = (1-r)C + r*nu*I. NOT "
                         "optional: 20 channels x lags x views against ~150 trials overfits an "
                         "unregularized CCA completely")
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
    ap.add_argument("--spatial", default="stereo", choices=["stereo", "mono", "both"],
                    help="which spatial render of the duo to TEST on. 'stereo' = the 154 duo of "
                         "the preprocessed release, the current behaviour and the default. "
                         "'mono' = the 155 duo that exist only in the raw release, with the "
                         "isolated sources borrowed from the same mixture's preprocessed twin "
                         "(see duo_test_recipes). 'both' pools the two halves -- POOLING IS A "
                         "SEPARATE DECISION and was explicitly not taken on 2026-07-30: the two "
                         "halves are reported separately first (vault: Exp. 4 pre-registration)")
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
    # The 20 EEG channels are in this order on BOTH paths: the preprocessed release stores
    # them so, and `raw_eeg_bandpassed` reorders the raw record to match. The lateralization
    # view indexes channels by name, so it must read the order from the data, not a literal.
    _s0 = next(iter(meta)); _k0 = next(iter(meta[_s0]))
    ch_names = list(meta[_s0][_k0]["eeg_info"]["ch_names"])
    # MODEL 9. The cca_* flags are asserted, not ignored: a run that carries them under
    # --estimator ridge would look like the new arm and be the old one (Comandamenti #8).
    cca_cfg = None
    if args.estimator == "cca":
        _c = _cca_mod()
        _views = _c.parse_views(args.cca_views, _c.EEG_VIEWS, "--cca_views")
        _sviews = _c.parse_views(args.cca_stim_views, _c.STIM_VIEWS, "--cca_stim_views")
        assert "source_repr" in _sviews, (
            "--cca_stim_views must contain source_repr: it is the only block the "
            "back-projection can be scored against, i.e. the only way to produce the number "
            "that is comparable with the ridge")
        assert args.cca_components >= 1 and 0.0 < args.cca_reg < 1.0, (
            "--cca_components >= 1 and --cca_reg in (0,1)")
        cca_cfg = dict(views=_views, stim_views=_sviews, k=args.cca_components,
                       reg=args.cca_reg, band=band, fs=args.target_fs, ch_names=ch_names,
                       blocks_x=None, blocks_y=None)
    else:
        assert (args.cca_views == "eeg_lagged" and args.cca_stim_views == "source_repr"
                and args.cca_components == 5 and args.cca_reg == 1e-3), (
            "--cca_* only mean something with --estimator cca")
    # AXIS 2 and AXIS 1 only exist on the raw path: `duos_kfold` has no solos to fit a
    # per-instrument decoder on, and the preprocessed release already carries the authors'
    # notch+ICA. Assert rather than silently ignoring -- a flag that is quietly dropped
    # produces a run that looks like the new arm and is the old one.
    if args.train_on != "raw_solos":
        assert args.filters == "pooled", "--filters per_instrument needs --train_on raw_solos"
        assert args.eeg_clean == "none", "--eeg_clean needs --train_on raw_solos (the preprocessed release is already cleaned)"
    # The mono half lives only in the raw release: there is no preprocessed EEG for it, so it
    # cannot be reached from the preprocessed test path. Assert instead of silently falling
    # back to stereo -- a run that looks like the mono arm and is the stereo one would be the
    # worst possible failure here, because the two arms differ by ONE trial in count.
    if args.spatial != "stereo":
        assert args.train_on == "raw_solos" and args.test_eeg == "raw", (
            "--spatial mono/both needs --train_on raw_solos --test_eeg raw: the mono duo have "
            "no preprocessed EEG at all")
        assert args.ensemble == "duo", (
            "--spatial mono/both is duo-only here: the 93 mono TRIO of the raw release are "
            "untouched data and opening them is a separate, explicit decision")
    ica_diag = {}

    if args.check_alignment:
        assert args.train_on == "raw_solos" and args.test_eeg == "raw", (
            "--check_alignment checks the raw-EEG duo test set: run it with "
            "--train_on raw_solos --test_eeg raw")

    keep = {"duo"} if args.ensemble == "duo" else ({"trio"} if args.ensemble == "trio" else {"duo", "trio"})
    trials = [(s, k) for s in meta for k in meta[s] if meta[s][k].get("ensemble") in keep]
    donors = soli_donors(meta, keep)
    print(f"[madeeg] subjects={list(meta.keys())} ensemble={args.ensemble} n_trials={len(trials)} "
          f"target={args.target} n_bands={n_bands} target_fs={args.target_fs} lags={n_lags} band={band} "
          f"estimator={args.estimator} filters={args.filters} eeg_clean={args.eeg_clean} "
          f"spatial={args.spatial}")

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
        if cca_cfg is not None:
            # MODEL 9, counted and not assumed: every training solo is a MONO render, so the
            # position view is identically 0 across the whole training set. That is exactly
            # why position enters the frozen configuration by DECLARATION and can never be
            # selected on inner_val_r -- there is nothing there for it to improve.
            solo_keys = [k for s in sq for k in sq[s] if "solo" in k]
            assert solo_keys and all(key_spatial(k) == "mono" for k in solo_keys), (
                "MODEL 9 assumes the training solos are mono renders; they are not")
            print(f"[madeeg] MODEL 9: {len(solo_keys)} training solo keys, all mono -> the "
                  f"position view is 0 for every training pair")

    if args.check_alignment:
        check_alignment(meta, sq, data, raw_f, keep, args.spatial, donors, stim_dir)
        return

    if args.alpha_li:
        # THE PAIRED ALPHA TEST (vault: Exp. 6 pre-registration, 10/8). Nothing is trained and
        # no audio is read: this is a sign test on a TONIC quantity, which is why it lives
        # outside the CCA -- a within-trial correlation centres tonic levels out.
        alz = _load_by_file("_alpha_lat", "models", "alpha_lateralization.py")
        assert args.ensemble == "duo", (
            "--alpha_li is duo-only: the trios are untouched data and opening them is a "
            "separate, explicit decision")
        assert args.train_on == "raw_solos", (
            "--alpha_li reads the continuous RAW record: run it with --train_on raw_solos "
            "(nothing is trained; that flag is what loads the raw release)")
        assert args.spatial != "both", (
            "--alpha_li must not pool the two renders: mono is the CONTROL arm and pooling it "
            "into the primary destroys it. Run stereo and mono separately")
        inj_rng = np.random.RandomState(args.seed)
        rows = []
        for subj in sorted(sq):
            raw_chs = list(raw_info[subj]["ch_names"])
            # A WIDE pre-filter only: the alpha selection happens inside laterality_index, so an
            # injected control signal passes through the very filter under test. The 1-8 Hz
            # analysis band of the reconstruction arm is irrelevant here and is NOT used --
            # an alpha index computed inside a band that excludes alpha is a different quantity.
            eeg_wide = raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, (1.0, 45.0),
                                          args.eeg_clean, raw_info, args.seed, ica_diag)
            recipes, _ = duo_test_recipes(meta, sq, subj, keep, args.spatial, donors)
            for rec_key, raw_key, (ds_, dk_), instr, target in recipes:
                seg = duo_eeg_segments(eeg_wide, data[ds_][dk_]["soli"].shape[1],
                                       sq[subj][raw_key]["n_ech"],
                                       meta[ds_][dk_]["wav_info"]["sfreq"])
                if seg is None:
                    continue
                # The NOMINAL side of the attended source, from the mixture's panning. In mono
                # there is no side physically present; the same nominal panning is used so the
                # RULE is literally identical and the only thing removed is the spatial
                # separation itself. That is what makes mono a control and not a different test.
                pan = [float(p) for p in meta[ds_][dk_]["wav_info"]["panning"][:len(instr)]]
                side = {i: (0.0 if abs(p - 0.5) < 1e-9 else (-1.0 if p < 0.5 else 1.0))
                        for i, p in zip(instr, pan)}[target]
                if args.alpha_inject:
                    seg = alz.inject_lateralized_alpha(seg, pre_chs, EEG_FS, side,
                                                       args.alpha_inject, inj_rng)
                rows.append(dict(subject=subj, stim=rec_key,
                                 mixture="_".join(mixture_id(rec_key)),
                                 render=key_spatial(rec_key), target=target, side=side,
                                 li=alz.laterality_index(seg, pre_chs, EEG_FS)))
            print(f"  subj {subj}: trials with an LI = "
                  f"{sum(1 for r in rows if r['subject'] == subj)}", flush=True)

        li = pd.DataFrame(rows)
        pairs, dropped_centre, dropped_same = [], 0, 0
        for (subj, mix), grp in li.groupby(["subject", "mixture"]):
            if len(grp) != 2 or grp.target.nunique() != 2:
                continue
            a, b = grp.iloc[0], grp.iloc[1]
            if a.side == 0.0 or b.side == 0.0:
                # A target at panning 0.5 has NO side. The Exp. 5 contract (9/8) excludes those
                # trials A PRIORI -- all 5 are pop_mixtape_duo_GtVx_theme1_stereo_Vx, where the
                # voice is centred -- and the pre-registered n=44 is the subset that remains.
                dropped_centre += 1
                continue
            ok = alz.pair_correct(a.li, b.li, a.side, b.side)
            if ok is None:                      # both targets on the same side: no sign predicted
                dropped_same += 1
                continue
            pairs.append(dict(subject=subj, mixture=mix, render=a.render,
                              target_a=a.target, target_b=b.target, side_a=a.side, side_b=b.side,
                              li_a=a.li, li_b=b.li, correct=ok))
        pr = pd.DataFrame(pairs)
        out = os.path.join(args.log_dir, args.training_date + ("_inject" if args.alpha_inject else ""))
        os.makedirs(out, exist_ok=True)
        li.to_csv(os.path.join(out, "madeeg_alpha_li.csv"), index=False)
        pr.to_csv(os.path.join(out, "madeeg_alpha_pairs.csv"), index=False)
        import math as _m
        k, n = int(pr.correct.sum()), len(pr)
        p = sum(_m.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
        thr = next(c for c in range(n + 1)
                   if sum(_m.comb(n, i) for i in range(c, n + 1)) / 2.0 ** n < 0.05)
        lines = ["== MAD-EEG alpha laterality, TWIN-PAIR sign test =="]
        if args.alpha_inject:
            lines.append(f"🔴 CAVEAT, INSIDE THE FILE AND ABOVE THE NUMBERS: this is the POSITIVE "
                         f"CONTROL. A lateralized alpha modulation of amplitude "
                         f"{args.alpha_inject} was INJECTED into the real EEG with the side taken "
                         f"from the true panning. These numbers say the index and the pairing are "
                         f"wired correctly and say NOTHING about the real data. Not a result.")
        lines += [f"spatial={args.spatial} eeg_clean={args.eeg_clean} band=alpha "
                  f"{alz.ALPHA} pairs={alz.LAT_PAIRS} seed={args.seed}",
                  "RULE, fixed before the run and NOT flipped afterwards: alpha desynchronization "
                  "is CONTRALATERAL, so attending on the right raises log P(right)-log P(left). "
                  "A pair is correct when sign(LI_a - LI_b) == sign(side_a - side_b).",
                  "🔴 NULL = 0.5 and it is EXACT, not estimated: under H0 the two twins are "
                  "EXCHANGEABLE -- same subject, same mixture, same audio, same session, only "
                  "the attended source differs -- so the sign of their LI difference is equally "
                  "likely either way. No constant of subject, channel or impedance can survive: "
                  "it cancels in the difference, algebraically.",
                  f"trials with an LI: {len(li)}   twin pairs with a predicted sign: {n}   "
                  f"(excluded: {dropped_centre} with a CENTRED target -- panning 0.5, no side, "
                  f"excluded a priori by the 9/8 contract -- and {dropped_same} with both "
                  f"targets on the same side)",
                  f"PAIRED ACCURACY: {k}/{n} = {k / n:.4f}   null 0.500   "
                  f"one-sided exact binomial p={p:.4f}",
                  f"pre-registered threshold at n={n}: {thr}/{n} = {thr / n:.4f} "
                  f"(p={sum(_m.comb(n, i) for i in range(thr, n + 1)) / 2.0 ** n:.4f}; {thr - 1} "
                  f"would give {sum(_m.comb(n, i) for i in range(thr - 1, n + 1)) / 2.0 ** n:.4f})",
                  f"-> {'ABOVE' if k >= thr else 'BELOW'} the pre-registered threshold",
                  "per-subject: " + " ".join(f"{s}={v:.2f}"
                                             for s, v in pr.groupby('subject').correct.mean().items())]
        if args.spatial == "mono":
            lines.append("📌 THIS IS THE CONTROL ARM. In mono there is no side physically present, "
                         "so the same nominal panning of the mixture is used and the rule is "
                         "identical: only the spatial separation is removed. The mono arm must "
                         "stay AT CHANCE. If it matches or beats stereo, the spatial reading is "
                         "FALSIFIED even if stereo passes.")
        else:
            lines.append("⚠️ EXPLORATORY BY CONSTRUCTION: the 154 stereo and 155 mono duos are "
                         "already spent (vault: ledger of looks). No threshold written beforehand "
                         "makes this confirmatory, and the thesis section must say so in its "
                         "title, not in a footnote.")
        txt = "\n".join(lines)
        open(os.path.join(out, "madeeg_alpha_summary.txt"), "w").write(txt + "\n")
        print("\n" + txt + f"\n[madeeg] wrote -> {out}")
        return

    if args.own_vs_other:
        # THE REPLACEMENT GATE (vault: Exp. 6 pre-registration, 10/8). Held-out SOLO segments
        # only: no duo is loaded, no attention decision is taken, no trio is touched. The
        # ledger of looks declares this free -- the solos are training material.
        assert args.train_on == "raw_solos", (
            "--own_vs_other scores held-out SOLO segments: it needs --train_on raw_solos")
        # AXIS 2 is NOT wired here, so it must break rather than be dropped in silence: with
        # per-instrument decoders it is ambiguous which decoder should reconstruct the OTHER
        # segment's audio, and an unstated choice there is exactly how the 0.1100 artefact of
        # 2026-07-29 happened. One decoder, one meaning.
        assert args.filters == "pooled", (
            "--own_vs_other is pooled-only: with --filters per_instrument the choice of which "
            "decoder scores the 'other' audio is a methodological decision, not a default")
        # A DEDICATED rng for the split, so the folds and the pairings are identical for every
        # estimator regardless of how much randomness the estimator itself consumed. Without
        # this the paired test would silently compare different comparisons.
        pair_rng = np.random.RandomState(args.seed)
        rows = []
        for subj in sorted(meta):
            raw_chs = list(raw_info[subj]["ch_names"])
            eeg_full_bp = raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band,
                                             args.eeg_clean, raw_info, args.seed, ica_diag)
            trials = build_solo_trials(eeg_full_bp, sq, stim_dir, subj, args.target_fs, band,
                                       args.clamp, args.compression, n_bands, args.target)
            if len(trials) < args.cv_folds + 1:
                print(f"  subj {subj}: only {len(trials)} solo segments -> skip"); continue
            n_pairs = 0
            for f, fold in enumerate(kfold_indices(len(trials), args.cv_folds, pair_rng)):
                te = fold.tolist(); tr = [i for i in range(len(trials)) if i not in set(te)]
                model, _, _ = fit_pool([trials[i] for i in tr], [0.0] * len(tr),
                                       n_lags, lam_grid, rng, shrink, cca_cfg)
                for i, j in mutual_pairs(trials, te):
                    n_pairs += 1
                    for a, b in ((i, j), (j, i)):        # BOTH directions -> null exactly 0.5
                        n_keep = min(trials[a][0].shape[1], trials[b][0].shape[1])
                        sc = [native_score(model, trials[a][0], trials[x][1][0], n_lags, cca_cfg,
                                           ch_names, band, args.target_fs, n_keep) for x in (a, b)]
                        rows.append(dict(subject=subj, fold=f, seg=a, other=b,
                                         instr=trials[a][5], other_instr=trials[b][5],
                                         n_keep=n_keep, s_own=sc[0], s_other=sc[1],
                                         correct=int(sc[0] > sc[1])))
            print(f"  subj {subj}: solo segments={len(trials)} mutual pairs={n_pairs} "
                  f"comparisons={2 * n_pairs}", flush=True)

        ov = pd.DataFrame(rows)
        out = os.path.join(args.log_dir, args.training_date)
        os.makedirs(out, exist_ok=True)
        ov.to_csv(os.path.join(out, "madeeg_ownvsother.csv"), index=False)
        k, n = int(ov.correct.sum()), len(ov)
        from scipy import stats as _st
        p = _st.binomtest(k, n, 0.5, alternative="greater").pvalue
        stat = "rho" if cca_cfg else "band_pearson"
        lines = ["== MAD-EEG own-vs-other on held-out SOLO segments ==",
                 "WHAT THIS IS: for a held-out solo segment, does the estimator score that "
                 "segment's EEG higher against its OWN audio than against another solo's? It is "
                 "discrimination -- the same operation the real decision performs -- but on "
                 "TRAINING material: no duo is loaded and no attention decision is taken. The "
                 "ledger of looks declares it free.",
                 "🔴 THE SIMILARITY IS THIS ESTIMATOR'S OWN and is NOT comparable across "
                 f"estimators (here: {stat}). Only the ACCURACY below is comparable, because it "
                 "is dimensionless. Never put a rho next to a band_pearson.",
                 f"estimator={args.estimator} eeg_clean={args.eeg_clean} filters={args.filters} "
                 f"target={args.target} n_bands={n_bands} target_fs={args.target_fs} "
                 f"band={band} lags={n_lags} folds={args.cv_folds} seed={args.seed}"]
        if cca_cfg is not None:
            lines.append(f"MODEL 9: views={','.join(cca_cfg['views'])} "
                         f"stim_views={','.join(cca_cfg['stim_views'])} k={cca_cfg['k']} "
                         f"reg={cca_cfg['reg']:g}")
        lines += [f"comparisons={n}  (mutual pairs x 2 directions)",
                  "🔴 NULL = 0.5, and it is EXACT BY SYMMETRY, not estimated: every pair is "
                  "scored in both directions, so an estimator with nothing but a fixed "
                  "instrument preference gets exactly one of the two right.",
                  f"own-vs-other accuracy: {k}/{n} = {k / n:.4f}   null 0.500   "
                  f"one-sided binomial p={p:.3g}",
                  "per-subject: " + " ".join(f"{s}={v:.3f}"
                                             for s, v in ov.groupby('subject').correct.mean().items()),
                  f"mean {stat}(own)={ov.s_own.mean():.4f}  mean {stat}(other)={ov.s_other.mean():.4f}",
                  "NEXT: the paired comparison between two estimators is McNemar on the SAME "
                  "comparisons -- python src/madeeg_diagnose.py --mcnemar <ridge.csv> <cca.csv>"]
        txt = "\n".join(lines)
        open(os.path.join(out, "madeeg_ownvsother_summary.txt"), "w").write(txt + "\n")
        print("\n" + txt + f"\n[madeeg] wrote -> {out}")
        return

    def decide(subj, st, test_built, predict, records, inner_val_r=float("nan"), fold=-1,
               instr_map=None, inner_val_rho=float("nan")):
        """`predict` is one callable (pooled) or, under AXIS 2, a dict instrument->callable,
        in which case every candidate source is reconstructed by ITS OWN decoder. Under
        MODEL 9 it is a CCAModel (or a dict of them) and the score is rho(s) instead.

        `instr_map` gives stim -> instruments explicitly; it is required for the mono half,
        whose stim keys have no entry in the preprocessed metadata at all."""
        for i in range(len(test_built)):
            eeg, reps, tgt, npr, ens = test_built[i]
            m = meta.get(subj, {}).get(st[i])                     # absent under --self_test
            instr = (instr_map[st[i]] if instr_map else
                     (list(m["instruments"]) if m else [str(j) for j in range(npr)]))
            extra = {}
            if cca_cfg is not None:
                # MODEL 9: rho(s) = sum of the first k canonical correlations of CCA(X, Y_s).
                # The rule is unchanged -- argmax over the sources PRESENT IN THIS TRIAL, no
                # label space anywhere -- but the statistic is not a band_pearson, so the
                # record says which one it is.
                pos = source_positions(meta, subj, st[i], npr)
                cca = _cca_mod()
                X, _ = cca.eeg_features(eeg, ch_names, cca_cfg["views"], n_lags,
                                        band, args.target_fs, lagged_design)
                pick = ((lambda j: predict[instr[j]]) if isinstance(predict, dict)
                        else (lambda j: predict))
                sims = []
                for j in range(npr):
                    Y, _, _ = cca.stim_features(reps[j], pos[j], cca_cfg["stim_views"])
                    sims.append(pick(j).rho(X, Y))
                extra = dict(score_kind="rho_cca", inner_val_rho=inner_val_rho,
                             target_position=float(pos[tgt]))
            elif callable(predict):
                shat = predict(eeg)
                sims = [band_pearson(shat, reps[j].T) for j in range(npr)]
            else:
                sims = [band_pearson(predict[instr[j]](eeg), reps[j].T) for j in range(npr)]
            pred = int(np.argmax(sims))
            records.append(dict(subject=subj, stim=st[i], ensemble=ens, target_idx=tgt, n_present=npr,
                                pred=pred, correct=int(pred == tgt),
                                r_attended=sims[tgt], r_best_unattended=max(s for j, s in enumerate(sims) if j != tgt),
                                inner_val_r=inner_val_r, fold=fold,
                                target_instr=instr[tgt], pred_instr=instr[pred], **extra))

    if args.self_test:
        # Positive control (see synth_attended_eeg): swap every trial's EEG for a synthetic
        # linear mixture of ITS attended source at lags 0..n_lags + noise, then run the REAL
        # pipeline (fit_ridge_pool + decide) unchanged. Passing proves that chance-level AAD on
        # the real EEG is a genuine absence of decodable signal, not a loader/decoder bug.
        lags_op = W_op = None
        built, stims, built_pos = [], [], []
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
            built_pos.append(source_positions(meta, s, k, npr)[tgt])   # MODEL 9 only
        st_records = []
        pooled = []
        for fold in kfold_indices(len(built), args.cv_folds, rng):
            te = set(fold.tolist()); tr = [i for i in range(len(built)) if i not in te]
            # F1 -- the control has to be crossable on this machine. At target_fs=256 the full
            # pool is ~123 trials x ~7000 samples x 1300 columns = 7.7 GB of design matrix and
            # does not fit in 16 GB, which left AXIS 0 and AXIS 4 at 256 Hz with no positive
            # control crossed at all. Capping the TRAINING pool (never the test fold, so all
            # trials are still decided) makes it fit. Default 0 = no cap = the archived
            # behaviour, so the 64 Hz PASS stays bit-identical.
            if args.self_test_pool and len(tr) > args.self_test_pool:
                tr = sorted(rng.permutation(tr)[:args.self_test_pool].tolist())
            pooled.append(len(tr))
            predict, val_r, val_rho = fit_pool([built[i] for i in tr], [built_pos[i] for i in tr],
                                               n_lags, lam_grid, rng, shrink, cca_cfg)
            decide("synthetic", [stims[i] for i in fold], [built[i] for i in fold], predict,
                   st_records, val_r, inner_val_rho=val_rho)
        rec = pd.DataFrame(st_records)
        acc, ra, ru = rec.correct.mean(), rec.r_attended.mean(), rec.r_best_unattended.mean()
        chance = float(np.mean(1.0 / rec.n_present))
        # Threshold 0.90, declared in the code before the run and unchanged for MODEL 9:
        # applying the SAME decision rule to the envelopes alone decides 154/154 trials, so
        # the signal is separable in principle and only the wiring is on trial here.
        ok = acc >= 0.90 and ra > ru
        who = "CCA" if cca_cfg is not None else "ridge"
        stat = "rho_cca" if cca_cfg is not None else "r"
        head = ["\n== MAD-EEG SELF-TEST (synthetic attended-signal positive control) ==",
                "CAVEAT, INSIDE THE FILE AND ABOVE THE NUMBERS: the EEG here is SYNTHETIC -- a "
                "fixed linear mixture of each trial's ATTENDED source at the model's lags, plus "
                "noise. These numbers say the pipeline is wired correctly and say NOTHING about "
                "the real data. They are not a result and must never be cited as one.",
                f"n_trials={len(rec)} chance={chance:.3f} snr={args.self_test_snr} "
                f"n_ch={n_ch} n_bands={n_bands} lags={n_lags} target_fs={args.target_fs} "
                f"estimator={args.estimator}",
                f"training trials pooled per fold: {pooled}"
                + (f"  (capped at --self_test_pool {args.self_test_pool}; the threshold below is "
                   f"unchanged, a smaller pool only makes the control HARDER to pass)"
                   if args.self_test_pool else "  (full pool)"),
                f"AAD accuracy={acc:.4f}  mean {stat}(attended)={ra:.4f}  "
                f"mean {stat}(best unattended)={ru:.4f}",
                f"[{'PASS' if ok else 'FAIL'}] " + (
                    f"{who} recovers the attended source ({stat}_att >> {stat}_unatt, AAD>=0.90) "
                    "-> pipeline is wired correctly, so chance-level AAD on real EEG is genuine "
                    "signal absence."
                    if ok else
                    f"{who} did NOT recover a signal it should trivially decode -> fix the "
                    "loader/decoder before interpreting the real-data result.")]
        if cca_cfg is not None:
            head.append(f"views={','.join(cca_cfg['views'])} stim_views={','.join(cca_cfg['stim_views'])} "
                        f"k={cca_cfg['k']} reg={cca_cfg['reg']:g} "
                        f"blocks_x={cca_cfg['blocks_x']} blocks_y={cca_cfg['blocks_y']}")
            head.append(f"mean inner_val_r (back-projected, comparable with the ridge)="
                        f"{rec.inner_val_r.mean():.4f}   mean inner_val_rho (canonical, NOT "
                        f"comparable)={rec.inner_val_rho.mean():.4f}")
        txt = "\n".join(head)
        print(txt)
        # A control never writes over a result (Comandamenti #9): its own directory, and the
        # caveat lives in the file, which outlives the terminal.
        st_out = os.path.join(args.log_dir, args.training_date + "_selftest")
        os.makedirs(st_out, exist_ok=True)
        open(os.path.join(st_out, "madeeg_selftest_summary.txt"), "w").write(txt.lstrip("\n") + "\n")
        rec.to_csv(os.path.join(st_out, "madeeg_selftest_records.csv"), index=False)
        print(f"[madeeg] wrote -> {st_out}")
        return

    records, ival = [], []
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
                predict, vrs, vrhos = {}, [], []
                for instrument in sorted(groups):
                    p, vr, vrho = fit_pool(groups[instrument], [0.0] * len(groups[instrument]),
                                           n_lags, lam_grid, rng, shrink, cca_cfg)
                    predict[instrument] = p; vrs.append(vr); vrhos.append(vrho)
                val_r, val_rho = float(np.mean(vrs)), float(np.mean(vrhos))
            else:
                predict, val_r, val_rho = fit_pool(train_trials, [0.0] * len(train_trials),
                                                   n_lags, lam_grid, rng, shrink, cca_cfg)
            if args.inner_val_only:
                # Stop before a single test duo is even BUILT: nothing downstream can produce
                # an accuracy, so this configuration costs no look at the 154/155 spent trials.
                ival.append(dict(subject=subj, fold=-1, n_train=len(train_trials),
                                 inner_val_r=val_r, inner_val_rho=val_rho))
                print(f"  subj {subj}: solos={len(train_trials)} inner_val_r={val_r:.4f}"
                      + (f" inner_val_rho={val_rho:.4f}" if cca_cfg else "")
                      + "  [inner_val_only: no test trial built]", flush=True)
                continue
            # build test duos: raw EEG (no mismatch) or preprocessed EEG
            test_instr = {}
            if args.test_eeg == "raw":
                recipes, dropped_rec = duo_test_recipes(meta, sq, subj, keep, args.spatial, donors)
                for why in sorted({w for _, w in dropped_rec}):
                    n_why = sum(1 for _, w in dropped_rec if w == why)
                    print(f"  subj {subj}: {n_why} trial(s) not built -- {why}")
                test_built, st_used = [], []
                for rec_key, raw_key, (ds_, dk_), instr, target in recipes:
                    dm = meta[ds_][dk_]
                    t = build_duo_trial_raweeg(eeg_full_bp, read_f64(data[ds_][dk_]["soli"]),
                                               sq[subj][raw_key]["n_ech"], instr.index(target), len(instr),
                                               dm["wav_info"]["sfreq"], args.target_fs, band, args.clamp,
                                               args.compression, n_bands, args.target)
                    if t is not None:
                        test_built.append(t); st_used.append(rec_key); test_instr[rec_key] = instr
                n_candidates = len(recipes) + len(dropped_rec)
            else:
                st_used = st
                test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands, args.target) for k in st]
                test_instr = {k: list(meta[subj][k]["instruments"]) for k in st}
                n_candidates = len(st)
            if isinstance(predict, dict):
                # AXIS 2 -- the test is `is this a per-instrument dict`, NOT `is this not a
                # callable`. A MODEL 9 pooled decoder is a CCAModel: not a dict, and not a
                # callable either, so the old negative test sent it in here and `x in predict`
                # raised TypeError on the first trial. Positive tests, always.
                # A trial is undecidable when one of its sources has no solo for this
                # subject (subject 0007 recorded 7 solos, not 14). Dropped and COUNTED, never
                # silently reconstructed with somebody else's filter.
                keepable = [i for i, k in enumerate(st_used)
                            if all(x in predict for x in test_instr[k])]
                dropped = len(st_used) - len(keepable)
                if dropped:
                    print(f"  subj {subj}: {dropped} trial(s) dropped -- no solo decoder for one of their sources")
                test_built = [test_built[i] for i in keepable]; st_used = [st_used[i] for i in keepable]
            decide(subj, st_used, test_built, predict, records, val_r, instr_map=test_instr,
                   inner_val_rho=val_rho)
            print(f"  subj {subj}: solos={len(train_trials)} test={len(test_built)}/{n_candidates} "
                  f"spatial={args.spatial} test_eeg={args.test_eeg} "
                  f"inner_val_r={val_r:.3f} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)
        else:
            test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands, args.target) for k in st]
            if len(st) < args.cv_folds:
                print(f"  subj {subj}: only {len(st)} trials -> skip"); continue
            folds = kfold_indices(len(test_built), args.cv_folds, rng)
            for f in range(args.cv_folds):
                te = set(folds[f].tolist()); tr = [i for i in range(len(test_built)) if i not in te]
                # MODEL 9: here the training trials ARE duos, so the attended source has a real
                # side. (Under --train_on raw_solos they are solos, all mono -> position 0.)
                tr_pos = [source_positions(meta, subj, st[i], test_built[i][3])[test_built[i][2]]
                          for i in tr]
                predict, val_r, val_rho = fit_pool([test_built[i] for i in tr], tr_pos,
                                                   n_lags, lam_grid, rng, shrink, cca_cfg)
                if args.inner_val_only:          # fit, score the inner split, decide nothing
                    ival.append(dict(subject=subj, fold=f, n_train=len(tr),
                                     inner_val_r=val_r, inner_val_rho=val_rho))
                    continue
                decide(subj, [st[i] for i in folds[f]], [test_built[i] for i in folds[f]], predict,
                       records, val_r, fold=f, inner_val_rho=val_rho)
            if args.inner_val_only:
                print(f"  subj {subj}: trials={len(st)} inner_val_r="
                      f"{np.mean([r['inner_val_r'] for r in ival if r['subject'] == subj]):.4f}"
                      "  [inner_val_only: nothing decided]", flush=True)
            else:
                print(f"  subj {subj}: trials={len(st)} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)

    out = os.path.join(args.log_dir, args.training_date)
    os.makedirs(out, exist_ok=True)

    if args.inner_val_only:
        iv = pd.DataFrame(ival)
        iv.to_csv(os.path.join(out, "madeeg_innerval.csv"), index=False)
        head = ["== MAD-EEG inner-validation ONLY (no trial was decided) ==",
                "CAVEAT, INSIDE THE FILE AND ABOVE THE NUMBERS: this run built no test trial "
                "and produced no attention decision. There is no accuracy here and there is "
                "none on disk. inner_val_r is reconstruction quality on an inner split of the "
                "TRAINING material -- it says how well the decoder reconstructs, never whether "
                "it selects the attended source.",
                f"train_on={args.train_on} ensemble={args.ensemble} target={args.target} "
                f"n_bands={n_bands} target_fs={args.target_fs} lags={n_lags} band={band} "
                f"estimator={args.estimator} filters={args.filters} eeg_clean={args.eeg_clean} "
                f"spatial={args.spatial} folds={args.cv_folds}"]
        if cca_cfg is not None:
            head += [f"MODEL 9: views={','.join(cca_cfg['views'])} "
                     f"stim_views={','.join(cca_cfg['stim_views'])} k={cca_cfg['k']} "
                     f"reg={cca_cfg['reg']:g} blocks_x={cca_cfg['blocks_x']} "
                     f"blocks_y={cca_cfg['blocks_y']}",
                     "🔴 inner_val_r is BACK-PROJECTED and comparable with the ridge; "
                     "inner_val_rho is a sum of canonical correlations and is NOT comparable "
                     "with any ridge number (a canonical correlation is >= a Pearson by "
                     "construction)."]
        head += ["WEIGHTING, and it is not the same as a full run's: the mean below is over "
                 "(subject x fold) UNWEIGHTED, because no test trial was built and the "
                 "per-subject trial counts are unknown here. A full run averages the same "
                 "quantity over TRIALS, so the two differ whenever subjects contribute "
                 "unequal numbers of trials (raw_solos ridge: 0.0562 here vs 0.0582 "
                 "trial-weighted). Compare inner_val_only against inner_val_only.",
                 f"units={len(iv)} (subject x fold)",
                 f"mean inner_val_r = {iv.inner_val_r.mean():.4f}",
                 "per-subject inner_val_r: " + " ".join(
                     f"{s}={v:.4f}" for s, v in iv.groupby('subject').inner_val_r.mean().items())]
        if cca_cfg is not None:
            head.append(f"mean inner_val_rho = {iv.inner_val_rho.mean():.4f}  (NOT comparable)")
        txt = "\n".join(head)
        open(os.path.join(out, "madeeg_innerval_summary.txt"), "w").write(txt + "\n")
        print("\n" + txt + f"\n[madeeg] wrote -> {out}")
        return

    rec = pd.DataFrame(records)
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
             f"filters={args.filters} eeg_clean={args.eeg_clean} test_eeg={args.test_eeg} "
             f"spatial={args.spatial}",
             # F3 -- provenance of the analysis band, inside the file that carries the numbers.
             # The WASPAA 2019 paper describes its whole EEG preprocessing (50 Hz notch, ICA on
             # EOG/ECG) and never band-passes anywhere; the band below is OURS.
             f"PROVENANCE: the {band[0]}-{band[1]} Hz analysis band is OUR choice, not the "
             f"paper's -- Cantisani et al. (WASPAA 2019) do not band-pass the EEG anywhere. "
             f"lags 0-{args.lags_ms:g} ms and shrinkage lambda are the paper's; n_mels and "
             f"target_fs are ours unless set to the published 24 / 256.",
             f"n_trials={len(rec)}  chance={chance:.3f}",
             f"OVERALL AAD accuracy: {rec.correct.mean():.4f}",
             "per-subject: " + " ".join(f"{s}={v:.2f}" for s, v in per_subj.items()),
             f"F1 over the attended-instrument label: micro={f1s['micro']:.4f} (= accuracy) "
             f"macro={f1s['macro']:.4f} weighted={f1s['weighted']:.4f}",
             "  ^ the paper (WASPAA 2019 Table 1) reports F1 without naming the averaging;"
             " its duets column is AE 58 / MAG 74 / MEL 79",
             f"mean {'rho_cca' if cca_cfg else 'r'}(attended)={rec.r_attended.mean():.4f}  "
             f"mean {'rho_cca' if cca_cfg else 'r'}(best unattended)={rec.r_best_unattended.mean():.4f}",
             f"mean inner_val_r (in-distribution reconstruction)={rec.inner_val_r.mean():.4f}"]
    if cca_cfg is not None:
        # MODEL 9. The caveat goes INSIDE the file and ABOVE the numbers (Comandamenti #9):
        # this file is what survives the terminal, and the two statistics below look alike.
        lines.insert(1, "\n".join([
            f"MODEL 9 (multi-view CCA): views={','.join(cca_cfg['views'])} "
            f"stim_views={','.join(cca_cfg['stim_views'])} k={cca_cfg['k']} reg={cca_cfg['reg']:g} "
            f"blocks_x={cca_cfg['blocks_x']} blocks_y={cca_cfg['blocks_y']}",
            "  band-power/lateralization views use only the declared bands that fit inside the "
            "analysis band above; a band outside it is ZERO by construction and is dropped, not "
            "carried as noise. Widening the band is a separate, declared change (--band_high).",
            "🔴 TWO STATISTICS IN THIS FILE, AND THEY ARE NOT COMPARABLE:",
            "   inner_val_r = band_pearson between the BACK-PROJECTED reconstruction and the "
            "attended representation, computed with the SAME function the ridge uses. THIS is "
            "the number to put next to the ridge's inner_val_r (0.0582 on raw_solos, 0.0245 on "
            "duos_kfold).",
            "   rho_cca / inner_val_rho = the sum of the first k canonical correlations. A "
            "canonical correlation is >= a plain Pearson BY CONSTRUCTION -- CCA optimizes both "
            "sides to maximize it -- so comparing it with any ridge number shows an improvement "
            "that does not exist. It is the DECISION statistic and nothing else.",
            "   The position view is constant in time within a trial, so a within-trial Pearson "
            "centres it out: it cannot change the decision. Verified as an assertion in "
            "src/models/cca_multiview.py. position_env is the time-varying form (still 0 for "
            "every source in mono)."]))
        lines.append(f"mean inner_val_rho (canonical, NOT comparable with the ridge)="
                     f"{rec.inner_val_rho.mean():.4f}")
        lines.append(f"trials by target side (from wav_info.panning): "
                     + " ".join(f"{int(k):+d}:{v}" for k, v in
                                sorted(rec.target_position.value_counts().items())))
    if ica_diag:
        lines.append("EEG cleaning (AXIS 1) per subject -- positive control is the drop in "
                     "frontal-EOG coupling:")
        lines += [f"  {s}: {v}" for s, v in sorted(ica_diag.items())]
    txt = "\n".join(lines)
    open(os.path.join(out, "madeeg_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[madeeg] wrote -> {out}")


if __name__ == "__main__":
    main()
