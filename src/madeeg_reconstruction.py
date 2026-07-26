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

EEG_FS = 256


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


def ridge_fit(X, Y, lam):
    """Closed-form ridge, multi-output: W = (X^T X + lam I)^-1 X^T Y. Y may be (T,) or (T,k)."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


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


def source_repr(src, src_fs, out_len, target_fs, band, compression, n_mels):
    """Isolated source (1-D @44100) -> representation (n_bands, out_len), band-matched to EEG.
    n_mels<=1 -> broadband Hilbert envelope; else -> log-mel spectrogram (per-band temporal
    series). Each band is band-passed to the EEG band and z-scored."""
    x = np.asarray(src, dtype=np.float64)
    if not np.any(x):
        return np.zeros((max(1, n_mels), out_len))
    if n_mels <= 1:
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


def build_trial(data, meta, subj, stim, target_fs, band, clamp, compression, n_mels):
    m = meta[subj][stim]
    instruments = list(m["instruments"])
    n_present = len(instruments)
    target_idx = instruments.index(m["target"])
    eeg, L = process_eeg(read_f64(data[subj][stim]["response"]), target_fs, band, clamp)
    soli = read_f64(data[subj][stim]["soli"])
    reps = np.stack([source_repr(soli[i], m["wav_info"]["sfreq"], L, target_fs, band, compression, n_mels)
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


def raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band):
    """Read the continuous raw EEG for a subject, reorder the 20 EEG channels to the
    preprocessed order, band-pass once. Returns (20, N)."""
    idx = [raw_chs.index(c) for c in pre_chs]
    return bandpass(read_f64(raw_f[subj])[idx], EEG_FS, band[0], band[1])


def build_solo_trials(eeg_full_bp, sq, stimuli_dir, subj, target_fs, band, clamp, compression, n_bands):
    """Paper protocol training pairs: each solo stimulus repetition is cut from the
    band-passed continuous raw EEG at its n_ech sample index and paired with the solo
    source audio (from stimuli/). Returns trials [eeg(20,L), reps(1,n_bands,L), 0, 1, 'solo']."""
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
            rep = source_repr(audio, sr, L, target_fs, band, compression, n_bands)
            trials.append([eeg_proc, rep[None, ...], 0, 1, "solo"])
    return trials


def build_duo_trial_raweeg(eeg_full_bp, soli, n_ech, target_idx, n_present, src_fs,
                           target_fs, band, clamp, compression, n_bands):
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
    reps = np.stack([source_repr(soli[i], src_fs, L, target_fs, band, compression, n_bands)
                     for i in range(n_present)], axis=0)
    return [eeg_proc, reps, target_idx, n_present, "duo"]


# --------------------------------------------------------------------------- #
# decoder (ridge) + AAD decision
# --------------------------------------------------------------------------- #
def fit_ridge_pool(train_trials, n_lags, lam_grid, rng):
    """Backward model on the ATTENDED source representation, pooled over training trials.
    Multi-output (one column per band). Returns (predict_fn, lambda, inner_val_r)."""
    Xs = [lagged_design(t[0], n_lags) for t in train_trials]
    Ys = [t[1][t[2]].T for t in train_trials]                       # (T, n_bands) attended rep
    tr_idx, val_idx = _inner_split(len(train_trials), rng)
    sc = StandardScaler().fit(np.concatenate([Xs[i] for i in tr_idx], 0))
    Xtr = sc.transform(np.concatenate([Xs[i] for i in tr_idx], 0))
    Ytr = np.concatenate([Ys[i] for i in tr_idx], 0)
    best_lam, best_r = lam_grid[0], -np.inf
    for lam in lam_grid:
        W = ridge_fit(Xtr, Ytr, lam)
        r = float(np.mean([band_pearson(sc.transform(Xs[i]) @ W, Ys[i]) for i in val_idx]))
        if r > best_r:
            best_r, best_lam = r, lam
    sc = StandardScaler().fit(np.concatenate(Xs, 0))
    W = ridge_fit(sc.transform(np.concatenate(Xs, 0)), np.concatenate(Ys, 0), best_lam)

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
    ap.add_argument("--target", default="mel", choices=["mel", "envelope"], help="reconstruction target")
    ap.add_argument("--n_mels", type=int, default=8)
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
    n_bands = 1 if args.target == "envelope" else args.n_mels

    keep = {"duo"} if args.ensemble == "duo" else ({"trio"} if args.ensemble == "trio" else {"duo", "trio"})
    trials = [(s, k) for s in meta for k in meta[s] if meta[s][k].get("ensemble") in keep]
    print(f"[madeeg] subjects={list(meta.keys())} ensemble={args.ensemble} n_trials={len(trials)} "
          f"target={args.target} n_bands={n_bands} target_fs={args.target_fs} lags={n_lags} band={band}")

    if args.inspect:
        for s, k in trials[:args.inspect]:
            m = meta[s][k]
            print(f"\n  subj={s} stim={k}\n    ensemble={m['ensemble']} instruments={m['instruments']} "
                  f"target={m['target']} spatial={m['spatial']}")
            print(f"    response={data[s][k]['response'].shape}@{m['eeg_info']['sfreq']}Hz "
                  f"soli={data[s][k]['soli'].shape}@{m['wav_info']['sfreq']}Hz")
        return

    lam_grid = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]

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

    def decide(subj, st, test_built, predict, records, inner_val_r=float("nan")):
        for i in range(len(test_built)):
            eeg, reps, tgt, npr, ens = test_built[i]
            shat = predict(eeg)
            sims = [band_pearson(shat, reps[j].T) for j in range(npr)]
            records.append(dict(subject=subj, stim=st[i], ensemble=ens, target_idx=tgt, n_present=npr,
                                pred=int(np.argmax(sims)), correct=int(np.argmax(sims) == tgt),
                                r_attended=sims[tgt], r_best_unattended=max(s for j, s in enumerate(sims) if j != tgt),
                                inner_val_r=inner_val_r))

    if args.self_test:
        # Positive control (see synth_attended_eeg): swap every trial's EEG for a synthetic
        # linear mixture of ITS attended source at lags 0..n_lags + noise, then run the REAL
        # pipeline (fit_ridge_pool + decide) unchanged. Passing proves that chance-level AAD on
        # the real EEG is a genuine absence of decodable signal, not a loader/decoder bug.
        lags_op = W_op = None
        built, stims = [], []
        for s, k in trials:
            eeg, reps, tgt, npr, ens = build_trial(data, meta, s, k, args.target_fs, band,
                                                    args.clamp, args.compression, n_bands)
            if W_op is None:                       # one fixed forward operator, shared by all trials
                n_ch = eeg.shape[0]
                lags_op = rng.randint(0, n_lags + 1, size=n_ch)
                W_op = rng.standard_normal((n_ch, n_bands))
            built.append([synth_attended_eeg(reps[tgt], lags_op, W_op, rng, args.self_test_snr),
                          reps, tgt, npr, ens])
            stims.append(k)
        st_records = []
        for fold in kfold_indices(len(built), args.cv_folds, rng):
            te = set(fold.tolist()); tr = [i for i in range(len(built)) if i not in te]
            predict, _, val_r = fit_ridge_pool([built[i] for i in tr], n_lags, lam_grid, rng)
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
            eeg_full_bp = raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band)
            train_trials = build_solo_trials(eeg_full_bp, sq, stim_dir, subj,
                                             args.target_fs, band, args.clamp, args.compression, n_bands)
            if len(train_trials) < 3:
                print(f"  subj {subj}: only {len(train_trials)} solo segments -> skip"); continue
            predict, lam, val_r = fit_ridge_pool(train_trials, n_lags, lam_grid, rng)
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
                                               args.compression, n_bands)
                    if t is not None:
                        test_built.append(t); st_used.append(k)
            else:
                st_used = st
                test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands) for k in st]
            decide(subj, st_used, test_built, predict, records, val_r)
            print(f"  subj {subj}: solos={len(train_trials)} test={len(test_built)}/{len(st)} test_eeg={args.test_eeg} "
                  f"inner_val_r={val_r:.3f} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)
        else:
            test_built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression, n_bands) for k in st]
            if len(st) < args.cv_folds:
                print(f"  subj {subj}: only {len(st)} trials -> skip"); continue
            folds = kfold_indices(len(test_built), args.cv_folds, rng)
            for f in range(args.cv_folds):
                te = set(folds[f].tolist()); tr = [i for i in range(len(test_built)) if i not in te]
                predict, lam, val_r = fit_ridge_pool([test_built[i] for i in tr], n_lags, lam_grid, rng)
                decide(subj, [st[i] for i in folds[f]], [test_built[i] for i in folds[f]], predict, records, val_r)
            print(f"  subj {subj}: trials={len(st)} AAD_acc={np.mean([r['correct'] for r in records if r['subject']==subj]):.3f}", flush=True)

    rec = pd.DataFrame(records)
    out = os.path.join(args.log_dir, args.training_date)
    os.makedirs(out, exist_ok=True)
    rec.to_csv(os.path.join(out, "madeeg_records.csv"), index=False)
    chance = float(np.mean(1.0 / rec.n_present))
    per_subj = rec.groupby("subject").correct.mean()
    lines = ["== MAD-EEG stimulus-reconstruction AAD ==",
             f"train_on={args.train_on} ensemble={args.ensemble} target={args.target} n_bands={n_bands} "
             f"target_fs={args.target_fs} lags={n_lags} band={band} folds={args.cv_folds}",
             f"n_trials={len(rec)}  chance={chance:.3f}",
             f"OVERALL AAD accuracy: {rec.correct.mean():.4f}",
             "per-subject: " + " ".join(f"{s}={v:.2f}" for s, v in per_subj.items()),
             f"mean r(attended)={rec.r_attended.mean():.4f}  mean r(best unattended)={rec.r_best_unattended.mean():.4f}",
             f"mean inner_val_r (in-distribution reconstruction)={rec.inner_val_r.mean():.4f}"]
    txt = "\n".join(lines)
    open(os.path.join(out, "madeeg_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[madeeg] wrote -> {out}")


if __name__ == "__main__":
    main()
