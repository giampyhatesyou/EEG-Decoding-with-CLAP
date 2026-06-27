# CHANGED(baseline): NEW FILE — POSITIVE thesis arm. Stimulus-reconstruction (backward
#   model) AAD on the MAD-EEG dataset (Cantisani et al., SMM 2019; Zenodo 4537751).
#   Unlike the original (Akama) dataset, MAD-EEG presents the SAME mixture with DIFFERENT
#   attended instruments across trials -> within-stimulus contrast -> attention decoding
#   is IDENTIFIABLE. The decision is internal to one trial (reconstruction vs the present
#   sources of the SAME mixture), so it cannot be won by stimulus identity.
#
#   Self-contained on purpose (does not import the original-dataset pipeline). Reuses the
#   same backward-model math as src/stimulus_reconstruction.py. No new deps beyond h5py
#   (already pulled in by the stack) / pyyaml / scipy / sklearn / torch.
#
# Schema (verified from tutorial-MAD-EEG.ipynb):
#   data[subj][stim]['response'] : EEG (20, M) @ 256 Hz  (ch F3..O2, incl. occipital)
#   data[subj][stim]['soli']     : isolated sources (3, N) @ 44100 (3rd row empty if duo),
#                                  ordered as metadata[subj][stim]['instruments']
#   metadata[subj][stim] : {target, instruments[list], ensemble(duo/trio), spatial,
#                           genre, song, theme, eeg_info{sfreq:256}, wav_info{sfreq:44100}}
#
# Method: per subject, k-fold over duo (and/or trio) trials; ridge (or CNN) backward model
#   reconstructs the ATTENDED source envelope from EEG (band-pass 1-8 Hz, downsample to
#   64 Hz, lags 0-250 ms); at test, argmax of Pearson r with each PRESENT source envelope.
#   Chance = 1/n_present (0.50 duo, 0.33 trio). Target to beat (paper, linear, duets): ~78%.

import os
import argparse
import numpy as np
import h5py
import yaml
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt, resample
from sklearn.preprocessing import RobustScaler, StandardScaler

EEG_FS = 256


# --------------------------------------------------------------------------- #
# backward-model primitives (same math as stimulus_reconstruction.py)
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


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


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
    """EEG (20, M)@256 -> band-pass -> per-channel RobustScaler+clamp -> resample to target_fs.
    Returns (20, L)."""
    x = np.asarray(resp, dtype=np.float64)
    x = bandpass(x, EEG_FS, band[0], band[1])
    for i in range(x.shape[0]):
        x[i] = RobustScaler().fit(x[i].reshape(-1, 1)).transform(x[i].reshape(-1, 1)).reshape(-1)
    x = np.clip(x, -clamp, clamp)
    L = int(round(x.shape[1] * target_fs / EEG_FS))
    return resample(x, L, axis=1), L


def source_envelope(src, src_fs, out_len, target_fs, band, compression):
    """Isolated source (1-D @44100) -> Hilbert envelope -> anti-alias -> resample to out_len
    -> band-pass @target_fs (matched to EEG) -> z-score."""
    x = np.asarray(src, dtype=np.float64)
    if not np.any(x):
        return np.zeros(out_len)
    env = np.abs(hilbert(x))
    if compression and compression != 1.0:
        env = np.power(np.maximum(env, 0.0), compression)
    nyq = 0.5 * src_fs
    b, a = butter(4, min(20.0, 0.99 * nyq) / nyq, btype="low")
    env = filtfilt(b, a, env)
    env = resample(env, out_len)
    env = bandpass(env, target_fs, band[0], band[1])
    s = env.std()
    return (env - env.mean()) / (s + 1e-8)


def build_trial(data, meta, subj, stim, target_fs, band, clamp, compression):
    m = meta[subj][stim]
    instruments = list(m["instruments"])
    n_present = len(instruments)
    target_idx = instruments.index(m["target"])
    eeg, L = process_eeg(data[subj][stim]["response"][:], target_fs, band, clamp)
    soli = data[subj][stim]["soli"][:]
    envs = np.stack([source_envelope(soli[i], m["wav_info"]["sfreq"], L, target_fs, band, compression)
                     for i in range(n_present)], axis=0)
    return eeg, envs, target_idx, n_present, m["ensemble"]


# --------------------------------------------------------------------------- #
# decoder (ridge) + AAD decision
# --------------------------------------------------------------------------- #
def fit_ridge_pool(train_trials, n_lags, lam_grid, rng):
    """Train a backward model on attended-source envelopes pooled over training trials."""
    Xs = [lagged_design(t[0], n_lags) for t in train_trials]
    ys = [t[1][t[2]] for t in train_trials]                       # attended env per trial
    tr_idx, val_idx = _inner_split(len(train_trials), rng)
    sc = StandardScaler().fit(np.concatenate([Xs[i] for i in tr_idx], 0))
    Xtr = sc.transform(np.concatenate([Xs[i] for i in tr_idx], 0))
    ytr = np.concatenate([ys[i] for i in tr_idx], 0)
    best_lam, best_r = lam_grid[0], -np.inf
    for lam in lam_grid:
        w = ridge_fit(Xtr, ytr, lam)
        r = float(np.mean([pearson(sc.transform(Xs[i]) @ w, ys[i]) for i in val_idx]))
        if r > best_r:
            best_r, best_lam = r, lam
    sc = StandardScaler().fit(np.concatenate(Xs, 0))
    w = ridge_fit(sc.transform(np.concatenate(Xs, 0)), np.concatenate(ys, 0), best_lam)

    def predict(eeg):
        return sc.transform(lagged_design(eeg, n_lags)) @ w
    return predict, best_lam, best_r


def kfold_indices(n, k, rng):
    idx = np.arange(n); rng.shuffle(idx)
    return [idx[i::k] for i in range(k)]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="MAD-EEG stimulus-reconstruction AAD")
    ap.add_argument("--madeeg_dir", required=True, help="dir with madeeg_preprocessed.hdf5 + .yaml")
    ap.add_argument("--log_dir", default="../results")
    ap.add_argument("--training_date", default="madeeg_recon")
    ap.add_argument("--inspect", type=int, default=0, help="print schema for first N trials and exit")
    ap.add_argument("--ensemble", default="duo", choices=["duo", "trio", "both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--target_fs", type=float, default=64.0)
    ap.add_argument("--band_low", type=float, default=1.0)
    ap.add_argument("--band_high", type=float, default=8.0)
    ap.add_argument("--lags_ms", type=float, default=250.0)
    ap.add_argument("--clamp", type=float, default=20.0)
    ap.add_argument("--compression", type=float, default=0.3)
    args = ap.parse_args()

    h5 = os.path.join(args.madeeg_dir, "madeeg_preprocessed.hdf5")
    yml = os.path.join(args.madeeg_dir, "madeeg_preprocessed.yaml")
    data = h5py.File(h5, "r")
    meta = yaml.load(open(yml), Loader=yaml.FullLoader)
    band = (args.band_low, args.band_high)
    rng = np.random.RandomState(args.seed)
    n_lags = int(round(args.lags_ms / 1000.0 * args.target_fs))

    # enumerate trials
    keep = {"duo"} if args.ensemble == "duo" else ({"trio"} if args.ensemble == "trio" else {"duo", "trio"})
    trials = [(s, k) for s in meta for k in meta[s] if meta[s][k].get("ensemble") in keep]
    print(f"[madeeg] subjects={list(meta.keys())} ensemble={args.ensemble} n_trials={len(trials)} "
          f"target_fs={args.target_fs} lags={n_lags} band={band}")

    if args.inspect:
        for s, k in trials[:args.inspect]:
            m = meta[s][k]
            print(f"\n  subj={s} stim={k}\n    ensemble={m['ensemble']} instruments={m['instruments']} "
                  f"target={m['target']} spatial={m['spatial']}")
            print(f"    response={data[s][k]['response'].shape}@{m['eeg_info']['sfreq']}Hz "
                  f"soli={data[s][k]['soli'].shape}@{m['wav_info']['sfreq']}Hz")
        return

    lam_grid = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
    records = []
    for subj in meta:
        st = [k for k in meta[subj] if meta[subj][k].get("ensemble") in keep]
        if len(st) < args.cv_folds:
            print(f"  subj {subj}: only {len(st)} trials -> skip"); continue
        built = [build_trial(data, meta, subj, k, args.target_fs, band, args.clamp, args.compression) for k in st]
        folds = kfold_indices(len(built), args.cv_folds, rng)
        for f in range(args.cv_folds):
            te = list(folds[f]); tr = [i for i in range(len(built)) if i not in set(te)]
            predict, lam, val_r = fit_ridge_pool([built[i] for i in tr], n_lags, lam_grid, rng)
            for i in te:
                eeg, envs, tgt, npr, ens = built[i]
                shat = predict(eeg)
                sims = [pearson(shat, envs[j]) for j in range(npr)]
                records.append(dict(subject=subj, stim=st[i], ensemble=ens, target_idx=tgt, n_present=npr,
                                    pred=int(np.argmax(sims)), correct=int(np.argmax(sims) == tgt),
                                    r_attended=sims[tgt], r_best_unattended=max(s for j, s in enumerate(sims) if j != tgt)))
        acc = np.mean([r["correct"] for r in records if r["subject"] == subj])
        print(f"  subj {subj}: trials={len(st)} AAD_acc={acc:.3f}")

    rec = pd.DataFrame(records)
    out = os.path.join(args.log_dir, args.training_date)
    os.makedirs(out, exist_ok=True)
    rec.to_csv(os.path.join(out, "madeeg_records.csv"), index=False)
    chance = float(np.mean(1.0 / rec.n_present))
    per_subj = rec.groupby("subject").correct.mean()
    lines = ["== MAD-EEG stimulus-reconstruction AAD ==",
             f"ensemble={args.ensemble} target_fs={args.target_fs} lags={n_lags} band={band} folds={args.cv_folds}",
             f"n_trials={len(rec)}  chance={chance:.3f}",
             f"OVERALL AAD accuracy: {rec.correct.mean():.4f}",
             "per-subject: " + " ".join(f"{s}={v:.2f}" for s, v in per_subj.items()),
             f"mean r(attended)={rec.r_attended.mean():.4f}  mean r(best unattended)={rec.r_best_unattended.mean():.4f}"]
    txt = "\n".join(lines)
    open(os.path.join(out, "madeeg_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[madeeg] wrote -> {out}")


if __name__ == "__main__":
    main()
