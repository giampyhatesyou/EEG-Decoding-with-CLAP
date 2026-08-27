"""ARM D -- the window-CV leakage audit on MAD-EEG (arm D of the 2026-08-11 plan).

Niu et al. 2024 (Neuroscience Letters) report 92.6% duo attention decoding on MAD-EEG
with 1-second windows and a CSP+DNN pipeline. On the same data the honest signal-family
methods reach ~0.55 on whole trials, so a 92.6% at 1 s has the exact profile of the
window-level cross-validation leakage denounced by Rotaru et al. (JNE 2024) and Yan et
al. (Interspeech 2025): windows of the SAME trial land in both train and test, and the
classifier reads the trial's fingerprint, not attention.

THE AUDIT, declared before the run:
  * unit = 1-s non-overlapping windows of the 154 stereo duo trials (preprocessed
    release), per subject; features = log-variance per channel in 5 bands
    (1-4, 4-8, 8-13, 13-20, 20-40 Hz), classifier = LDA. No deep net needed: if leakage
    is present, even LDA inflates.
  * TRUE-LABEL arm: y = attended instrument. Two CVs: WINDOW-level StratifiedKFold(10)
    (the leaky protocol) vs GroupKFold(10) grouped by TRIAL (leak-free at trial level).
    Null = the majority-class rate, printed next to every accuracy (never assume 0.5).
  * PSEUDO-LABEL arm (Yan-style, the smoking gun and the audit's own positive control):
    y = a BALANCED random binary label per trial -- exactly half the subject's trials
    get 1, via a seeded permutation -- so the null is 0.500 EXACTLY by construction. It
    carries zero attention information: trial-level CV must sit at 0.5; any window-level
    accuracy above it measures pure trial-fingerprint leakage.
    (v1 of this audit drew one independent random bit per trial; with ~19 trials per
    subject the majority-class null landed at 0.64-0.80 and made the comparison weak.
    The printed null caught the miscalibration -- outputs preserved as *_v1_unbalanced*
    -- and the balanced draw fixes the construction. The interpretation rule below is
    unchanged.)
  * INTERPRETATION RULE (written here, before the numbers): leakage is demonstrated if
    pseudo-label window-CV accuracy > 0.60 while pseudo-label trial-CV lies in
    [0.40, 0.60]. The true-label pair then quantifies how much of a window-CV "attention"
    number is fingerprint.

EXPLORATORY BY CONSTRUCTION: the duos are spent (ledger of looks). This audit re-reads
spent material and is labeled so; it produces a methods claim, not an attention claim.

Run (CPU, ~2'):
  python src/madeeg_leakage_audit.py --madeeg_dir ~/madeeg
"""
import os
import argparse
import numpy as np
import h5py
import yaml
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GroupKFold

BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 40)]
FS = 256
WIN = FS  # 1-second windows, the protocol under audit


def band_logvar(eeg):
    """(n_ch, T) trial -> (n_windows, n_ch * n_bands) log-variance features."""
    feats = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
        x = filtfilt(b, a, eeg, axis=1)
        n_win = x.shape[1] // WIN
        w = x[:, :n_win * WIN].reshape(x.shape[0], n_win, WIN)
        feats.append(np.log(w.var(axis=2) + 1e-12).T)          # (n_win, n_ch)
    return np.concatenate(feats, axis=1)                        # (n_win, n_ch*n_bands)


def cv_acc(X, y, groups, level, seed):
    """Mean accuracy under window-level (stratified, leaky) or trial-level (grouped) CV."""
    if len(np.unique(y)) < 2:
        return np.nan
    accs = []
    if level == "window":
        splits = StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)
    else:
        n_groups = len(np.unique(groups))
        splits = GroupKFold(min(10, n_groups)).split(X, y, groups)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LinearDiscriminantAnalysis()
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--log_dir", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(args.log_dir or os.path.join(root, "runs", "results"), "armD_leakage_audit")
    os.makedirs(out, exist_ok=True)

    f = h5py.File(os.path.join(os.path.expanduser(args.madeeg_dir), "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.safe_load(open(os.path.join(os.path.expanduser(args.madeeg_dir), "madeeg_preprocessed.yaml")))
    rng = np.random.RandomState(args.seed)

    rows = []
    for subj in sorted(f.keys()):
        duo_stims = [k for k in sorted(f[subj].keys())
                     if meta.get(subj, {}).get(k, {}).get("ensemble") == "duo"]
        # Balanced pseudo-labels: exactly half the trials get 1, so null = 0.500 exact.
        pseudo_bits = np.zeros(len(duo_stims), dtype=int)
        pseudo_bits[:len(duo_stims) // 2] = 1
        pseudo_bits = pseudo_bits[rng.permutation(len(duo_stims))]
        Xs, y_true, y_rand, grp = [], [], [], []
        n_trials = 0
        for stim in duo_stims:
            m = meta[subj][stim]
            eeg = np.asarray(f[subj][stim]["response"], dtype=np.float64)
            feats = band_logvar(eeg)
            Xs.append(feats)
            y_true += [m["target"]] * len(feats)
            y_rand += [int(pseudo_bits[n_trials])] * len(feats)
            grp += [n_trials] * len(feats)
            n_trials += 1
        if n_trials < 10:
            continue
        X = np.concatenate(Xs)
        y_true, y_rand, grp = map(np.array, (y_true, y_rand, grp))
        null_true = max(np.mean(y_true == c) for c in np.unique(y_true))
        null_rand = max(np.mean(y_rand == c) for c in np.unique(y_rand)) if len(np.unique(y_rand)) > 1 else 1.0
        row = dict(subject=subj, n_trials=n_trials, n_windows=len(X),
                   null_true=null_true, null_rand=null_rand,
                   true_window=cv_acc(X, y_true, grp, "window", args.seed),
                   true_trial=cv_acc(X, y_true, grp, "trial", args.seed),
                   rand_window=cv_acc(X, y_rand, grp, "window", args.seed),
                   rand_trial=cv_acc(X, y_rand, grp, "trial", args.seed))
        rows.append(row)
        print(f"  {subj}: trials={n_trials} windows={len(X)}  "
              f"true w/t={row['true_window']:.3f}/{row['true_trial']:.3f} (null {null_true:.3f})  "
              f"rand w/t={row['rand_window']:.3f}/{row['rand_trial']:.3f} (null {null_rand:.3f})", flush=True)
    f.close()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "leakage_audit_per_subject.csv"), index=False)
    mw, mt = df.rand_window.mean(), df.rand_trial.mean()
    tw, tt = df.true_window.mean(), df.true_trial.mean()
    leak = mw > 0.60 and 0.40 <= mt <= 0.60
    lines = [
        "== ARM D -- window-CV leakage audit on MAD-EEG stereo duos ==",
        "!! EXPLORATORY BY CONSTRUCTION: the duos are spent (ledger of looks). This is a",
        "!! METHODS claim about a protocol, not an attention claim about the EEG.",
        "!! The pseudo-label arm is the audit's positive control: its labels are random per",
        "!! trial, so any window-CV accuracy above its null measures pure trial-fingerprint",
        "!! leakage. Interpretation rule (declared in code BEFORE the run): leakage is",
        "!! demonstrated if pseudo window-CV > 0.60 while pseudo trial-CV is in [0.40,0.60].",
        f"subjects={len(df)}  features=log-var, {len(BANDS)} bands x 20 ch  classifier=LDA  windows=1 s",
        f"TRUE labels  : window-CV {tw:.4f}  vs trial-CV {tt:.4f}   (mean majority-class null {df.null_true.mean():.4f})",
        f"PSEUDO labels: window-CV {mw:.4f}  vs trial-CV {mt:.4f}   (null {df.null_rand.mean():.4f} by construction)",
        f"[{'LEAKAGE DEMONSTRATED' if leak else 'RULE NOT MET'}] " +
        ("a plain LDA decodes a label with ZERO attention content at "
         f"{mw:.0%} from 1-s windows when windows of the same trial cross the train/test "
         "split, and falls to chance when they do not. Any window-CV accuracy on this "
         "dataset -- including a 92.6% -- is dominated by trial fingerprints."
         if leak else "the declared rule was not met; report the numbers as they are."),
    ]
    txt = "\n".join(lines)
    open(os.path.join(out, "leakage_audit_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[armD] wrote -> {out}")


if __name__ == "__main__":
    main()
