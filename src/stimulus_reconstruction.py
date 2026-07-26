# CHANGED(baseline): NEW FILE — not part of the Akama upstream and not part of the
#   byte-identical contrastive baseline. This is a separate, self-contained method
#   (a stimulus-reconstruction / backward-model AAD decoder) added by this thesis.
#   It does NOT import or modify main.py / the contrastive LightningModule, so the
#   baseline reproduction path is untouched. It DOES reuse the dataset class
#   (`Preprocessing_EEGMusic_dataset`) only to obtain the *identical* cross-validation
#   splits (same cv_mode / cv_held_out_id gate), so its leave-song-out folds line up
#   one-to-one with the contrastive experiments and `sweeps/report.py` can read its
#   output unchanged.
#
# WHY THIS METHOD (scientific rationale)
# --------------------------------------
# The original dataset has a fixed song->attended-element mapping (song == target,
# 1:1, identical across subjects, verified empirically). A *classifier* trained on it
# therefore learns song identity and collapses to chance under leave-song-out. A
# backward/stimulus-reconstruction model instead makes a *within-song, relative*
# decision at test time: it reconstructs an audio feature (the broadband envelope)
# from EEG and asks whether the reconstruction correlates more with the ATTENDED stem
# than with the UNATTENDED stems OF THE SAME SONG. Because the comparison is internal
# to one song, it cannot be won by recognising the song. It can still be confounded by
# acoustic salience (attended ~ foreground), which is an honest limitation of THIS
# dataset and is reported as such — a clean separation of attention from salience would
# require a within-stimulus attention manipulation (e.g. MAD-EEG).
#
# This is the field-standard AAD method (de Cheveigne et al. 2018; O'Sullivan et al.
# 2015; Cantisani et al. 2019 / MAD-EEG). The linear ridge decoder below is the
# reproducible anchor; a non-linear decoder can be plugged in via --recon_model.
#
# Decision/metric: 4-way (vocal/drum/bass/others), argmax over per-stem Pearson r.
# Chance = 0.25. Output rows are compatible with the contrastive test_records.csv so
# the existing aggregation + diagnostics apply.

import sys
import os

# Keep CPU-only safe (the ridge path needs no GPU); mirror main.py's guard.
try:
    import torch as _torch
    _ok = _torch.cuda.is_available()
except Exception:
    _ok = False
if not _ok:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from scipy.signal import hilbert, butter, filtfilt
from sklearn.preprocessing import RobustScaler, StandardScaler

from datasets import get_dataset
from utils import yaml_config_hook, paths

TASK_NAMES = ["vocal", "drum", "bass", "others"]
EEG_FS = 256
AUDIO_FS = 44100
EEG_LEN = 256 * 15          # 3840 — the windowing length used by the dataset pipeline
WIN = 768                   # test window (3 s) — matches test_window_size
STRIDE = 256                # matches test_stride
N_WINDOWS = (EEG_LEN - WIN) // STRIDE + 1   # 13 windows per (subject, song) trial


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def bandpass(x, fs, low, high):
    """Zero-phase Butterworth band-pass along the last axis. Disabled if high<=0.
    Standard AAD pre-step: restrict EEG and envelope to the cortical
    envelope-tracking band (~1-8 Hz, delta-theta) before the backward model."""
    if not high or high <= 0:
        return x
    nyq = 0.5 * fs
    lo = max(low, 0.01) / nyq
    hi = min(high, 0.99 * nyq) / nyq
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x, axis=-1)


def stem_envelope(wav_path, out_len, env_cutoff_hz, compression, band_low, band_high):
    """Broadband envelope of one stem, resampled to the EEG time grid (out_len @256 Hz),
    then band-passed to the same band as the EEG so the two are matched.

    mono(mean) -> |analytic signal| (Hilbert) -> optional power-law compression ->
    anti-alias low-pass @audio_fs -> resample to out_len -> band-pass @256 -> z-score.
    """
    wav, sr = torchaudio.load(wav_path)              # (channels, T)
    x = wav.mean(0).numpy().astype(np.float64)        # mono
    env = np.abs(hilbert(x))                           # broadband envelope @audio_fs
    if compression and compression != 1.0:
        env = np.power(np.maximum(env, 0.0), compression)
    # anti-alias / smooth before downsampling
    nyq = 0.5 * sr
    b, a = butter(4, min(env_cutoff_hz, 0.99 * nyq) / nyq, btype="low")
    env = filtfilt(b, a, env)
    # resample to the EEG grid (length out_len) by linear interpolation
    t_src = np.linspace(0.0, 1.0, num=len(env), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    env256 = np.interp(t_dst, t_src, env)
    # match the EEG band (e.g. 1-8 Hz) so reconstruction and target live in the same band
    env256 = bandpass(env256, EEG_FS, band_low, band_high)
    # z-score (decision is correlation-based; this only stabilises the ridge target)
    env256 = (env256 - env256.mean()) / (env256.std() + 1e-8)
    return env256.astype(np.float64)


def load_eeg(eeg_path, normalization, clamp_value, band_low, band_high):
    """Load one (subject, song) EEG trial, optionally band-pass to the envelope-tracking
    band, normalise as in the training pipeline (per-channel RobustScaler + clamp),
    truncate to EEG_LEN. Returns (4, EEG_LEN) float64."""
    with open(eeg_path, "rb") as f:
        eeg = pickle.load(f)                          # torch.Tensor (4, ~3864)
    x = eeg.float().numpy().astype(np.float64)        # (4, T)
    # band-pass BEFORE normalisation: removes Muse DC/drift that RobustScaler cannot
    x = bandpass(x, EEG_FS, band_low, band_high)
    if normalization == "MetaAI":
        # per-channel RobustScaler + clamp (equivalent to dataset.normalize_EEG_4)
        for idx in range(x.shape[0]):
            ch = x[idx].reshape(-1, 1)
            x[idx] = RobustScaler().fit(ch).transform(ch).reshape(-1)
        x = np.clip(x, -int(clamp_value), int(clamp_value))
    if x.shape[1] < EEG_LEN:
        x = np.pad(x, ((0, 0), (0, EEG_LEN - x.shape[1])), mode="edge")
    return x[:, :EEG_LEN]


def lagged_design(eeg, n_lags):
    """Backward-model design matrix. ŝ(t) = sum_{c,tau} g(c,tau) * eeg(c, t+tau),
    tau in [0, n_lags]. Edge-pad the EEG tail by n_lags so the reconstruction spans
    the full EEG_LEN. Returns X of shape (EEG_LEN, n_channels*(n_lags+1))."""
    c, T = eeg.shape
    padded = np.pad(eeg, ((0, 0), (0, n_lags)), mode="edge")
    cols = [padded[:, tau:tau + T] for tau in range(n_lags + 1)]   # each (c, T)
    X = np.concatenate(cols, axis=0).T                              # (T, c*(n_lags+1))
    return X


# --------------------------------------------------------------------------- #
# Ridge backward model (closed form)
# --------------------------------------------------------------------------- #
def ridge_fit(X, y, lam):
    """Closed-form ridge: w = (X^T X + lam I)^-1 X^T y. X assumed standardised."""
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# Per-subject decoder: train on training songs, reconstruct held-out song
# --------------------------------------------------------------------------- #
def build_trial(row, normalization, clamp_value, env_cutoff_hz, compression, band_low, band_high):
    """Return (eeg (4,EEG_LEN), envs (4,EEG_LEN), task, attention, song)."""
    eeg = load_eeg(row["eeg_path"], normalization, clamp_value, band_low, band_high)
    envs = np.stack([
        stem_envelope(row[f"audio_path{k}"], EEG_LEN, env_cutoff_hz, compression, band_low, band_high)
        for k in range(4)
    ], axis=0)
    return eeg, envs, int(row["task"]), int(row["attention_score"]), int(row["song"])


def _inner_split(n, rng):
    idx = np.arange(n); rng.shuffle(idx)
    n_val = max(1, n // 5)
    return idx[n_val:].tolist(), idx[:n_val].tolist()   # train_idx, val_idx


def fit_ridge(train_rows, n_lags, lam_grid, cfg, rng):
    """Subject-specific LINEAR backward model (ridge) with inner-CV lambda selection.
    Target = ATTENDED stem envelope. Returns (predict_fn, inner_val_r, info)."""
    trials = [build_trial(r, *cfg) for _, r in train_rows.iterrows()]
    Xs = [lagged_design(eeg, n_lags) for eeg, _, _, _, _ in trials]
    ys = [envs[task] for _, envs, task, _, _ in trials]
    tr_idx, val_idx = _inner_split(len(trials), rng)
    scaler = StandardScaler().fit(np.concatenate([Xs[i] for i in tr_idx], 0))
    Xtr_s = scaler.transform(np.concatenate([Xs[i] for i in tr_idx], 0))
    ytr = np.concatenate([ys[i] for i in tr_idx], 0)
    best_lam, best_r = lam_grid[0], -np.inf
    for lam in lam_grid:
        w = ridge_fit(Xtr_s, ytr, lam)
        r = float(np.mean([pearson(scaler.transform(Xs[i]) @ w, ys[i]) for i in val_idx]))
        if r > best_r:
            best_r, best_lam = r, lam
    # refit on ALL training trials with best lambda
    scaler = StandardScaler().fit(np.concatenate(Xs, 0))
    w = ridge_fit(scaler.transform(np.concatenate(Xs, 0)), np.concatenate(ys, 0), best_lam)

    def predict(eeg):
        return scaler.transform(lagged_design(eeg, n_lags)) @ w
    return predict, best_r, f"lam={best_lam:g}"


# --------------------------------------------------------------------------- #
# Non-linear backward model (temporal CNN) — "our model", GPU path
# --------------------------------------------------------------------------- #
import torch.nn as nn
import torch.nn.functional as F


class EnvCNN(nn.Module):
    """Small temporal CNN: EEG (B, 4, T) -> reconstructed envelope (B, T).
    Receptive field ~250 ms via stacked/dilated convs; BN + dropout for regularisation
    given the tiny per-subject sample (~39 trials)."""
    def __init__(self, n_ch=4, hidden=16, ksize=17, dropout=0.3):
        super().__init__()
        p = ksize // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, hidden, ksize, padding=p), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, ksize, padding=2 * p, dilation=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_cnn(train_rows, cfg, rng, device, hidden, dropout, lr, wd, epochs, patience):
    """Subject-specific NON-LINEAR backward model. Early-stopping on inner-val Pearson.
    Returns (predict_fn, inner_val_r, info)."""
    trials = [build_trial(r, *cfg) for _, r in train_rows.iterrows()]
    X = np.stack([t[0] for t in trials]).astype(np.float32)             # (n, 4, L)
    Y = np.stack([t[1][t[2]] for t in trials]).astype(np.float32)        # (n, L) attended env
    tr_idx, val_idx = _inner_split(len(trials), rng)
    Xt = torch.tensor(X[tr_idx], device=device); Yt = torch.tensor(Y[tr_idx], device=device)
    Xv = torch.tensor(X[val_idx], device=device); Yv = Y[val_idx]
    net = EnvCNN(hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_r, best_state, bad = -np.inf, None, 0
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = F.mse_loss(net(Xt), Yt); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv).cpu().numpy()
        r = float(np.mean([pearson(pv[i], Yv[i]) for i in range(len(val_idx))]))
        if r > best_r:
            best_r, bad = r, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(eeg):
        with torch.no_grad():
            xt = torch.tensor(eeg[None].astype(np.float32), device=device)
            return net(xt).cpu().numpy()[0]
    return predict, best_r, f"cnn(h={hidden})"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Stimulus-reconstruction (backward model) AAD")
    config = yaml_config_hook("../configs/baseline.yaml")
    # shared defaults (only the ones we use); ignore the rest
    parser.add_argument("--dataset", type=str, default=config.get("dataset", "preprocessing_eegmusic"))
    parser.add_argument("--dataset_dir", type=str, default=config.get("dataset_dir", "../dataset"))
    parser.add_argument("--log_dir", type=str, default="../results")
    parser.add_argument("--training_date", type=str, required=True)
    parser.add_argument("--cv_mode", type=str, default="leave_song_out")
    parser.add_argument("--cv_held_out_id", type=int, required=True)
    parser.add_argument("--eeg_normalization", type=str, default=config.get("eeg_normalization", "MetaAI"))
    parser.add_argument("--clamp_value", type=int, default=config.get("clamp_value", 20))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--attention_values", type=int, nargs="+", default=[4, 5])
    # recon-specific (this is where "our model" can roam)
    parser.add_argument("--recon_model", type=str, default="ridge", choices=["ridge", "cnn"],
                        help="backward-model family; 'ridge' = reproducible linear anchor, 'cnn' = non-linear (GPU)")
    parser.add_argument("--recon_hidden", type=int, default=16, help="CNN hidden channels")
    parser.add_argument("--recon_dropout", type=float, default=0.3, help="CNN dropout")
    parser.add_argument("--recon_lr", type=float, default=1e-3, help="CNN Adam lr")
    parser.add_argument("--recon_wd", type=float, default=1e-4, help="CNN weight decay")
    parser.add_argument("--recon_epochs", type=int, default=200, help="CNN max epochs")
    parser.add_argument("--recon_patience", type=int, default=20, help="CNN early-stopping patience (inner-val Pearson)")
    parser.add_argument("--recon_lags_ms", type=float, default=250.0)
    parser.add_argument("--recon_env_cutoff_hz", type=float, default=20.0)
    parser.add_argument("--recon_compression", type=float, default=0.3, help="power-law envelope compression; 1.0 = none")
    parser.add_argument("--recon_band_low", type=float, default=1.0, help="EEG+envelope band-pass low edge (Hz); standard AAD delta-theta band")
    parser.add_argument("--recon_band_high", type=float, default=8.0, help="EEG+envelope band-pass high edge (Hz); <=0 disables band-pass")
    args = parser.parse_args()

    # path resolution consistent with main.py
    if args.dataset_dir == config.get("dataset_dir", "../dataset"):
        args.dataset_dir = paths.resolve_dataset_dir(config.get("dataset_dir", "../dataset"))
    if args.log_dir == "../results":
        args.log_dir = paths.resolve_log_dir("../results")
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    n_lags = int(round(args.recon_lags_ms / 1000.0 * EEG_FS))
    lam_grid = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = (args.eeg_normalization, args.clamp_value, args.recon_env_cutoff_hz, args.recon_compression,
           args.recon_band_low, args.recon_band_high)
    high_attn_set = set(args.attention_values)

    print(f"[recon] model={args.recon_model} device={device} cv_mode={args.cv_mode} held_out={args.cv_held_out_id} "
          f"lags={n_lags} ({args.recon_lags_ms}ms) env_cutoff={args.recon_env_cutoff_hz}Hz "
          f"compression={args.recon_compression} band=[{args.recon_band_low},{args.recon_band_high}]Hz", flush=True)

    # identical CV split via the dataset class (un-windowed trial table)
    train_df = get_dataset(args.dataset, args.dataset_dir, subset="train", download=False,
                           cv_mode=args.cv_mode, cv_held_out_id=args.cv_held_out_id).df
    test_df = get_dataset(args.dataset, args.dataset_dir, subset="test", download=False,
                          cv_mode=args.cv_mode, cv_held_out_id=args.cv_held_out_id).df
    for df in (train_df, test_df):
        for col in ("subject", "song", "task", "attention_score"):
            df[col] = df[col].astype(int)
    print(f"[recon] train trials={len(train_df)} test trials={len(test_df)} "
          f"subjects={sorted(test_df.subject.unique())}")

    records = []          # per-window rows (compatible CSV)
    trial_rows = []       # per-trial AAD summary
    for subj in sorted(test_df.subject.unique()):
        tr = train_df[train_df.subject == subj]
        te = test_df[test_df.subject == subj]
        if len(tr) < 3 or len(te) == 0:
            print(f"[recon]  subj {subj}: insufficient data (train={len(tr)} test={len(te)}) -> skip")
            continue
        if args.recon_model == "cnn":
            predict, val_r, info = fit_cnn(tr, cfg, rng, device, args.recon_hidden,
                                           args.recon_dropout, args.recon_lr, args.recon_wd,
                                           args.recon_epochs, args.recon_patience)
        else:
            predict, val_r, info = fit_ridge(tr, n_lags, lam_grid, cfg, rng)
        for _, row in te.iterrows():
            eeg, envs, task, attn, song = build_trial(row, *cfg)
            shat = predict(eeg)
            # per-trial decision (full ~15 s)
            sims_trial = [pearson(shat, envs[k]) for k in range(4)]
            trial_rows.append(dict(subject=subj, song=song, task=task, attention=attn,
                                   pred=int(np.argmax(sims_trial)),
                                   correct=int(np.argmax(sims_trial) == task),
                                   r_attended=sims_trial[task]))
            # per-window decisions (compatible with contrastive test_records.csv)
            for win in range(N_WINDOWS):
                sl = slice(win * STRIDE, win * STRIDE + WIN)
                sims = [pearson(shat[sl], envs[k][sl]) for k in range(4)]
                pos = sims[task]; neg = max(sims[k] for k in range(4) if k != task)
                records.append(dict(
                    subject=subj, song=song, task=task, attention=attn,
                    sim_vocal=sims[0], sim_drum=sims[1], sim_bass=sims[2], sim_others=sims[3],
                    pos_sim=pos, max_neg_sim=neg, margin=pos - neg,
                    correct=int(np.argmax(sims) == task),
                    high_attention=bool(attn in high_attn_set),
                ))
        print(f"[recon]  subj {subj}: {info} inner_val_r={val_r:.3f} "
              f"trial_acc={np.mean([t['correct'] for t in trial_rows if t['subject']==subj]):.2f}", flush=True)

    rec = pd.DataFrame(records)
    trials = pd.DataFrame(trial_rows)

    # write outputs in the contrastive layout so sweeps/report.py + diagnostics work
    out_dir = Path(args.log_dir) / args.training_date / f"nmed-CL-{args.dataset}" / "version_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out_dir / "test_records.csv", index=False)

    chance = 0.25
    per_class_win = rec.groupby("task").correct.mean().reindex(range(4))
    per_subj_trial = trials.groupby("subject").correct.mean()
    summary = []
    summary.append("== Stimulus-reconstruction (backward model) breakdown ==")
    summary.append(f"cv_mode={args.cv_mode} held_out_song={args.cv_held_out_id} model={args.recon_model} "
                   f"lags={n_lags} env_cutoff={args.recon_env_cutoff_hz}Hz compression={args.recon_compression} "
                   f"band=[{args.recon_band_low},{args.recon_band_high}]Hz")
    summary.append(f"n_windows total: {len(rec)}  (decision: 4-way argmax, chance={chance})")
    summary.append(f"global accuracy (windows, all):  {rec.correct.mean():.4f}")
    if (rec.high_attention).any():
        summary.append(f"global accuracy (windows, attn): {rec[rec.high_attention].correct.mean():.4f}")
    summary.append(f"per-class window acc [v d b o]: "
                   + " ".join(f"{TASK_NAMES[c]}={per_class_win[c]:.3f}" if not np.isnan(per_class_win[c]) else f"{TASK_NAMES[c]}=NA" for c in range(4)))
    summary.append(f"per-TRIAL accuracy (full 15s):   {trials.correct.mean():.4f}  (n_trials={len(trials)})")
    summary.append("per-subject TRIAL acc: " + " ".join(f"s{int(s)}={v:.2f}" for s, v in per_subj_trial.items()))
    summary.append(f"mean r(reconstruction, ATTENDED stem): {trials.r_attended.mean():.4f}")
    txt = "\n".join(summary)
    (out_dir / "test_breakdown_summary.txt").write_text(txt + "\n")
    trials.to_csv(out_dir / "recon_trial_records.csv", index=False)
    print("\n" + txt)
    print(f"\n[recon] wrote -> {out_dir}")


if __name__ == "__main__":
    main()
