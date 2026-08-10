# CHANGED(baseline): NEW FILE -- model 9 of the genealogy. Multi-view features + regularized
#   CCA for the MAD-EEG backward-model arm, used by `madeeg_reconstruction.py --estimator cca`.
#
#   Why CCA and not another ridge: de Cheveigne et al. (NeuroImage 2018) report that canonical
#   component analysis "finds significantly higher correlation scores, thus providing increased
#   sensitivity to relatively small effects". Our effect IS small (r_att 0.0202 vs r_unatt
#   0.0118), so this is the single best-supported lever, and it stays in the target-is-a-SIGNAL
#   family that Yan et al. (Interspeech 2025) showed survives a non-block design -- the decision
#   is still "which candidate source correlates more", never a class label.
#
#   This module owns FEATURES and ALGEBRA only. It never opens a file, never knows about
#   subjects, folds or trials: the entrypoint owns the data path and the decision rule.
"""Multi-view blocks (EEG side and stimulus side) and a regularized CCA over them.

    X (EEG)      eeg_lagged     the ridge's own lagged design matrix, unchanged
                 band_power     per-channel log power envelope, one block per declared band
                 lateralization log P(right) - log P(left) on the symmetric pairs, alpha band
    Y (stimulus) source_repr    the log-mel representation of the candidate source, unchanged
                 position       the source's side from wav_info.panning (-1 / 0 / +1)
                 position_env   position x the source's own envelope (see POSITION note below)

    rho(s) = sum of the first k canonical correlations of CCA(X, Y_s); the caller takes
    argmax over the present sources.

TWO STATISTICS, AND THEY ARE NOT COMPARABLE
    A canonical correlation is >= a plain Pearson correlation BY CONSTRUCTION: CCA optimizes
    both sides to maximize it. Comparing rho against the ridge's inner_val_r (0.0582) would
    show an improvement that does not exist -- the same trap that made AXIS 2 look like a
    doubling on 2026-07-29 until the like-for-like test gave 36/70, p=0.29.
    So `CCAModel.reconstruct` back-projects into the stimulus space and the caller scores it
    with the SAME band_pearson the ridge uses. That number, and only that number, is
    comparable. Both are reported, each with its label.

POSITION IS INERT IN A WITHIN-TRIAL CORRELATION -- measured here, not assumed
    `position` is constant over time within a trial, so it adds a constant to the canonical
    stimulus variate, and a Pearson correlation removes constants. rho(s) is therefore
    IDENTICAL with and without the position block. `_selfcheck` pins this as an assertion.
    `position_env` (position x the source's own envelope) is the time-varying form that keeps
    every declared property -- in mono it is identically 0 for every source, so the spatial
    block still cannot contribute BY CONSTRUCTION -- and it is a separate view name, chosen
    explicitly, never a silent redefinition of `position`.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

EEG_VIEWS = ("eeg_lagged", "band_power", "lateralization")
STIM_VIEWS = ("source_repr", "position", "position_env")

# de Vries, Marinato & Baldauf (2021) band edges, as adopted by the Exp. 5 pre-registration.
BANDS = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 14.0), ("beta", 14.0, 30.0))
# The symmetric pairs the MAD-EEG montage actually has (F3 F1 Fz F2 F4 · C3 C1 Cz C2 C4 · CPz ·
# P3 P1 Pz P2 P4 · POz · O1 Oz O2). Left first, so the index is log P(right) - log P(left).
LAT_PAIRS = (("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"))
LAT_BAND = "alpha"          # the lateralization index is the ALPHA one or it is not that index


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def parse_views(spec, allowed, flag):
    """Comma-separated list -> tuple in canonical order. A misspelled name BREAKS.

    Vault Comandamenti #8: a flag that quietly falls back produces a run that looks like the
    new arm and is the old one. `--cca_views eeg_laged` must not silently mean "no views"."""
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    assert names, f"{flag} is empty"
    bad = [n for n in names if n not in allowed]
    assert not bad, f"{flag}: unknown view(s) {bad} -- allowed: {list(allowed)}"
    assert len(set(names)) == len(names), f"{flag}: duplicate view in {names}"
    return tuple(n for n in allowed if n in names)


def usable_bands(band_low, band_high, fs):
    """The declared bands that fit inside the analysis band AND below 0.45*fs.

    The EEG reaching this module has already been band-passed to (band_low, band_high) and
    resampled to fs by the entrypoint, so a band outside that window is not weak, it is ZERO
    by construction. It is dropped and the drop is declared rather than being carried as a
    noise column. Widening the band is a SEPARATE, declared change (--band_high): it is never
    a side effect of asking for a band-power view."""
    hi = min(band_high, 0.45 * fs)
    return [(n, lo, h) for n, lo, h in BANDS if lo >= band_low - 1e-9 and h <= hi + 1e-9]


def _z(a, axis=-1):
    return (a - a.mean(axis=axis, keepdims=True)) / (a.std(axis=axis, keepdims=True) + 1e-8)


def _log_power(x, fs, lo, hi):
    """(n_ch, T) -> log power envelope in [lo, hi] Hz, same T. Hilbert magnitude squared."""
    nyq = 0.5 * fs
    b, a = butter(4, [max(lo, 0.01) / nyq, min(hi, 0.98 * nyq) / nyq], btype="band")
    y = filtfilt(b, a, np.asarray(x, dtype=np.float64), axis=-1)
    return np.log(np.abs(hilbert(y, axis=-1)) ** 2 + 1e-12)


def eeg_features(eeg, ch_names, views, n_lags, band, fs, lagged_design):
    """EEG (n_ch, T) -> (T, d) and the block sizes.

    `lagged_design` is passed in rather than reimplemented: the `eeg_lagged` view has to be
    the SAME matrix the ridge fits, or the regression canary means nothing."""
    blocks, sizes = [], []
    for v in views:
        if v == "eeg_lagged":
            B = lagged_design(eeg, n_lags)
        elif v == "band_power":
            usable = usable_bands(band[0], band[1], fs)
            assert usable, (f"--cca_views band_power: none of {[b[0] for b in BANDS]} fits inside "
                            f"the analysis band {band} at target_fs={fs} -- the block would be "
                            f"empty. Widen --band_high (a separate, declared change)")
            B = _z(np.concatenate([_log_power(eeg, fs, lo, hi) for _, lo, hi in usable], axis=0)).T
        elif v == "lateralization":
            assert LAT_BAND in [n for n, _, _ in usable_bands(band[0], band[1], fs)], (
                f"--cca_views lateralization: the {LAT_BAND} band is outside the analysis band "
                f"{band} at target_fs={fs}. An alpha lateralization index computed without alpha "
                f"is a different quantity wearing the same name -- widen --band_high first")
            lo, hi = next((l, h) for n, l, h in BANDS if n == LAT_BAND)
            P = _log_power(eeg, fs, lo, hi)
            idx = {c: i for i, c in enumerate(ch_names)}
            missing = [c for pair in LAT_PAIRS for c in pair if c not in idx]
            assert not missing, f"--cca_views lateralization: montage lacks {missing}"
            B = _z(np.stack([P[idx[r]] - P[idx[l]] for l, r in LAT_PAIRS])).T
        else:                                                    # unreachable: parse_views asserts
            raise AssertionError(f"unknown EEG view {v}")
        blocks.append(B)
        sizes.append((v, B.shape[1]))
    return np.concatenate(blocks, axis=1), sizes


def stim_features(rep, pos, views):
    """One candidate source: rep (n_bands, T) and its position in {-1, 0, +1} -> (T, d).

    Returns (Y, sizes, repr_cols) where `repr_cols` is the slice of Y holding source_repr --
    the only block that the back-projection is scored against, because it is the only one the
    ridge also reconstructs."""
    T = rep.shape[1]
    blocks, sizes, repr_cols = [], [], None
    col = 0
    for v in views:
        if v == "source_repr":
            B = np.asarray(rep, dtype=np.float64).T
            repr_cols = slice(col, col + B.shape[1])
        elif v == "position":
            B = np.full((T, 1), float(pos))
        elif v == "position_env":
            B = (float(pos) * _z(np.asarray(rep, dtype=np.float64).mean(axis=0))).reshape(-1, 1)
        else:
            raise AssertionError(f"unknown stimulus view {v}")
        blocks.append(B)
        sizes.append((v, B.shape[1]))
        col += B.shape[1]
    return np.concatenate(blocks, axis=1), sizes, repr_cols


# --------------------------------------------------------------------------- #
# regularized CCA
# --------------------------------------------------------------------------- #
def _shrink(C, reg):
    """C' = (1-reg) C + reg * nu * I, nu = trace/d = the average eigenvalue.

    Same normalized form the entrypoint's AXIS-3 estimator uses, so `reg` is a smoothing
    parameter in [0, 1] and does not have to be rescaled when the block count changes.
    Regularization is not optional here: 20 channels x 17 lags x views against ~150 trials
    overfits an unregularized CCA completely."""
    d = C.shape[0]
    return (1.0 - reg) * C + reg * (np.trace(C) / d) * np.eye(d)


def _inv_sqrt(C, floor=1e-10):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, floor * max(w.max(), 1e-12))
    return (V * w ** -0.5) @ V.T


class CCAModel:
    """Fitted canonical filters + the back-projection into stimulus space."""

    def __init__(self, mx, sx, my, Wx, Wy, B, rho_train):
        self.mx, self.sx, self.my = mx, sx, my
        self.Wx, self.Wy, self.B = Wx, Wy, B
        self.rho_train = rho_train                    # canonical correlations ON TRAINING data

    def _x(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mx) / self.sx

    def rho(self, X, Y):
        """Sum of the per-component correlations between the projected pair, on THESE data.

        Signed on purpose: a component whose training pattern does not hold on this trial
        should subtract, not be rescued by an absolute value."""
        U = self._x(X) @ self.Wx
        V = (np.asarray(Y, dtype=np.float64) - self.my) @ self.Wy
        tot = 0.0
        for i in range(U.shape[1]):
            a, b = U[:, i] - U[:, i].mean(), V[:, i] - V[:, i].mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            tot += float((a * b).sum() / den) if den > 1e-12 else 0.0
        return tot

    def reconstruct(self, X):
        """Back-projection: EEG -> canonical components -> stimulus space, (T, dy).

        This is what makes a like-for-like comparison with the ridge possible at all; the
        caller slices out the source_repr columns and scores them with band_pearson."""
        return (self._x(X) @ self.Wx) @ self.B + self.my


def fit(X, Y, k, reg):
    """Regularized CCA on pooled (T, dx) / (T, dy). Returns a CCAModel."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape[0] == Y.shape[0] and X.shape[0] > X.shape[1], (
        f"CCA needs more samples than EEG features: got {X.shape} vs {Y.shape}")
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0) + 1e-8                     # per-column scaling, exactly what the ridge does
    Xc, Yc = (X - mx) / sx, Y - my
    n = X.shape[0]
    Kx = _inv_sqrt(_shrink(Xc.T @ Xc / n, reg))
    Ky = _inv_sqrt(_shrink(Yc.T @ Yc / n, reg))
    U, S, Vt = np.linalg.svd(Kx @ (Xc.T @ Yc / n) @ Ky, full_matrices=False)
    k = int(min(k, len(S)))
    Wx, Wy = Kx @ U[:, :k], Ky @ Vt[:k].T
    # Back-projection fitted, not pseudo-inverted: least squares from the canonical EEG
    # variates to the stimulus block is the reconstruction the comparable number scores.
    B = np.linalg.lstsq(Xc @ Wx, Yc, rcond=None)[0]
    return CCAModel(mx, sx, my, Wx, Wy, B, np.clip(S[:k], 0.0, 1.0))


# --------------------------------------------------------------------------- #
def _selfcheck():
    """Run: python src/models/cca_multiview.py

    Four assertions, each pinning a property the entrypoint relies on."""
    rng = np.random.RandomState(0)
    fs, T, n_ch = 64.0, 4096, 20
    ch = ["F3", "F1", "Fz", "F2", "F4", "C3", "C1", "Cz", "C2", "C4",
          "CPz", "P3", "P1", "Pz", "P2", "P4", "POz", "O1", "Oz", "O2"]

    # (1) a known linear relation is recovered: rho ~ 1 on the planted component
    A = rng.standard_normal((6, 3))
    src = rng.standard_normal((T, 6))
    Xd = np.concatenate([src, rng.standard_normal((T, 4))], axis=1)
    Yd = np.concatenate([src @ A + 0.05 * rng.standard_normal((T, 3)),
                         rng.standard_normal((T, 2))], axis=1)
    m = fit(Xd, Yd, k=3, reg=1e-3)
    assert m.rho_train[0] > 0.98, m.rho_train
    assert m.rho(Xd, Yd) > 2.5, m.rho(Xd, Yd)

    # (2) the back-projection reconstructs the related columns
    Yhat = m.reconstruct(Xd)
    r = np.corrcoef(Yhat[:, 0], Yd[:, 0])[0, 1]
    assert r > 0.9, r

    # (3) THE ONE THAT MATTERS: a constant `position` column cannot move rho, because a
    #     within-trial Pearson removes constants. Reported, not worked around.
    rep = rng.standard_normal((8, T))
    eeg = rng.standard_normal((n_ch, T))

    def lag0(e, n):                                   # stand-in for the entrypoint's lagger
        return e.T
    Xf, _ = eeg_features(eeg, ch, ("eeg_lagged",), 0, (1.0, 8.0), fs, lag0)
    Yp, _, _ = stim_features(rep, +1.0, ("source_repr", "position"))
    Y0, _, _ = stim_features(rep, 0.0, ("source_repr", "position"))
    mp = fit(Xf, Yp, k=3, reg=1e-3)
    assert abs(mp.rho(Xf, Yp) - mp.rho(Xf, Y0)) < 1e-9, "position is NOT inert -- re-derive"

    # (4) the flags assert instead of falling back silently
    for bad in ("eeg_laged", "", "eeg_lagged,eeg_lagged"):
        try:
            parse_views(bad, EEG_VIEWS, "--cca_views")
        except AssertionError:
            pass
        else:
            raise AssertionError(f"parse_views accepted {bad!r}")
    assert parse_views("lateralization,eeg_lagged", EEG_VIEWS, "-") == ("eeg_lagged", "lateralization")
    assert [b[0] for b in usable_bands(1.0, 8.0, 64.0)] == ["delta", "theta"]
    assert [b[0] for b in usable_bands(1.0, 40.0, 64.0)] == ["delta", "theta", "alpha"]

    print("[cca_multiview] PASS  rho_train=" + " ".join(f"{v:.4f}" for v in m.rho_train)
          + f"  back-projected r={r:.4f}"
          + "\n  position block verified INERT for the decision: rho is identical with the "
            "column at +1 and at 0,\n  because a within-trial Pearson centres a per-trial "
            "constant out. Use position_env for a\n  time-varying spatial regressor (0 for "
            "every source in mono, so the control still holds).")


if __name__ == "__main__":
    _selfcheck()
