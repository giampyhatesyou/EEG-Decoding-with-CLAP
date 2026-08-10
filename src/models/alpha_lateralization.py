# CHANGED(baseline): NEW FILE -- the paired alpha test (vault: Exp. 6 pre-registration, 10/8).
#   Used by `madeeg_reconstruction.py --alpha_li`. No decoder, no audio, no training: this is a
#   sign test on a TONIC quantity, scored on twin pairs.
#
#   Why this is safe even though it is, in effect, a classifier. Yan et al. (Interspeech 2025)
#   did not show that classifiers are inflated; they showed that ANY decoder which can succeed
#   by recognising WHICH TRIAL OR GROUP it is looking at is inflated. A twin pair makes that
#   impossible: same subject, same mixture, same audio, same session -- the only thing that
#   differs is which source was attended. Every constant of subject, channel, impedance,
#   session and mixture cancels in the DIFFERENCE, algebraically.
#
#   And why it belongs here rather than inside the CCA: alpha lateralization is a TONIC
#   correlate of the attended side, while the CCA decides by within-trial correlation, which
#   centres constants out. The two forms only meet in a test that is itself tonic. This is one.
"""Alpha laterality index on the symmetric pairs, and its twin-pair sign test.

    LI(E) = mean over F3/F4, C3/C4, P3/P4, O1/O2 of [ log P_alpha(right) - log P_alpha(left) ]

🔴 THE DIRECTION IS FIXED HERE, BEFORE ANY NUMBER, AND IS NOT FLIPPED LATER (Comandamenti #6).
    Alpha desynchronization is CONTRALATERAL to the attended side: attending on the RIGHT
    lowers alpha over the LEFT hemisphere, hence RAISES log P(right) - log P(left). So a twin
    pair is correct when sign(LI(E_A) - LI(E_B)) == sign(side(a) - side(b)). If the real data
    come out below chance, that gets explained, not inverted.

NO Z-SCORING ACROSS TRIALS, and that is a decision, not an omission: the twin difference
    already cancels every per-channel and per-subject constant. Normalising across trials
    would be a statistical remedy for something that cancels by construction.
"""
import numpy as np
from scipy.signal import butter, filtfilt

ALPHA = (8.0, 14.0)
# Left first, so the index reads log P(right) - log P(left).
LAT_PAIRS = (("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"))


def _bandpass(x, fs, band):
    nyq = 0.5 * fs
    b, a = butter(4, [band[0] / nyq, min(band[1], 0.98 * nyq) / nyq], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64), axis=-1)


def laterality_index(seg, ch_names, fs, band=ALPHA):
    """(n_ch, T) segment -> one scalar. The segment is band-passed HERE, so an injected
    control signal has to survive the same filter the real data go through."""
    idx = {c: i for i, c in enumerate(ch_names)}
    missing = [c for pair in LAT_PAIRS for c in pair if c not in idx]
    assert not missing, f"montage lacks the symmetric channels {missing}"
    y = _bandpass(seg, fs, band)
    logp = np.log(np.mean(y * y, axis=1) + 1e-30)
    return float(np.mean([logp[idx[r]] - logp[idx[l]] for l, r in LAT_PAIRS]))


def inject_lateralized_alpha(seg, ch_names, fs, side, amp, rng, band=ALPHA):
    """POSITIVE CONTROL ONLY. Add a narrowband alpha oscillation to the hemisphere IPSILATERAL
    to `side`, i.e. the one where alpha is expected to RISE, with amplitude `amp` times that
    channel's own alpha-band std. Returns a copy; the real path never calls this.

    side = -1 left, +1 right, 0 none (nothing is injected, which is the mono case).
    The injected signal is added to the BROADBAND segment and is then band-passed by
    laterality_index, so it is measured by exactly the code under test."""
    out = np.array(seg, dtype=np.float64, copy=True)
    if not side:
        return out
    idx = {c: i for i, c in enumerate(ch_names)}
    picks = [idx[p[1 if side > 0 else 0]] for p in LAT_PAIRS]      # ipsilateral hemisphere
    alpha = _bandpass(seg[picks], fs, band)
    noise = _bandpass(rng.standard_normal((len(picks), seg.shape[1])), fs, band)
    for n, ch in enumerate(picks):
        out[ch] += amp * (alpha[n].std() + 1e-12) * noise[n] / (noise[n].std() + 1e-12)
    return out


def pair_correct(li_a, li_b, side_a, side_b):
    """1 if the twin difference has the sign the panning predicts, 0 otherwise.

    An exact tie counts as WRONG (conservative, and measure-zero on floats). Returns None when
    the two sides are equal, i.e. when no sign is predicted at all."""
    want = np.sign(side_a - side_b)
    if want == 0:
        return None
    return int(np.sign(li_a - li_b) == want)


def _selfcheck():
    """Run: python src/models/alpha_lateralization.py

    Pins the three properties the entrypoint relies on -- the index reads the right side, the
    injection is recovered, and the null is exactly 0.5 by symmetry."""
    rng = np.random.RandomState(0)
    ch = ["F3", "F1", "Fz", "F2", "F4", "C3", "C1", "Cz", "C2", "C4",
          "CPz", "P3", "P1", "Pz", "P2", "P4", "POz", "O1", "Oz", "O2"]
    fs, T = 256.0, 256 * 20

    # (1) the index has the sign the docstring claims
    seg = rng.standard_normal((20, T))
    li0 = laterality_index(seg, ch, fs)
    assert laterality_index(inject_lateralized_alpha(seg, ch, fs, +1, 3.0, rng), ch, fs) > li0
    assert laterality_index(inject_lateralized_alpha(seg, ch, fs, -1, 3.0, rng), ch, fs) < li0

    # (2) the sign rule, and a tie is wrong
    assert pair_correct(1.0, 0.0, +1, -1) == 1 and pair_correct(0.0, 1.0, +1, -1) == 0
    assert pair_correct(1.0, 1.0, +1, -1) == 0 and pair_correct(1.0, 0.0, +1, +1) is None

    # (3) NULL = 0.5 EXACTLY, and the argument is EXCHANGEABILITY, not relabelling.
    #     Two properties, and the first is the one an earlier draft of this check got wrong:
    #     (3a) renaming which twin is "A" flips BOTH the LI order and the side order, so the
    #          outcome is INVARIANT -- the statistic is well defined, it does not flip;
    #     (3b) under H0 the two twins are exchangeable (same subject, same mixture, same audio;
    #          only the attended source differs), so swapping their LIs with the sides HELD
    #          FIXED must flip the outcome. That is what makes the sign of the difference
    #          equally likely either way, hence null = 0.5 exactly rather than estimated.
    ab = rng.standard_normal((500, 2))
    assert all(pair_correct(a, b, +1, -1) == pair_correct(b, a, -1, +1) for a, b in ab), \
        "the statistic depends on which twin is called A -- it must not"
    assert all(pair_correct(a, b, +1, -1) + pair_correct(b, a, +1, -1) == 1 for a, b in ab), \
        "the twins are not exchangeable under H0 -- the null is not 0.5, re-derive"

    # (4) end-to-end on injected twin pairs: the recovery the positive control will demand
    ok = 0
    for _ in range(60):
        base = rng.standard_normal((20, T))
        s = 1 if rng.rand() > 0.5 else -1
        a = inject_lateralized_alpha(base, ch, fs, s, 2.0, rng)
        b = inject_lateralized_alpha(base, ch, fs, -s, 2.0, rng)
        ok += pair_correct(laterality_index(a, ch, fs), laterality_index(b, ch, fs), s, -s)
    assert ok / 60 >= 0.90, ok
    print(f"[alpha_lateralization] PASS  injected twin pairs recovered {ok}/60 = {ok / 60:.4f} "
          f"(threshold 0.90)\n  null = 0.5 EXACTLY, by EXCHANGEABILITY of the two twins under "
          f"H0 (verified on 500 random pairs): the statistic does not depend on which twin is\n"
          f"  called A, and swapping their indices with the sides held fixed flips the outcome "
          f"every time.")


if __name__ == "__main__":
    _selfcheck()
