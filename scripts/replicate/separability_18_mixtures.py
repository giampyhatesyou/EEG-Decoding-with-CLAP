"""The Exp. 9 / Exp. 13 stem correlations counted on the 18 duo mixtures instead of 36 stim ids.

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed. AUDIO ONLY:
like Exp. 13 it opens the preprocessed HDF5 for `soli` and metadata, never ['response'].

WHY: the 36 units of Exp. 9/13 are 18 duo mixtures x 2 attended targets with the SAME audio
(docs/provenance/2026-08-12_exp13_stem_separability.txt, lines 9-12), so every count over 36
is twice a count over 18 and the binomial on 36 treats duplicated units as independent.

POSITIVE CONTROL, crossed before any number over 18 is printed (exit 1 otherwise):
  mel-8 mean 0.1772, flux mean 0.2874, flux more similar on 30/36, mean diff +0.1103 (Exp. 9);
  the |corr|-below-mel-8 count of every Exp. 13 / Exp. 16A row as published (C1..C6 18 18 32
  16 16 28, noise-512 36/36); every count over 36 even, and the two stim ids of each mixture
  giving IDENTICAL correlations under every representation.
THEN, over the 18 mixtures: per-mixture mel-8 and flux correlation and their difference; exact
  sign test, Wilcoxon signed-rank, bootstrap 95% interval of the mean difference (10 000
  replicates, seed 42), each with its null; the gate counts over 18 and the exact p of the
  threshold equivalent to 26/36 (13/18) against 0.05/6, printed without a verdict.

Usage: /opt/miniconda3/bin/python scripts/replicate/separability_18_mixtures.py \
         --madeeg_dir ~/madeeg [--fig <pdf>]
"""
import argparse
import os
import sys
from math import comb

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from madeeg_stem_separability import (CANDIDATES, CLAP_DIM, REFERENCES, load_stems,  # noqa: E402
                                      stats)

OUT = "2026-09-27_separability_18_mixtures.txt"
# published counts over 36 (|corr| strictly below mel-8), Exp. 13 table and Exp. 16A row N1
PUBLISHED_K36 = {"mel-8": 0, "flux": 8, "C1 mel-24": 18, "C2 mel-64": 18, "C3 MFCC-13": 32,
                 "C4 flux-8band": 16, "C5 flux-24band": 16, "C6 contrast-6": 28,
                 f"N1 noise-{CLAP_DIM}": 36}
NOISE = (f"N1 noise-{CLAP_DIM}", ("noise", CLAP_DIM))
BONF = 0.05 / 6
N_BOOT, SEED = 10_000, 42


def tail(k, n):
    """P(X >= k | n, 0.5), exact."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--fig", default="")
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    out_path = os.path.join(root, "docs", "provenance", OUT)
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed",
         "!! (Exp. 9 audio diagnostic, Exp. 13, Exp. 16A). AUDIO ONLY: no EEG was read.",
         "!! It recounts published numbers on the 18 distinct mixtures; it spends no look.",
         ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    def stop(msg):
        say(f"POSITIVE CONTROL FAILED: {msg}. Nothing below this line exists.")
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)

    units, stems = load_stems(os.path.expanduser(args.madeeg_dir))
    mix_of = [frozenset(keys) for _, keys in units]
    mixtures = sorted(set(mix_of), key=lambda m: min(k for (k, ks) in units if frozenset(ks) == m))
    name_of = {m: min(k for (k, ks) in units if frozenset(ks) == m).rsplit("_", 1)[0]
               for m in mixtures}
    say(f"units: {len(units)} stim ids · {len(mixtures)} distinct mixtures · {len(stems)} distinct stems")

    w = {}
    for name, spec in REFERENCES + CANDIDATES + [NOISE]:
        w[name] = stats(units, stems, spec)[0]
    mel, flux = w["mel-8"], w["flux"]

    say()
    say("=== POSITIVE CONTROL (published numbers over 36, crossed first) ===")
    checks = [("mel-8 mean", round(mel.mean(), 4), 0.1772), ("flux mean", round(flux.mean(), 4), 0.2874),
              ("flux > mel-8", int((flux > mel).sum()), 30),
              ("mean diff", round(float(np.mean(flux - mel)), 4), 0.1103)]
    ref = np.abs(mel)
    k36 = {name: int((np.abs(v) < ref).sum()) for name, v in w.items()}
    checks += [(f"{n} below mel-8", k36[n], PUBLISHED_K36[n]) for n in PUBLISHED_K36]
    for label, got, exp in checks:
        say(f"  {label:26s} {got!s:>8s}  published {exp!s:>8s}  [{'OK' if got == exp else 'FAIL'}]")
        if got != exp:
            stop(label)
    for name, v in w.items():
        for m in mixtures:
            vals = v[[i for i, mm in enumerate(mix_of) if mm == m]]
            if len(vals) != 2 or vals[0] != vals[1]:
                stop(f"{name}: the two stim ids of {name_of[m]} differ ({vals})")
    if any(k % 2 for k in k36.values()):
        stop("a count over 36 is odd")
    say("  every count over 36 is even, and the two stim ids of each of the 18 mixtures give")
    say("  bit-identical correlations under all 9 representations -> [OK]")

    first = [min(i for i, mm in enumerate(mix_of) if mm == m) for m in mixtures]
    m18, f18 = mel[first], flux[first]
    d = f18 - m18

    say()
    say("=== FLUX vs MEL-8, one row per mixture (signed r, mean over bands) ===")
    say(f"  {'mixture':42s} {'mel-8':>8s} {'flux':>8s} {'flux-mel':>9s}")
    for m, a, b in zip(mixtures, m18, f18):
        say(f"  {name_of[m]:42s} {a:8.4f} {b:8.4f} {b - a:+9.4f}")
    k = int((d > 0).sum())
    say(f"  mean mel-8 {m18.mean():.4f} · mean flux {f18.mean():.4f} · mean diff {d.mean():+.4f} "
        f"· median diff {np.median(d):+.4f}")
    say()
    say("=== TESTS on the 18 differences (unit = mixture) ===")
    say(f"  sign test: flux more similar in {k}/18   null 9/18 = 0.500   one-sided p = {tail(k, 18):.4f}"
        f"   two-sided p = {min(1.0, 2 * tail(max(k, 18 - k), 18)):.4f}")
    wt = wilcoxon(f18, m18, alternative="two-sided", method="exact")
    wg = wilcoxon(f18, m18, alternative="greater", method="exact")
    say(f"  Wilcoxon signed-rank (exact): W = {wt.statistic:.0f}   null: median difference 0   "
        f"two-sided p = {wt.pvalue:.6f}   one-sided p = {wg.pvalue:.6f}")
    rng = np.random.RandomState(SEED)
    boot = d[rng.randint(0, 18, size=(N_BOOT, 18))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    say(f"  bootstrap of the mean difference over mixtures ({N_BOOT} replicates, seed {SEED}): "
        f"{d.mean():+.4f}, 95% percentile interval [{lo:+.4f}, {hi:+.4f}]   null 0")
    say("  LIMIT: the 18 mixtures are not fully independent either: they reuse "
        f"{len(stems)} distinct stems over 36 stem slots.")

    say()
    say("=== GATE COUNTS over 18 (|corr| strictly below mel-8; the Exp. 13/16A criterion 1) ===")
    say(f"  {'representation':18s} {'k/36':>6s} {'P(X>=k|36)':>11s} {'k/18':>6s} {'P(X>=k|18)':>11s}")
    for name in PUBLISHED_K36:
        if name == "mel-8":
            continue
        kk = k36[name]
        say(f"  {name:18s} {kk:>3d}/36 {tail(kk, 36):11.6f} {kk // 2:>3d}/18 {tail(kk // 2, 18):11.6f}")
    say("  C7 CLAP-512 (Exp. 17 file, 32/36) is not recomputed: its series exist only on the GPU")
    say(f"  node. Over 18 it would be 16/18, P = {tail(16, 18):.6f}, IF its two halves are identical")
    say("  like every row above (the noise and every fixed representation are functions of the")
    say("  stems alone, so they are; that CLAP's are is inferred, not measured).")
    say()
    say(f"  threshold of the gate: 26/36, P(X>=26|36) = {tail(26, 36):.6f}, Bonferroni 0.05/6 = {BONF:.6f}")
    say(f"  the same proportion over 18 is 13/18: P(X>=13|18) = {tail(13, 18):.6f}")
    kmin = next(kk for kk in range(19) if tail(kk, 18) <= BONF)
    say(f"  the smallest count over 18 with P <= 0.05/6 is {kmin}/18 (P = {tail(kmin, 18):.6f})")
    say("  (printed as information for the text; no verdict is re-taken here)")

    open(out_path, "w").write("\n".join(L) + "\n")
    print(f"\n-> {out_path}")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        for a, b in zip(m18, f18):
            ax.plot([0, 1], [a, b], color="0.35" if b > a else "tab:red", lw=0.9, marker="o", ms=3)
        ax.plot([0, 1], [m18.mean(), f18.mean()], color="k", lw=2.2, label="mean")
        ax.set_xticks([0, 1], ["log-mel (8 bands)", "spectral flux"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel("correlation between the two stems")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"18 duo mixtures: flux higher in {k}/18", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.fig)
        print(f"-> {args.fig}")


if __name__ == "__main__":
    main()
