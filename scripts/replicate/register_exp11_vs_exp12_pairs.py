"""Exp. 11 against Exp. 12 stereo-from-raw, pair by pair (Ch. 9, the register under two cleanings).

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed. Exp. 12 was
re-run on 2026-09-27 (scripts/replicate/exp12_spectral_register_mono.sh, CPU) only to save
its per-pair record, which the 2026-08-17 run did not write; no calculation changed.

POSITIVE CONTROL, crossed first (exit 1 otherwise): the Exp. 12 record gives stereo 24/47,
mono 22/42, pooled 46/89 and the Exp. 11 record 24/47, as published; the 47 stereo pairs of
the two files match one to one on (subject, centroid distance), which identifies the mixture.
THEN: agreement pair by pair, with the agreement expected from the two accuracies alone as
its null, and the exact McNemar test on the discordant pairs (null: discordances split 50/50).

Usage: /opt/miniconda3/bin/python scripts/replicate/register_exp11_vs_exp12_pairs.py
"""
import os
import sys

import pandas as pd
from scipy.stats import binomtest

PROV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "provenance")
E11 = "2026-08-17_exp11_primary_RESULT_pairs.csv"
E12 = "2026-09-27_exp12_primary_RESULT_pairs.csv"
OUT = "2026-09-27_register_exp11_vs_exp12_pairs.txt"


def main():
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed:",
         "!! the 47 stereo register pairs under the authors' cleaning (Exp. 11) and under the",
         "!! raw-release cleaning (Exp. 12), compared decision by decision. Exploratory: the",
         "!! duos are spent material.",
         f"inputs: {E11} · {E12}", ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    a = pd.read_csv(os.path.join(PROV, E11), dtype={"subj": str})
    b = pd.read_csv(os.path.join(PROV, E12), dtype={"subj": str})
    bs = b[b.spatial == "stereo"]
    counts = [("Exp. 12 stereo", int(bs.hit.sum()), len(bs), (24, 47)),
              ("Exp. 12 mono", int(b[b.spatial == "mono"].hit.sum()), int((b.spatial == "mono").sum()), (22, 42)),
              ("Exp. 12 pooled", int(b.hit.sum()), len(b), (46, 89)),
              ("Exp. 11 stereo", int(a.hit.sum()), len(a), (24, 47))]
    say("=== POSITIVE CONTROL ===")
    ok = True
    for name, k, n, pub in counts:
        ok &= (k, n) == pub
        say(f"  {name:15s} {k}/{n}  published {pub[0]}/{pub[1]}  [{'OK' if (k, n) == pub else 'FAIL'}]")
    m = a.merge(bs, on=["subj", "dcent"], how="inner", suffixes=("_11", "_12"))
    one = len(m) == 47 and not a.duplicated(["subj", "dcent"]).any() and not bs.duplicated(["subj", "dcent"]).any()
    say(f"  stereo pairs matched one to one on (subject, centroid distance): {len(m)}/47  [{'OK' if one else 'FAIL'}]")
    if not (ok and one):
        say("POSITIVE CONTROL FAILED: nothing below this line exists.")
        open(os.path.join(PROV, OUT), "w").write("\n".join(L) + "\n")
        sys.exit(1)

    h11, h12 = m.hit_11.astype(bool), m.hit_12.astype(bool)
    both, neither = int((h11 & h12).sum()), int((~h11 & ~h12).sum())
    only11, only12 = int((h11 & ~h12).sum()), int((~h11 & h12).sum())
    agree = both + neither
    p11, p12 = h11.mean(), h12.mean()
    chance = p11 * p12 + (1 - p11) * (1 - p12)
    say()
    say("=== PAIR BY PAIR (47 stereo pairs, 8 participants) ===")
    say(f"  both correct {both} · both wrong {neither} · only Exp. 11 correct {only11} · only Exp. 12 correct {only12}")
    say(f"  agreement {agree}/47 = {agree / 47:.4f}   null (independent decisions with these accuracies) "
        f"{chance:.4f}   exact binomial p (one-sided, agreement above null) = "
        f"{binomtest(agree, 47, chance, alternative='greater').pvalue:.4f}")
    nd = only11 + only12
    say(f"  McNemar exact: discordant {nd}, split {only11} vs {only12}   null 50/50   two-sided p = "
        f"{binomtest(only11, nd, 0.5).pvalue:.4f}")
    say("  per subject (agreement / pairs):  " + " · ".join(
        f"{s} {int((g.hit_11 == g.hit_12).sum())}/{len(g)}" for s, g in m.groupby("subj")))
    open(os.path.join(PROV, OUT), "w").write("\n".join(L) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
