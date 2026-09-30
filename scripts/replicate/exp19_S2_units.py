"""The independent units behind the 113/188 of Exp. 19 stage 2, counted from its own records.

ANALYSIS OF 2026-09-30, on records already on disk (docs/provenance): no EEG is read, no model
is trained, no new split is made, so it spends no look. The published p = 0.003404 treats the
188 held-out pairs as independent, and the run itself calls it descriptive because the pairs
share EEG windows. This script counts the units the pairs come from (recordings, participants)
and prints the exact 95% interval and the sign test over recordings.

POSITIVE CONTROL, crossed first (exit 1 otherwise): the published figures of both stages,
recomputed from the same records (counts, per-recording means, number of recordings).

Usage: /opt/miniconda3/bin/python scripts/replicate/exp19_S2_units.py
"""
import csv
import os
import sys
from collections import defaultdict

from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROV = os.path.join(ROOT, "docs", "provenance")
OUT = os.path.join(PROV, "2026-09-30_exp19_S2_units.txt")


def load(stage):
    rows = list(csv.DictReader(open(os.path.join(PROV, f"2026-08-13_exp19_{stage}_RESULT_records.csv"))))
    by_rec, by_subj = defaultdict(list), defaultdict(list)
    for r in rows:
        c = int(r["correct"])
        by_rec[(r["subject"], r["stim"])].append(c)
        by_subj[r["subject"]].append(c)
    return rows, by_rec, by_subj


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("!! ANALYSIS OF 2026-09-30 on the stored records of Exp. 19 (no EEG, no training, no new")
    say("!! split): the independent units behind the stage-2 figure 113/188.")
    say("")
    say("=== POSITIVE CONTROL (published figures, crossed first) ===")
    ok = True
    for stage, k_pub, n_pub, mean_pub, recs_pub in [("S1", 244, 536, 0.4215, 21), ("S2", 113, 188, 0.5999, 13)]:
        rows, by_rec, _ = load(stage)
        k = sum(int(r["correct"]) for r in rows)
        m = sum(sum(v) / len(v) for v in by_rec.values()) / len(by_rec)
        good = (k, len(rows), round(m, 4), len(by_rec)) == (k_pub, n_pub, mean_pub, recs_pub)
        ok &= good
        say(f"  {stage}: {k}/{len(rows)}  per-recording mean {m:.4f} over {len(by_rec)} recordings"
            f"  published {k_pub}/{n_pub}, {mean_pub}, {recs_pub}  [{'OK' if good else 'FAILED'}]")
    if not ok:
        say("POSITIVE CONTROL FAILED -- STOP")
        sys.exit(1)
    say("")

    rows, by_rec, by_subj = load("S2")
    k, n = sum(int(r["correct"]) for r in rows), len(rows)
    ci2 = binomtest(k, n).proportion_ci(0.95, method="exact")
    say("=== STAGE 2, THE PAIRS (null 0.5000, measured after balancing) ===")
    say(f"  {k}/{n} = {k / n:.4f}   exact two-sided 95% CI [{ci2.low:.3f}, {ci2.high:.3f}]"
        f"   one-sided binomial p = {binomtest(k, n, 0.5, alternative='greater').pvalue:.6f}"
        "  (descriptive: pairs share EEG windows)")
    say("")
    say("=== THE PARTICIPANTS BEHIND THE 188 PAIRS ===")
    for s, v in sorted(by_subj.items()):
        say(f"  {s}: {sum(v)}/{len(v)} = {sum(v) / len(v):.4f}")
    say(f"  -> {len(by_subj)} of 8 participants contribute held-out pairs")
    say("")
    say("=== THE RECORDINGS (subject x solo key) ===")
    above = sum(sum(v) / len(v) > 0.5 for v in by_rec.values())
    equal = sum(sum(v) / len(v) == 0.5 for v in by_rec.values())
    for (s, stim), v in sorted(by_rec.items()):
        say(f"  {s} {stim}: {sum(v)}/{len(v)}")
    say(f"  -> {above} of {len(by_rec)} recordings above 0.5, {equal} exactly at 0.5")
    p_sign = binomtest(above, len(by_rec), 0.5, alternative="greater").pvalue
    say(f"  sign test over recordings, one-sided, null 0.5: P(X >= {above} | {len(by_rec)}) = {p_sign:.4f}")
    say("  (recordings of the same participant are not independent either, so this is still")
    say("   generous: the participant is the unit a new-trial claim would generalise over)")
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
