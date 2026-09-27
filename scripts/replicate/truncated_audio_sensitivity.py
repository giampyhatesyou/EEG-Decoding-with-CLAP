"""Sensitivity of the raw-path duo counts to the eight truncated trials (Ch. 8, truncated trials).

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed. It opens no
EEG and trains nothing: it recounts the decisions already saved in docs/provenance.

WHAT: on pop_mixtape_BsDr_theme2 the fourth released wav is 4.966 s instead of ~6 s, so in
the eight STEREO trials of that mixture 1.04 s of EEG is paired with audio never played.
Every record file in docs/provenance that contains the mixture is listed. A file is
recounted only if EVERY row has fold == -1 (decoder trained on the solos, so the truncated
trials sit only in the test set and removing them needs no retraining); any other file is
listed and NOT recounted. Mono rows are not affected (Ch. 8: worst discrepancy 45 ms).
POSITIVE CONTROL, per file, before its recount: the accuracy of the saved decisions equals
the OVERALL AAD accuracy printed by the run's own summary; for the three numbers the thesis
quotes, the exact count too (74/154, 76/154, 86/150). A failure exits with no number.

Usage: /opt/miniconda3/bin/python scripts/replicate/truncated_audio_sensitivity.py
"""
import glob
import os
import re
import sys

import pandas as pd
from scipy.stats import binomtest

MIX = "pop_mixtape_duo_BsDr_theme2"
MIX_ALPHA = "pop_mixtape_BsDr_theme2"      # the alpha files name mixtures without `duo`
OUT = "2026-09-27_truncated_audio_sensitivity.txt"
QUOTED = {"2026-08-10_canary_rawsolos": (74, 154), "2026-08-11_exp9_P1_rawsolos_flux": (76, 154),
          "2026-07-29_ax_ica_perinstr": (86, 150)}


def p1(k, n):
    return binomtest(k, n, 0.5, alternative="greater").pvalue


def main():
    prov = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "provenance")
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed.",
         "!! A recount of saved decisions without the 8 truncated stereo trials; no EEG opened,",
         "!! nothing retrained, no new look. Every p is exact one-sided binomial against the",
         "!! null 0.500 (every duo mixture is presented with both instruments as the target).",
         ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    rows, skipped = [], []
    for f in sorted(glob.glob(os.path.join(prov, "*.csv"))):
        d = pd.read_csv(f, dtype={"subject": str})
        col = "stim" if "stim" in d else "mixture" if "mixture" in d else None
        if col is None or not d[col].astype(str).str.contains(f"{MIX}|{MIX_ALPHA}").any():
            continue
        name = os.path.basename(f)
        hit = d[col].astype(str).str.contains(f"{MIX}|{MIX_ALPHA}")
        if "fold" not in d or "correct" not in d:
            skipped.append((name, "no fold column: training on solos cannot be verified"))
            continue
        if (d.fold != -1).any():
            skipped.append((name, f"folds {sorted(d.fold.unique().tolist())}: the truncated trials "
                                  "may also enter training"))
            continue
        stereo = hit & d[col].str.contains("_stereo_")
        if not stereo.any():
            skipped.append((name, f"its {int(hit.sum())} rows of the mixture are mono: not affected"))
            continue
        tag = name.split("_RESULT_")[0]
        summ = f.replace("_records.csv", "_summary.txt")
        acc = re.search(r"OVERALL AAD accuracy: ([0-9.]+)", open(summ).read()).group(1)
        k, n = int(d.correct.sum()), len(d)
        ok = f"{k / n:.4f}" == acc and QUOTED.get(tag, (k, n)) == (k, n)
        say(f"  control {tag:40s} {k}/{n} = {k / n:.4f}  summary {acc}"
            f"{'  quoted %d/%d' % QUOTED[tag] if tag in QUOTED else ''}  [{'OK' if ok else 'FAIL'}]")
        if not ok:
            say("POSITIVE CONTROL FAILED: nothing below this line exists.")
            open(os.path.join(prov, OUT), "w").write("\n".join(L) + "\n")
            sys.exit(1)
        kk, nn = int(d.correct[~stereo].sum()), int((~stereo).sum())
        rows.append((tag, k, n, kk, nn, int(stereo.sum()), int(d.correct[stereo].sum())))
    say("  -> every recounted file reproduces its run's own total.")

    say()
    say(f"=== RECOUNT without the stereo trials of {MIX} (fold == -1 in every row) ===")
    say(f"  {'run':40s} {'with':>14s} {'p':>7s} | {'without':>14s} {'p':>7s} | removed")
    for tag, k, n, kk, nn, nr, kr in rows:
        say(f"  {tag:40s} {k:>3d}/{n} {k / n:.4f} {p1(k, n):7.4f} | {kk:>3d}/{nn} {kk / nn:.4f} "
            f"{p1(kk, nn):7.4f} | {kr}/{nr} correct")
    say()
    say("=== LISTED, NOT RECOUNTED ===")
    for name, why in skipped:
        say(f"  {name}: {why}")
    open(os.path.join(prov, OUT), "w").write("\n".join(L) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
