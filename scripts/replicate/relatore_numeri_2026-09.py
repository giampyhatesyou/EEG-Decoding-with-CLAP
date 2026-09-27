"""The small numbers asked by the supervisor on 2026-09-26, in one place.

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed: arithmetic on
counts already published (no record is re-read, no EEG opened), plus one count over the
file tree of the released benchmark split.

  1. Fisher's exact test, paired alpha mono 24/42 against stereo 23/44 (expected p = 0.67).
  2. Tab. 9.1: exact one-sided binomial p and exact 95% interval of the three anchors.
  3. Exact 95% intervals of the null results quoted in the answer to the supervisor.
  4. The within split of the benchmark (dataset.zip): of the 62 test trials, how many have
     their song in the training data of the same participant, of another, of nobody.
  5. G3: every count the thesis reports as not significant (p > 0.05 or at the null) in the
     abstract and Ch. 1, 2, 7, 9, 10, 11, 12, with its exact 95% interval and the file and
     line where it appears; written also as a TSV for the revision.
POSITIVE CONTROL, crossed first (exit 1 otherwise): items 1-3 reproduce the values already
computed on 2026-09-26 (p 0.67; 0.0853/0.4679/0.7136 and their intervals; the six null
intervals) and item 4 the count of 2026-09-25 (0 / 59 / 3). Every interval is
Clopper-Pearson; every p is against the null printed next to it.

Usage: /opt/miniconda3/bin/python scripts/replicate/relatore_numeri_2026-09.py \
         --thesis <thesis dir> [--dataset_zip ...] [--tsv <IC_nulli.tsv>]
"""
import argparse
import glob
import os
import re
import sys
import zipfile

from scipy.stats import binomtest, fisher_exact

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROV = os.path.join(ROOT, "docs", "provenance")
OUT = "2026-09-27_relatore_numeri.txt"
TEX = ["chapters/00b_abstract.tex", "chapters/01_summary.tex", "chapters/02_introduction.tex",
       "chapters/07_confound.tex", "chapters/09_results.tex", "chapters/10_audits.tex",
       "chapters/11_groundwork.tex", "chapters/12_discussion.tex"]

TAB91 = [("ridge k-fold, log-mel (A1)", 86, 154, 0.0853, (0.476, 0.638)),
         ("ridge, Hilbert envelope (A2)", 78, 154, 0.4679, (0.425, 0.588)),
         ("ridge, published protocol (A3)", 74, 154, 0.7136, (0.399, 0.562))]
NULLS = [("register, pooled 89 pairs", 46, 89, (0.408, 0.624)), ("register, stereo", 24, 47, (0.361, 0.659)),
         ("register, mono", 22, 42, (0.364, 0.680)), ("flux on duos, paper protocol", 76, 154, (0.412, 0.575)),
         ("flux on duos, k-fold", 77, 154, (0.418, 0.582)), ("mono replication (Exp. 4)", 79, 150, (0.444, 0.609))]

# G3: the counts the thesis reports as not significant, their null and their provenance.
# Chosen by reading the context of every k/n in the eight files (2026-09-27). Left out on
# purpose: thresholds, positive controls, positive results, descriptive cells of step C/D,
# and the genre split 47/90 · 32/60 (a cut R-1 forbids even as arithmetic).
G3 = {
    (86, 154): ("anchor A1, ridge k-fold, log-mel", 0.5, "2026-08-27_canary_duo_RESULT_*"),
    (78, 154): ("anchor A2, ridge on the Hilbert envelope", 0.5, "run archive (A2), scripts/replicate/README.md, ridge anchors"),
    (74, 154): ("anchor A3, ridge, published protocol", 0.5, "2026-08-10_canary_rawsolos_RESULT_*"),
    (76, 154): ("Exp. 9 flux, paper protocol (P1)", 0.5, "2026-08-11_exp9_P1_rawsolos_flux_RESULT_*"),
    (77, 154): ("Exp. 9 flux, k-fold (P2)", 0.5, "2026-08-27_exp9_P2_duo_flux_RESULT_*"),
    (65, 154): ("Exp. 15 MFCC, k-fold", 0.5, "2026-08-12_exp15_mfcc_differential.txt"),
    (75, 154): ("Exp. 15 MFCC, paper protocol", 0.5, "2026-08-12_exp15_mfcc_differential.txt"),
    (58, 154): ("step C, contrastive CLAP", 0.5, "results_manifest.tsv, madeeg_clap_kfold"),
    (70, 154): ("step D, contrastive CLAP within-mixture", 0.5, "2026-08-27_madeeg_clap_kfold_within_RESULT_*"),
    (80, 154): ("Exp. 14, one of the four directions", 0.5, "2026-08-12_exp14_mfcc_tracking.txt"),
    (28, 52): ("Exp. 14, one of the four directions", 0.5, "2026-08-12_exp14_mfcc_tracking.txt"),
    (24, 52): ("Exp. 14, one of the four directions", 0.5, "2026-08-12_exp14_mfcc_tracking.txt"),
    (79, 150): ("Exp. 4 mono replication", 0.5, "2026-07-30_exp04_mono_confirm_RESULT_*"),
    (86, 150): ("axis study maximum (nominal p 0.043, 16 configurations)", 0.5, "2026-07-29_ax_ica_perinstr_RESULT_*"),
    (202, 376): ("Exp. 6 own-vs-other, ridge without cleaning", 0.5, "2026-08-10_exp06_ovo_ridge_RESULT_*"),
    (66, 131): ("Exp. 6 McNemar, CCA vs ridge with ICA (discordant split)", 0.5, "2026-08-20_exp06_ovo_gate_RESULT_mcnemar.txt"),
    (72, 139): ("Exp. 6 McNemar, CCA vs ridge without ICA (discordant split)", 0.5, "2026-08-20_exp06_ovo_gate_RESULT_mcnemar.txt"),
    (33, 61): ("Exp. 7 McNemar, best configuration C3 (discordant split)", 0.5, "2026-08-11_ovo_sweep_mcnemar.txt"),
    (33, 64): ("Exp. 9 McNemar P1, flux vs mel (discordant split)", 0.5, "2026-08-11_exp9_flux_attention_mcnemar.txt"),
    (26, 61): ("Exp. 9 McNemar P2, flux vs mel (discordant split)", 0.5, "2026-08-11_exp9_flux_attention_mcnemar.txt"),
    (66, 138): ("Exp. 16B, one of the four directions", 0.5, "2026-08-12_exp16B_ccaviews_ownvsother.txt"),
    (72, 138): ("Exp. 16B, one of the four directions", 0.5, "2026-08-12_exp16B_ccaviews_ownvsother.txt"),
    (43, 93): ("Exp. 16B, one of the four directions", 0.5, "2026-08-12_exp16B_ccaviews_ownvsother.txt"),
    (50, 93): ("Exp. 16B, one of the four directions", 0.5, "2026-08-12_exp16B_ccaviews_ownvsother.txt"),
    (24, 47): ("Exp. 11 register, stereo (also Exp. 12 stereo from raw)", 0.5, "2026-08-17_exp11_primary_RESULT_*"),
    (11, 22): ("Exp. 11 dose-response, far pairs", 0.5, "2026-08-17_exp11_primary_RESULT_summary.txt"),
    (13, 25): ("Exp. 11 dose-response, near pairs", 0.5, "2026-08-17_exp11_primary_RESULT_summary.txt"),
    (22, 42): ("Exp. 12 register, mono", 0.5, "2026-08-17_exp12_primary_RESULT_*"),
    (46, 89): ("Exp. 12 register, pooled", 0.5, "2026-08-17_exp12_primary_RESULT_*"),
    (10, 16): ("Exp. 12, both trials at concentration 5 (descriptive)", 0.5, "2026-08-17_exp12_primary_RESULT_summary.txt"),
    (23, 44): ("paired alpha, stereo", 0.5, "2026-08-17_alpha_real_stereo_RESULT_*"),
    (24, 42): ("paired alpha, mono", 0.5, "2026-08-17_alpha_real_mono_RESULT_*"),
    (278, 536): ("Exp. 18 match-mismatch S1 (null 0.5000 measured)", 0.5, "2026-08-13_exp18_S1_RESULT_*"),
    (244, 536): ("Exp. 19 match-mismatch S1", 0.5, "2026-08-13_exp19_S1_RESULT_*"),
}
# the same k/n meaning something else on a given line
OVERRIDE = {("chapters/11_groundwork.tex", 140, 74, 154): ("Exp. 14, one of the four directions", 0.5, "2026-08-12_exp14_mfcc_tracking.txt"),
            ("chapters/11_groundwork.tex", 313, 202, 376): ("Exp. 16B, CCA with band power", 0.5, "2026-08-12_exp16B_ccaviews_ownvsother.txt")}


def cp(k, n):
    ci = binomtest(k, n).proportion_ci(method="exact")
    return round(ci.low, 3), round(ci.high, 3)


def p1(k, n, p0=0.5):
    return binomtest(k, n, p0, alternative="greater").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis", required=True)
    ap.add_argument("--dataset_zip", default=os.path.join(ROOT, "..", "upstream_akama", "dataset.zip"))
    ap.add_argument("--tsv", default="")
    args = ap.parse_args()
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed:",
         "!! arithmetic on published counts and one count over the released benchmark split.",
         "!! Intervals: exact (Clopper-Pearson) 95%. p: exact binomial, one-sided above the null",
         "!! printed next to it, unless stated.", ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    def done(code=0):
        open(os.path.join(PROV, OUT), "w").write("\n".join(L) + "\n")
        if code:
            sys.exit(code)

    say("=== POSITIVE CONTROL (values computed on 2026-09-25/26, crossed first) ===")
    ok = True
    f = fisher_exact([[24, 42 - 24], [23, 44 - 23]])
    good = round(f.pvalue, 2) == 0.67
    ok &= good
    say(f"  Fisher exact, alpha mono 24/42 vs stereo 23/44: two-sided p = {f.pvalue:.4f}  expected 0.67  [{'OK' if good else 'FAIL'}]")
    for lab, k, n, pp, ci in TAB91:
        good = round(p1(k, n), 4) == pp and cp(k, n) == ci
        ok &= good
        say(f"  Tab. 9.1 {lab:32s} {k}/{n} = {k / n:.4f}  p {p1(k, n):.4f}  95% [{cp(k, n)[0]:.3f}, {cp(k, n)[1]:.3f}]  "
            f"expected {pp} [{ci[0]}, {ci[1]}]  [{'OK' if good else 'FAIL'}]")
    for lab, k, n, ci in NULLS:
        good = cp(k, n) == ci
        ok &= good
        say(f"  null result {lab:30s} {k}/{n} = {k / n:.3f}  95% [{cp(k, n)[0]:.3f}, {cp(k, n)[1]:.3f}]  "
            f"expected [{ci[0]}, {ci[1]}]  [{'OK' if good else 'FAIL'}]")
    z = zipfile.ZipFile(args.dataset_zip)
    recs = [n.split("/")[2:7] for n in z.namelist()        # subject, split, song, task, attention
            if n.startswith("dataset/eeg_within_sub/") and n.endswith("eeg.pkl")]
    subs = {r[0] for r in recs}
    test = [r for r in recs if r[1] == "test"]
    split = {}
    for name, splits in (("train", ("train",)), ("train+valid", ("train", "valid"))):
        tr = {(r[2], r[0]) for r in recs if r[1] in splits}
        split[name] = (len(test), sum((r[2], r[0]) in tr for r in test),
                       sum(any((r[2], o) in tr for o in subs - {r[0]}) for r in test),
                       sum(not any((r[2], o) in tr for o in subs) for r in test))
    good = split["train"] == (62, 0, 59, 3)
    ok &= good
    say(f"  benchmark within split, song of a test trial in training: (test, same participant, another, nobody) "
        f"{split['train']}  expected (62, 0, 59, 3)  [{'OK' if good else 'FAIL'}]")
    if not ok:
        say("POSITIVE CONTROL FAILED: nothing below this line exists.")
        done(1)

    say()
    say("=== 1. Mono against stereo, paired alpha ===")
    say(f"  24/42 = 0.5714 vs 23/44 = 0.5227   null: equal accuracies   Fisher exact two-sided p = {f.pvalue:.4f}")
    say("  (the two sets share their 8 participants, so the test treats dependent sets as independent)")
    say()
    say("=== 2. Tab. 9.1 (n = 154, null 0.500, prospectively specified threshold 88/154) ===")
    for lab, k, n, _, _ in TAB91:
        say(f"  {lab:32s} {k}/{n} = {k / n:.4f}   one-sided p = {p1(k, n):.4f}   95% [{cp(k, n)[0]:.3f}, {cp(k, n)[1]:.3f}]")
    say()
    say("=== 3. Null results of the answer to the supervisor (null 0.500) ===")
    for lab, k, n, _ in NULLS:
        say(f"  {lab:30s} {k}/{n} = {k / n:.3f}   one-sided p = {p1(k, n):.4f}   95% [{cp(k, n)[0]:.3f}, {cp(k, n)[1]:.3f}]")
    say()
    say("=== 4. Benchmark within split (dataset.zip, eeg_within_sub/<participant>/<split>/<song>/...) ===")
    for name, (nt, same, other, none) in split.items():
        say(f"  training = {name:11s}: of {nt} test trials, song in training of the same participant {same}, "
            f"of another participant {other}, of nobody {none}")
    say(f"  ({len(subs)} participants, {len(recs)} trials; 'training' in the thesis is the train split)")

    say()
    say("=== 5. G3: every count reported as not significant, with its exact 95% interval ===")
    rows = []
    for rel in TEX:
        for i, line in enumerate(open(os.path.join(args.thesis, rel)), 1):
            if line.lstrip().startswith("%"):
                continue
            for m in re.finditer(r"(?<![\d.])(\d+)\s*(?:/|of|out of)\s*(\d+)(?![\d.])", line):
                k, n = int(m.group(1)), int(m.group(2))
                what = OVERRIDE.get((rel, i, k, n)) or G3.get((k, n))
                if what:
                    rows.append((rel, i, k, n) + what)
    missing = [kn for kn in G3 if not any((r[2], r[3]) == kn for r in rows)]
    say(f"  {len(rows)} occurrences of {len(G3)} counts in {len(TEX)} files"
        + (f"; not found in these files: {missing}" if missing else ""))
    for rel, i, k, n, lab, null, prov in rows:
        pat = prov.split(",")[0]
        if pat.startswith("20") and not glob.glob(os.path.join(PROV, pat)):
            say(f"  PROVENANCE MISSING for {k}/{n}: {pat}")
            done(1)
    say("  p is the exact binomial on the count; for own-vs-other the thesis prints the exact PAIRED")
    say("  p (e.g. 202/376: 0.0851 paired, 0.0819 binomial), which respects the 188 mutual pairs.")
    say(f"  {'file:line':34s} {'k/n':>8s} {'acc':>6s} {'95% CI':>15s} {'p':>7s}  what")
    for rel, i, k, n, lab, null, prov in rows:
        lo, hi = cp(k, n)
        say(f"  {rel.split('/')[-1] + ':' + str(i):34s} {k:>3d}/{n:<4d} {k / n:6.3f} [{lo:.3f}, {hi:.3f}] "
            f"{p1(k, n, null):7.4f}  {lab} (null {null})")
    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("file\triga\tk/n\tacc\tIC95\tp_unilaterale\tnull\tcosa\tprovenienza\n")
            for rel, i, k, n, lab, null, prov in rows:
                lo, hi = cp(k, n)
                fh.write(f"{rel}\t{i}\t{k}/{n}\t{k / n:.3f}\t{lo:.3f}-{hi:.3f}\t{p1(k, n, null):.4f}\t{null}\t{lab}\t{prov}\n")
        say(f"  table also written to {os.path.basename(args.tsv)} (vault, for the revision)")
    done()
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
