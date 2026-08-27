"""EXP. 16 ARM B -- does a RICHER EEG feature extraction help? band_power on own-vs-other.

Pre-registration, written BEFORE this file existed and not touched after:
  "Chapter 2 -- Exp. 16: the thesis premise put to the test -- CLAP as a target and rich
   EEG features, pre-registered criterion (2026-08-12)", section B.

THE QUESTION. The request was to combine "the best EEG feature extraction we have found".
The criterion corrects the premise: `--eeg_repr spectra` lives in the contrastive model -- the
family that memorises song identity -- and is excluded; and no EEG front end has ever been
shown better in this project. What DOES exist inside the safe family is
`src/models/cca_multiview.py::EEG_VIEWS`, whose `band_power` view (per-channel log power
envelope, one block per declared band) is the honest version of the idea. This file runs
it, once, on own-vs-other.

PRELIMINARY GATE (contract section B.2), settled BEFORE this file was written and recorded here:
the pinned 209/376 CCA run is `runs/results/ovo_cca_ica/`, whose summary reads
`MODEL 9: views=eeg_lagged stim_views=source_repr k=5 reg=0.001` -- the SINGLE view. So the
arm is not already done, and it proceeds. Had it read `eeg_lagged,band_power` this file
would not exist.

K = 1. ONE candidate, declared in the contract before this file existed:
  B1 = --estimator cca --cca_views eeg_lagged,band_power   (everything else invariant)
Primary bar: accuracy >= 208/376, one-sided exact binomial P(X >= 208 | 376, 0.5) = 0.0221
<= 0.025 = 0.05/2 (Bonferroni on K = 2 with arm A). 208/376 is the mel-8 ridge reference, so
the bar reads "tracks at least as well as what we use today".

WHAT IS SPENT: nothing. own-vs-other runs on the SOLOS, declared free training material by
the held-out-look ledger. No duo decision, no trio, no split by genre/melody/instrument/
subject. NOTE the standing debt the contract makes explicit: this is the 13th comparison on
the same 376 decisions (Exp. 7 x8, Exp. 8 x2, Exp. 14 x2, this x1). Free in the ledger of
LOOKS, not free in the family-wise error rate. Printed with the verdict.

GATES, DECLARED HERE AND CROSSED BEFORE ANY REAL NUMBER IS READ (contract section B.3):
  G1 REGRESSION CANARY on the DEFAULT path: --target mel --n_mels 8, ridge, must still give
     208/376 with md5(madeeg_ownvsother.csv) = 2eaa926244de340d31907c6deeb04b0c. If it
     moves, the diff touched the science: STOP.
  G2 SINGLE-VIEW CCA REPRODUCTION (an addition of this file, not a relaxation -- section 8 of the
     report). The contract asks for a descriptive McNemar against "the 209/376 of the
     single-view CCA". That number was measured on 2026-08-10 and the file has been edited
     twice since (Exp. 14 added --target mfcc). Re-measured here: it must give 209/376 AND
     be decision-identical to the archived run, comparison by comparison. Otherwise the
     object the McNemar compares against is not the pinned one: STOP.
  G3 POSITIVE CONTROL of the MULTI-VIEW CCA path (contract section B.3), threshold >= 0.95 -- the
     contract raises the 0.90 the file itself still prints, and any value between the two
     counts as FAILED here. Exp. 8/14 construction: every EEG replaced by a synthetic linear
     mixture of that trial's attended representation at the model lags + noise. It licenses
     the wiring of the multi-view path and says NOTHING about the real data. Crossed BEFORE
     the real accuracy is read.
Any gate failing => this script writes what it has and exits non-zero. Nothing is run "just
to see", and no second candidate exists in any branch.

Run (CPU, ~10', interpreter is NOT optional -- environment trap 1):
  /opt/miniconda3/bin/python src/madeeg_exp16b_ccaviews.py --madeeg_dir ~/madeeg
"""
import os
import re
import sys
import argparse
import subprocess
from math import comb

import numpy as np
import pandas as pd

PY = "/opt/miniconda3/bin/python"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON = os.path.join(ROOT, "src", "madeeg_reconstruction.py")
DIAGNOSE = os.path.join(ROOT, "src", "madeeg_diagnose.py")

# ---- thresholds, contract section B.3. Not editable after the fact. ----
CANARY_K, CANARY_N = 208, 376
CANARY_MD5 = "2eaa926244de340d31907c6deeb04b0c"
CCAREF_K, CCAREF_N = 209, 376          # the pinned single-view CCA, Exp. 6 [G]
CCAREF_DIR = os.path.join(ROOT, "runs", "results", "ovo_cca_ica")   # the 10/8 run, on disk
CTRL_MIN = 0.95                        # contract; the file's own built-in line still says 0.90
BAR_K, BAR_N = 208, 376                # primary: >= 208/376, one-sided exact p = 0.0221

# ---- the K = 1 candidate, contract section B.3. Declared before the run. ----
CANDIDATE = ("B1 CCA eeg_lagged+band_power", "cca_bp",
             ["--estimator", "cca", "--cca_views", "eeg_lagged,band_power"])
SINGLE_VIEW = ["--estimator", "cca", "--cca_views", "eeg_lagged"]

# Everything outside `extra` is the protocol of the pinned runs, byte for byte.
OVO_ARGS = ["--train_on", "raw_solos", "--own_vs_other", "--eeg_clean", "notch_ica",
            "--filters", "pooled", "--target", "mel", "--n_mels", "8", "--target_fs", "64",
            "--band_low", "1", "--band_high", "8", "--lags_ms", "250", "--seed", "42"]
CTRL_ARGS = ["--self_test", "--target_fs", "64", "--band_low", "1", "--band_high", "8",
             "--lags_ms", "250", "--seed", "42"]

CAVEAT = [
    "!! READ BEFORE THE NUMBERS (method rule 9).",
    "!! WHAT THIS MEASURES: own-vs-other measures TRACKING and nothing else. Given the EEG",
    "!! of a held-out SOLO segment, is that segment's own audio scored above ANOTHER solo's?",
    "!! The other segment is a different song, so nothing here says anything about separating",
    "!! the two stems of a duo, and nothing here is an attention decision.",
    "!! NULL = 0.500, EXACT BY SYMMETRY, not estimated: every pair is scored in both",
    "!! directions, so a decoder with nothing but a fixed instrument preference gets exactly",
    "!! one of the two right. The null is printed next to every accuracy.",
    "!! NOTHING IS SPENT FROM THE LEDGER OF LOOKS: the solos are declared free training",
    "!! material. No duo decision, no trio, no split by genre/melody/instrument/subject.",
    "!! BUT IT IS NOT FREE IN THE FAMILY-WISE ERROR RATE: with Exp. 7 (8 candidates),",
    "!! Exp. 8 (2) and Exp. 14 (2), this is the 13th comparison on the SAME 376 decisions.",
    "!! Any sentence of the form 'we found something that beats 208/376' must carry that",
    "!! count next to it (contract, preamble).",
    "!! THE POSITIVE-CONTROL NUMBER IS NOT A RESULT. It licenses the wiring of the",
    "!! multi-view path and says nothing about the real data.",
    "!! rho (CCA) and band_pearson (ridge) are DIFFERENT statistics and are never compared.",
    "!! Only the ACCURACY is comparable across estimators, because it is dimensionless.",
    "!! K = 1 for this arm and the bar (>= 208/376) were written in the pre-registration of",
    "!! 12/8/2026 BEFORE this file existed, and are not editable now. McNemar is DESCRIPTIVE,",
    "!! in both directions, against BOTH references, and does NOT enter the verdict.",
]


def exact_p(k, n):
    """One-sided exact binomial p against p0 = 0.5 -- P(X >= k)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def run_recon(madeeg_dir, tag, extra):
    """One madeeg_reconstruction.py run into its own directory (method rule 9: a control
    never writes over a result). Returns the output dir."""
    cmd = [PY, RECON, "--madeeg_dir", madeeg_dir, "--training_date", tag] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        sys.exit(f"run {tag} failed with code {r.returncode} -- STOP")
    return os.path.join(ROOT, "runs", "results", tag)


def ovo_result(d):
    """(k, n) from an own-vs-other summary, read from the file rather than from stdout."""
    txt = open(os.path.join(d, "madeeg_ownvsother_summary.txt")).read()
    m = re.search(r"own-vs-other accuracy: (\d+)/(\d+)", txt)
    assert m, f"no accuracy line in {d}"
    return int(m.group(1)), int(m.group(2))


def selftest_acc(d):
    # --self_test writes to <training_date>_selftest, never over a real run's directory.
    txt = open(os.path.join(d + "_selftest", "madeeg_selftest_summary.txt")).read()
    m = re.search(r"AAD accuracy=([0-9.]+)", txt)
    assert m, f"no accuracy line in {d}"
    return float(m.group(1))


def summary_lines(d, prefixes):
    txt = open(os.path.join(d, "madeeg_ownvsother_summary.txt")).read()
    return [l for l in txt.splitlines() if l.startswith(prefixes)]


def selftest_line(d, prefix):
    txt = open(os.path.join(d + "_selftest", "madeeg_selftest_summary.txt")).read()
    return [l for l in txt.splitlines() if l.startswith(prefix)]


def decision_identity(csv_a, csv_b):
    """G2. Not 'the same total' but 'the same decisions': merge on the comparison key and
    count the rows where the two runs disagree. A run that reaches 209 by a different route
    is not the object the contract names."""
    key = ["subject", "fold", "seg", "other"]
    a = pd.read_csv(csv_a)[key + ["correct"]]
    b = pd.read_csv(csv_b)[key + ["correct"]]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    return len(a), len(b), len(m), int((m.correct_a != m.correct_b).sum())


def mcnemar(a_csv, b_csv):
    r = subprocess.run([PY, DIAGNOSE, "--mcnemar", a_csv, b_csv], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mcnemar failed: {r.stderr[-2000:]}")
    return r.stdout.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out", default="", help="provenance file (default: "
                                              "docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt)")
    args = ap.parse_args()
    md = os.path.expanduser(args.madeeg_dir)
    out_path = args.out or os.path.join(ROOT, "docs", "provenance",
                                        "2026-08-12_exp16B_ccaviews_ownvsother.txt")
    L = list(CAVEAT)

    def say(line=""):
        L.append(line)
        print(line, flush=True)

    def dump(code=1):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w").write("\n".join(L) + "\n")
        print(f"\n-> {out_path}")
        if code:
            sys.exit(code)

    say()
    say("=== EXP. 16 ARM B -- a richer EEG feature extraction on own-vs-other ===")
    say(f"interpreter {PY} · protocol: raw_solos, pooled, notch+ICA, mel-8 target, 1-8 Hz,")
    say("64 Hz, lags 0-250 ms, seed 42, 5 folds -- every one of them identical to the two")
    say("pinned references. Single variable: the EEG-side feature set of the CCA.")

    # ---------------- the preliminary gate, settled before the code ----------------
    say()
    say("=== PRELIMINARY GATE (contract section B.2) -- which views produced the pinned 209/376? ===")
    ref_sum = os.path.join(CCAREF_DIR, "madeeg_ownvsother_summary.txt")
    model9 = [l for l in open(ref_sum).read().splitlines() if l.startswith("MODEL 9:")]
    say(f"  {CCAREF_DIR}/madeeg_ownvsother_summary.txt")
    for l in model9:
        say(f"    {l}")
    if any("band_power" in l for l in model9):
        say("  -> band_power was ALREADY in the pinned run: the contract says close the arm "
            "without running anything, and that is what happens.")
        dump()
    say("  -> the pinned run used the SINGLE view eeg_lagged: the arm is NOT already done, "
        "and it proceeds.")

    # ---------------- G1: regression canary on the DEFAULT path ----------------
    say()
    say("=== GATE 1 -- REGRESSION CANARY on the default path (crossed before anything else) ===")
    d_ref = run_recon(md, "exp16b_gate1_canary_mel8", OVO_ARGS)
    k, n = ovo_result(d_ref)
    ref_csv = os.path.join(d_ref, "madeeg_ownvsother.csv")
    h = md5(ref_csv)
    ok = (k, n) == (CANARY_K, CANARY_N) and h == CANARY_MD5
    say(f"  ridge --target mel --n_mels 8 -> {k}/{n} (expected {CANARY_K}/{CANARY_N})   null 0.500")
    say(f"  md5(madeeg_ownvsother.csv) = {h}")
    say(f"                    expected = {CANARY_MD5}")
    say(f"  -> [{'PASSED' if ok else 'FAILED'}] the default path is bit-for-bit what it was.")
    if not ok:
        say("  CANARY FAILED -> the diff moved the science. Nothing is run (method rules 3/8).")
        dump()

    # ---------------- G2: the single-view CCA reproduces, decision by decision ----------
    say()
    say("=== GATE 2 -- SINGLE-VIEW CCA REPRODUCTION (addition of this file, not a relaxation) ===")
    say(f"  The contract's descriptive comparison is against 'the {CCAREF_K}/{CCAREF_N} of the")
    say("  single-view CCA', measured on 10/8. madeeg_reconstruction.py has been edited twice")
    say("  since (Exp. 14). Re-measured here; it must reproduce the count AND the decisions.")
    d_cca1 = run_recon(md, "exp16b_gate2_cca_single", OVO_ARGS + SINGLE_VIEW)
    k1, n1 = ovo_result(d_cca1)
    cca1_csv = os.path.join(d_cca1, "madeeg_ownvsother.csv")
    na, nb, ncom, ndis = decision_identity(os.path.join(CCAREF_DIR, "madeeg_ownvsother.csv"),
                                           cca1_csv)
    ok = (k1, n1) == (CCAREF_K, CCAREF_N) and ndis == 0 and ncom == CCAREF_N
    say(f"  --cca_views eeg_lagged -> {k1}/{n1} (expected {CCAREF_K}/{CCAREF_N})   null 0.500")
    say(f"  decision-identity against the archived 10/8 run: {ncom}/{CCAREF_N} comparisons in "
        f"common, discordant = {ndis} (expected 0)")
    say(f"  md5 archived {md5(os.path.join(CCAREF_DIR, 'madeeg_ownvsother.csv'))} · "
        f"today {md5(cca1_csv)}")
    say("  (the md5 may differ without the gate failing: it also hashes the float columns.")
    say("   The gate is on the DECISIONS, which are what the McNemar pairs.)")
    say(f"  -> [{'PASSED' if ok else 'FAILED'}]")
    if not ok:
        say("  The single-view CCA is no longer the object the contract names -> STOP "
            "(method rule 8).")
        dump()

    # ---------------- G3: positive control of the MULTI-VIEW path ----------------
    name, tag, flags = CANDIDATE
    say()
    say(f"=== GATE 3 -- POSITIVE CONTROL of the multi-view CCA path, threshold >= {CTRL_MIN:.2f} ===")
    say("  Synthetic EEG (Exp. 8/14 construction). A PASS licenses the wiring of the multi-view")
    say("  path and says NOTHING about the real data. The file's own built-in PASS line still")
    say(f"  reads 0.90; the contract's {CTRL_MIN:.2f} is applied HERE and anything between the")
    say("  two counts as FAILED.")
    d_ctrl = run_recon(md, f"exp16b_gate3_ctrl_{tag}", CTRL_ARGS + flags)
    acc = selftest_acc(d_ctrl)
    ok = acc >= CTRL_MIN
    say(f"  {name} AAD accuracy on synthetic EEG = {acc:.4f}   chance 0.500   "
        f"-> [{'PASSED' if ok else 'FAILED'}]")
    for l in selftest_line(d_ctrl, "views="):
        say(f"    {l}")
    say("    ^ blocks_x is the measured proof that band_power actually entered, and with how")
    say("      many columns: 20 channels x the bands that FIT inside 1-8 Hz (delta 1-4, theta")
    say("      4-8). alpha and beta are ZERO by construction inside this band and the module")
    say("      drops them and declares the drop -- widening the band would be a separate,")
    say("      declared change and is NOT made here.")
    if not ok:
        say(f"  positive control below {CTRL_MIN:.2f} -> the candidate is not reported "
            "(method rule 3).")
        dump()
    say("  The wiring is licensed. The real number may now be read.")

    # ---------------- the real measurement ----------------
    say()
    say("=== THE MEASUREMENT -- own-vs-other on held-out SOLO segments, K = 1 ===")
    say(f"  primary bar, written before the code: >= {BAR_K}/{BAR_N}, one-sided exact binomial")
    say(f"  P(X >= {BAR_K} | {BAR_N}, 0.5) = {exact_p(BAR_K, BAR_N):.4f} <= 0.025 = 0.05/2 "
        "(Bonferroni, K = 2 with arm A).")
    say()
    d_cand = run_recon(md, f"exp16b_{tag}_ovo", OVO_ARGS + flags)
    kc, nc = ovo_result(d_cand)
    cand_csv = os.path.join(d_cand, "madeeg_ownvsother.csv")
    passed = kc >= BAR_K
    say(f"  {name:28s} {kc}/{nc} = {kc / nc:.4f}   null 0.500   one-sided exact binomial "
        f"p = {exact_p(kc, nc):.4g}   bar {BAR_K}/{BAR_N}   "
        f"-> {'CLEARS THE BAR' if passed else 'MISSES THE BAR'}")
    say(f"  {'ridge mel-8 (ref, = the bar)':28s} {CANARY_K}/{CANARY_N} = "
        f"{CANARY_K / CANARY_N:.4f}   null 0.500   p = {exact_p(CANARY_K, CANARY_N):.4g}")
    say(f"  {'CCA single view (ref)':28s} {k1}/{n1} = {k1 / n1:.4f}   null 0.500   "
        f"p = {exact_p(k1, n1):.4g}")

    say()
    say("  -- the lines that carry the CAUSE rather than the headline (null 0.500 each) --")
    for label, d in ((name, d_cand), ("CCA single view", d_cca1), ("ridge mel-8", d_ref)):
        say(f"  {label}:")
        for l in summary_lines(d, ("per-subject:", "mean rho", "mean band_pearson", "MODEL 9:")):
            say(f"    {l}")
    say("  Read them for the two questions the headline cannot answer: whether a change is one")
    say("  subject or many, and whether the accuracy follows the SIZE of the similarity. The")
    say("  decision is an argmax between two scores of the SAME estimator, so it is invariant")
    say("  to a uniform rescaling of them -- a bigger mean rho is not a better decoder.")

    # ---------------- descriptive McNemar, both references, both directions ----------
    say()
    say("=== DESCRIPTIVE -- McNemar, BOTH references, BOTH directions ===")
    say("  Does not enter the verdict (contract section B.3). Both directions are printed so that a")
    say("  candidate WORSE than a reference is as readable as one that is better. The second")
    say("  pair is what isolates what band_power adds, since it holds the estimator fixed and")
    say("  changes only the EEG feature set.")
    say(f"\n  -- {name} vs the ridge mel-8 reference ({CANARY_K}/{CANARY_N}) --")
    say(mcnemar(ref_csv, cand_csv))
    say(mcnemar(cand_csv, ref_csv))
    say(f"\n  -- {name} vs the single-view CCA ({CCAREF_K}/{CCAREF_N}) -- what band_power adds --")
    say(mcnemar(cca1_csv, cand_csv))
    say(mcnemar(cand_csv, cca1_csv))

    # ---------------- verdict, pre-declared ----------------
    say()
    say("=== VERDICT -- the reading was written before the numbers (contract section 5) ===")
    if passed:
        say("  B PASSES: a richer EEG feature helps the tracking. The contract's pre-declared")
        say("  reading is NOT CONCLUSIVE -- 'to be confirmed OUTSIDE own-vs-other before any claim'. It is")
        say("  not a result about attention: own-vs-other upper-bounds the attention decision")
        say("  and has already failed to predict it once (Exp. 8 positive -> Exp. 9 negative).")
    else:
        say("  B FAILS: on the EEG side too, the hand-picked-feature route is closed, with")
        say("  band_power -- the only EEG view this project ever proposed from the literature")
        say("  and never tried. Pre-declared reading, contract section 5.")
    say()
    say(f"  Family-wise bookkeeping, printed with the verdict: this is comparison 13 on the")
    say(f"  same 376 decisions (Exp. 7 x8 + Exp. 8 x2 + Exp. 14 x2 + this x1). The bar itself")
    say(f"  is Bonferroni-corrected only for K = 2 WITHIN Exp. 16.")
    say("  What this arm does NOT say, in any branch: nothing about the duo attention")
    say("  decision, nothing about any split, and nothing that licenses a second candidate.")

    dump(code=0)


if __name__ == "__main__":
    main()
