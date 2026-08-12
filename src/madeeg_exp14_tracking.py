"""EXP. 14 -- the TRACKING side of the gate: MFCC-13 and mel-64 on own-vs-other.

Vault contract, written BEFORE this file existed and not touched after:
  "Cap. 2 -- Exp. 14: il lato tracking del gate -- MFCC e mel-64 su own-vs-other,
   criterio pre-registrato (12 ago 2026)".

THE QUESTION. Exp. 13 (audio only) promoted MFCC-13 on SEPARABILITY: among six fixed
representations it is the one under which the two stems of a duo share the least. But
MFCC wins by dropping DCT coefficient 0 -- the overall log energy, i.e. the amplitude
envelope -- which is the one component with robustly documented cortical tracking. So the
gate may have bought separability with the very component the brain follows. own-vs-other
answers the other half, and only that half: given the EEG of a SOLO, is its own audio
scored above another solo's? The `other` is a DIFFERENT song, so stem separability does
not enter -- this measures TRACKING and nothing else.

K = 2, declared in the contract before this file existed:
  T1 = mfcc   (DCT-II ortho of log-mel-64, coefficients 1..13, c0 DROPPED)
  T2 = mel-64 (maximum spectral resolution with the energy intact)
Primary bar: accuracy >= 208/376, i.e. P(X >= 208 | 376, 0.5) = 0.0221 <= 0.025 = 0.05/2
(Bonferroni on K = 2). 208/376 is also the current mel-8 reference, so the bar reads
"tracks at least as well as the representation we use today".

WHAT IS SPENT: nothing. own-vs-other runs on the SOLOS, declared free training material by
the vault's ledger of looks and already used as the gauge by Exp. 7 and Exp. 8. No duo
decision, no trio, no split by genre/melody/instrument/subject.

GATES, DECLARED HERE AND CROSSED BEFORE ANY REAL NUMBER IS READ (contract SS2):
  G1 REGRESSION CANARY: --target mel --n_mels 8 must still give 208/376 with
     md5(madeeg_ownvsother.csv) = 2eaa926244de340d31907c6deeb04b0c. If it moves, the diff
     touched the science: STOP.
  G2 REPRESENTATION IDENTITY (an addition of this file, not a relaxation): the new
     --target mfcc branch must reproduce, to floating point on real audio, the C3
     construction Exp. 13 measured (madeeg_stem_separability._raw_mfcc + _post). Without
     it "MFCC-13 tracks" and "MFCC-13 separates" would be claims about two different
     objects. Audio only.
  G3 POSITIVE CONTROL per candidate: the Exp. 8 construction (--self_test: every EEG
     replaced by a synthetic linear mixture of that trial's attended representation at the
     model lags + noise), threshold >= 0.95 as the contract raises it (the file's own
     built-in PASS line still says 0.90 -- the stricter number is applied HERE and any
     candidate between the two counts as FAILED). Crossed BEFORE the real accuracies are
     read.
Any gate failing => this script writes what it has and exits non-zero. Nothing is run
"just to see".

Run (CPU, ~15', interpreter is NOT optional -- Legacy SS4 trap 1):
  /opt/miniconda3/bin/python src/madeeg_exp14_tracking.py --madeeg_dir ~/madeeg
"""
import os
import re
import sys
import glob
import argparse
import subprocess
from math import comb

import numpy as np

PY = "/opt/miniconda3/bin/python"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON = os.path.join(ROOT, "src", "madeeg_reconstruction.py")
DIAGNOSE = os.path.join(ROOT, "src", "madeeg_diagnose.py")

# ---- thresholds, contract SS2 and SS4. Not editable after the fact. ----
CANARY_K, CANARY_N = 208, 376
CANARY_MD5 = "2eaa926244de340d31907c6deeb04b0c"
CTRL_MIN = 0.95                 # contract SS2; the file's own built-in line still says 0.90
BAR_K, BAR_N = 208, 376         # primary: >= 208/376, one-sided exact p = 0.0221 <= 0.025

# ---- the K = 2 candidates, contract SS3. Declared before the run. ----
CANDIDATES = [
    ("T1 MFCC-13", "mfcc13", ["--target", "mfcc", "--n_mels", "64", "--n_mfcc", "13"]),
    ("T2 mel-64", "mel64", ["--target", "mel", "--n_mels", "64"]),
]
# ---- contract SS6: run ONLY if T1 misses the bar. Exploratory by construction. ----
SECONDARY = ("S1 MFCC-13+c0 (EXPLORATORY)", "mfcc13c0",
             ["--target", "mfcc_c0", "--n_mels", "64", "--n_mfcc", "13"])

OVO_ARGS = ["--train_on", "raw_solos", "--own_vs_other", "--eeg_clean", "notch_ica",
            "--estimator", "ridge", "--filters", "pooled", "--target_fs", "64",
            "--band_low", "1", "--band_high", "8", "--lags_ms", "250", "--seed", "42"]
CTRL_ARGS = ["--self_test", "--estimator", "ridge", "--target_fs", "64",
             "--band_low", "1", "--band_high", "8", "--lags_ms", "250", "--seed", "42"]

CAVEAT = [
    "!! READ BEFORE THE NUMBERS (Comandamenti #9).",
    "!! WHAT THIS MEASURES: own-vs-other measures TRACKING and nothing else. Given the EEG",
    "!! of a held-out SOLO segment, is that segment's own audio scored above ANOTHER",
    "!! solo's? The other segment is a different song, so nothing here says anything about",
    "!! separating the two stems of a duo, and nothing here is an attention decision.",
    "!! NULL = 0.500, EXACT BY SYMMETRY, not estimated: every pair is scored in both",
    "!! directions, so a decoder with nothing but a fixed instrument preference gets",
    "!! exactly one of the two right. The null is printed next to every accuracy.",
    "!! NOTHING IS SPENT: the solos are declared free training material by the vault's",
    "!! ledger of looks. No duo decision, no trio, no split by genre/melody/instrument/",
    "!! subject. The positive control is the Exp. 8 construction: it enumerates the duo",
    "!! stimuli to build its trial list, but every EEG is REPLACED by a synthetic mixture,",
    "!! so no real-EEG number is produced by it and none can be cited from it.",
    "!! THE POSITIVE-CONTROL NUMBERS ARE NOT RESULTS. They license the wiring and say",
    "!! nothing about the real data.",
    "!! K = 2, the bar (>= 208/376) and the four readings of the outcome were written in",
    "!! the vault contract of 12/8/2026 BEFORE this file existed, and are not editable now.",
    "!! McNemar against the mel-8 reference is DESCRIPTIVE, in both directions, and does",
    "!! NOT enter the verdict.",
]


def exact_p(k, n):
    """One-sided exact binomial p against p0 = 0.5 -- P(X >= k)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def run_recon(madeeg_dir, tag, extra):
    """One madeeg_reconstruction.py run into its own directory (Comandamenti #9: a control
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
    # --self_test writes to <training_date>_selftest, never over a real run's directory
    # (Comandamenti #9): a control must not be readable as a result by accident.
    txt = open(os.path.join(d + "_selftest", "madeeg_selftest_summary.txt")).read()
    m = re.search(r"AAD accuracy=([0-9.]+)", txt)
    assert m, f"no accuracy line in {d}"
    return float(m.group(1))


def detail(d):
    """The two lines of a run summary that carry the CAUSE rather than the headline: the
    per-subject spread (is a net gain one subject or many?) and the mean own/other
    similarity (is the accuracy tracking the correlation SCALE?). Read from the file."""
    txt = open(os.path.join(d, "madeeg_ownvsother_summary.txt")).read()
    return [l for l in txt.splitlines() if l.startswith(("per-subject:", "mean band_pearson"))]


def mcnemar(a_csv, b_csv):
    r = subprocess.run([PY, DIAGNOSE, "--mcnemar", a_csv, b_csv], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mcnemar failed: {r.stderr[-2000:]}")
    return r.stdout.rstrip("\n")


def check_repr_identity(madeeg_dir):
    """G2. The new --target mfcc branch must BE the Exp. 13 candidate, not a cousin of it:
    same construction, checked on a real stimulus wav. Audio only, no EEG opened."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import librosa
    from madeeg_reconstruction import source_repr
    import madeeg_stem_separability as sep
    wavs = sorted(glob.glob(os.path.join(madeeg_dir, "stimuli", "*.wav")))
    assert wavs, f"no stimuli wavs under {madeeg_dir}/stimuli"
    x, sr = librosa.load(wavs[0], sr=None, mono=True)
    x = np.asarray(x, dtype=np.float64)
    out_len = int(round(x.shape[0] / sr * 64.0))
    mine = source_repr(x, sr, out_len, 64.0, (1.0, 8.0), 0.3, 64, "mfcc", 13)
    theirs = sep._post(sep._raw_mfcc(x, sr, 13), out_len)
    assert mine.shape == theirs.shape == (13, out_len), (mine.shape, theirs.shape)
    return os.path.basename(wavs[0]), float(np.max(np.abs(mine - theirs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out", default="", help="provenance file (default: "
                                              "docs/provenance/2026-08-12_exp14_mfcc_tracking.txt)")
    args = ap.parse_args()
    md = os.path.expanduser(args.madeeg_dir)
    out_path = args.out or os.path.join(ROOT, "docs", "provenance",
                                        "2026-08-12_exp14_mfcc_tracking.txt")
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
    say("=== EXP. 14 -- own-vs-other tracking of the two gate representations ===")
    say(f"interpreter {PY} · protocol: ridge, pooled, notch+ICA, 1-8 Hz, 64 Hz, lags 0-250 ms,")
    say("seed 42, 5 folds -- every one of them identical to the mel-8 reference. Single variable:")
    say("the stimulus representation.")

    # ---------------- G1: regression canary, crossed FIRST ----------------
    say()
    say("=== GATE 1 -- REGRESSION CANARY (crossed before anything else) ===")
    d = run_recon(md, "exp14_gate1_canary_mel8", OVO_ARGS + ["--target", "mel", "--n_mels", "8"])
    k, n = ovo_result(d)
    h = md5(os.path.join(d, "madeeg_ownvsother.csv"))
    ok = (k, n) == (CANARY_K, CANARY_N) and h == CANARY_MD5
    say(f"  --target mel --n_mels 8 -> {k}/{n} (expected {CANARY_K}/{CANARY_N})   null 0.500")
    say(f"  md5(madeeg_ownvsother.csv) = {h}")
    say(f"                    expected = {CANARY_MD5}")
    say(f"  -> [{'PASSED' if ok else 'FAILED'}] the default path is bit-for-bit what it was "
        "before the diff.")
    if not ok:
        say("  CANARY FAILED -> the diff moved the science. No candidate is run (Comandamenti #3/#8).")
        dump()

    # ---------------- G2: the new branch IS the Exp. 13 candidate ----------------
    say()
    say("=== GATE 2 -- REPRESENTATION IDENTITY with the Exp. 13 candidate (audio only) ===")
    wav, diff = check_repr_identity(md)
    say(f"  source_repr(--target mfcc --n_mels 64 --n_mfcc 13) vs Exp. 13 C3 "
        f"(_raw_mfcc + _post) on {wav}")
    say(f"  max abs difference = {diff:.3e}  -> [{'PASSED' if diff == 0.0 else 'FAILED'}] "
        "(must be 0 to floating point)")
    if diff != 0.0:
        say("  The tracked object is not the separated object -> STOP.")
        dump()

    # ---------------- G3: positive control per candidate ----------------
    say()
    say(f"=== GATE 3 -- POSITIVE CONTROL per candidate, threshold >= {CTRL_MIN:.2f} "
        "(contract SS2) ===")
    say("  Synthetic EEG built FROM each candidate representation (Exp. 8 construction). A PASS")
    say("  licenses the wiring of that candidate and says NOTHING about the real data.")
    for name, tag, flags in CANDIDATES:
        d = run_recon(md, f"exp14_gate3_ctrl_{tag}", CTRL_ARGS + flags)
        acc = selftest_acc(d)
        ok = acc >= CTRL_MIN
        say(f"  {name:12s} AAD accuracy on synthetic EEG = {acc:.4f}   chance 0.500   "
            f"-> [{'PASSED' if ok else 'FAILED'}]")
        if not ok:
            say(f"  positive control below {CTRL_MIN:.2f} -> the candidate is not reported "
                "(Comandamenti #3).")
            dump()
    say("  Both wirings licensed. The real numbers may now be read.")

    # ---------------- the real measurement ----------------
    say()
    say("=== THE MEASUREMENT -- own-vs-other on held-out SOLO segments ===")
    say(f"  primary bar, written before the code: >= {BAR_K}/{BAR_N}, one-sided exact binomial")
    say(f"  P(X >= {BAR_K} | {BAR_N}, 0.5) = {exact_p(BAR_K, BAR_N):.4f} <= 0.025 = 0.05/2 "
        "(Bonferroni, K = 2).")
    say(f"  The bar is also the mel-8 reference itself ({CANARY_K}/{CANARY_N} = "
        f"{CANARY_K / CANARY_N:.4f}), re-measured by Gate 1 today.")
    say()
    ref_csv = os.path.join(ROOT, "runs", "results", "exp14_gate1_canary_mel8",
                           "madeeg_ownvsother.csv")
    res, dirs = {}, {}
    for name, tag, flags in CANDIDATES:
        d = run_recon(md, f"exp14_{tag}_ovo", OVO_ARGS + flags)
        k, n = ovo_result(d)
        res[name] = (k, n, os.path.join(d, "madeeg_ownvsother.csv"))
        dirs[name] = d
        say(f"  {name:12s} {k}/{n} = {k / n:.4f}   null 0.500   one-sided exact binomial "
            f"p = {exact_p(k, n):.4g}   bar {BAR_K}/{BAR_N}   "
            f"-> {'TRACKS' if k >= BAR_K else 'does NOT track'}")
    say(f"  {'mel-8 (ref)':12s} {CANARY_K}/{CANARY_N} = {CANARY_K / CANARY_N:.4f}   null 0.500   "
        f"one-sided exact binomial p = {exact_p(CANARY_K, CANARY_N):.4g}   (the bar itself)")

    say()
    say("  -- the two lines that carry the CAUSE, per run (null 0.500 applies to each) --")
    for name in [n for n, _, _ in CANDIDATES] + ["mel-8 (ref)"]:
        d = dirs.get(name, os.path.join(ROOT, "runs", "results", "exp14_gate1_canary_mel8"))
        say(f"  {name}:")
        for line in detail(d):
            say(f"    {line}")
    say("  Read them for two questions the headline cannot answer: whether a net gain is one")
    say("  subject or many, and whether the accuracy follows the SIZE of the reconstruction")
    say("  correlations. The decision is an argmax, so it is invariant to a uniform rescaling")
    say("  of those correlations -- which is exactly the contraction Exp. 13 flagged as the")
    say("  unresolved part of its own verdict.")

    # ---------------- descriptive McNemar, both directions ----------------
    say()
    say("=== DESCRIPTIVE -- McNemar against the mel-8 reference, BOTH directions ===")
    say("  Does not enter the verdict (contract SS4). Both directions are printed so that a")
    say("  candidate that is WORSE than the reference is as readable as one that is better.")
    for name, (_, _, csv_path) in res.items():
        say(f"\n  -- {name} vs mel-8 --")
        say(mcnemar(ref_csv, csv_path))          # B = candidate: 'does the candidate beat mel-8'
        say(mcnemar(csv_path, ref_csv))          # B = mel-8:     'does mel-8 beat the candidate'

    # ---------------- verdict, pre-declared ----------------
    t1 = res["T1 MFCC-13"][0] >= BAR_K
    t2 = res["T2 mel-64"][0] >= BAR_K
    say()
    say("=== VERDICT -- the reading was written before the numbers (contract SS5) ===")
    if t1 and t2:
        say("  BRANCH 1 (both track): arm C's front end is named -- MFCC-13, maximum separability")
        say("  AND tracking. Any EEG use beyond own-vs-other stays behind a NEW pre-registration.")
    elif t2 and not t1:
        say("  BRANCH 2 (only T2 tracks): THE ENERGY IS NECESSARY TO TRACKING -- MFCC's")
        say("  separability lives in a component the brain does not follow. The fixed target of")
        say("  arm C becomes mel-64, with a modest separability gain and a trade-off now measured")
        say("  on BOTH axes.")
    elif t1 and not t2:
        say("  BRANCH 3 (only T1 tracks): unexpected. Reported as it stands, with no post-hoc")
        say("  explanation beyond one line declared as a hypothesis.")
    else:
        say("  BRANCH 4 (neither tracks): the strong form of the thesis is demonstrated -- the")
        say("  separability gains live entirely in non-trackable components. No PASSIVE fixed")
        say("  front end can work, and arm C must LEARN the representation instead of receiving")
        say("  it. The 18.4 GiB download stays unjustified.")

    # ---------------- conditional secondary, contract SS6 ----------------
    say()
    say("=== CONDITIONAL SECONDARY (contract SS6) ===")
    if t1:
        say("  NOT RUN: it is defined only if T1 misses the bar, and T1 cleared it. No other")
        say("  variant is run, in any branch.")
    else:
        say("  T1 missed the bar -> the ONE declared diagnostic runs: the same MFCC WITH")
        say("  coefficient 0 (the log energy) put back, coefficients 0..13. It answers")
        say("  'is it the removed energy that costs the tracking?'.")
        say("  🔴 EXPLORATORY BY CONSTRUCTION: outside the Bonferroni family, not a test, and")
        say("  it cannot promote any candidate. Its own positive control is crossed first")
        say("  (Comandamenti #3 has no exemption for exploratory numbers).")
        name, tag, flags = SECONDARY
        d = run_recon(md, f"exp14_ctrl_{tag}", CTRL_ARGS + flags)
        acc = selftest_acc(d)
        say(f"  positive control {name}: {acc:.4f}   chance 0.500   threshold >= {CTRL_MIN:.2f}"
            f"   -> [{'PASSED' if acc >= CTRL_MIN else 'FAILED'}]")
        if acc < CTRL_MIN:
            say("  control failed -> the exploratory number is NOT produced.")
            dump()
        d = run_recon(md, f"exp14_{tag}_ovo", OVO_ARGS + flags)
        k, n = ovo_result(d)
        say(f"  {name}: {k}/{n} = {k / n:.4f}   null 0.500   one-sided exact binomial "
            f"p = {exact_p(k, n):.4g}")
        say(f"  descriptive vs T1 MFCC-13 (c0 dropped), both directions:")
        say(mcnemar(res["T1 MFCC-13"][2], os.path.join(d, "madeeg_ownvsother.csv")))
        say(mcnemar(os.path.join(d, "madeeg_ownvsother.csv"), res["T1 MFCC-13"][2]))
        say("  Read as a mechanism probe only: a rise here is CONSISTENT with 'the energy is")
        say("  what the EEG follows', it does not establish it, and no number in this section")
        say("  may be quoted as a test.")

    dump(code=0)


if __name__ == "__main__":
    main()
