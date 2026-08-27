"""EXP. 15 (EXPLORATORY) -- the STANDARDIZED attentional differential under MFCC-13.

Pre-registration, written BEFORE this file existed and not touched after:
  "Chapter 2 -- Exp. 15 (EXPLORATORY): the attentional differential under MFCC,
   pre-registered criterion (2026-08-12)".

THE QUESTION, AND WHY IT IS NOT ACCURACY. Exp. 13 promoted MFCC-13 on separability (the two
stems of a duo share ~4x less under it) and Exp. 14 showed it tracks the EEG at least as
well as mel-8. Does that raise the ATTENTION decision on duos? The contract's section 0 answered
"can this test be won?" BEFORE the design was fixed, with a generative model built from our
own measurements: the duo decision is an argmax of corr(s_hat, s_att) against
corr(s_hat, s_unatt), so its margin is ~ a*(1-rho) with rho the between-stem correlation
(Exp. 13: mel-8 0.1772, MFCC-13 0.0445) and a the reconstruction amplitude (Exp. 14: 0.0360
vs 0.0165). That gives a +16.1% relative margin and an expected duo accuracy of 87.4/154
against mel-8's 86/154 -- +1.4 trials, where the Exp. 9 bar of 90/154 needs +4. Power of the
binary test at that bar: 0.370. So a pre-registered accuracy primary would have been a
pre-registered failure, readable afterwards as evidence against the representation when it
is only evidence about n.

=> The primary changes QUANTITY, not threshold (the Exp. 9 bar is NOT lowered, method rules 5/6). It becomes the continuous quantity that governs the accuracy:

    d_i(R) = corr(s_hat_i, s_att,i) - corr(s_hat_i, s_unatt,i)      raw differential
    D(R)   = mean_i d_i(R) / sd_i d_i(R)                            STANDARDIZED

THE STANDARDIZATION IS NOT COSMETIC. It is the correction of the defect Exp. 13 left open:
the MFCC correlations are ~46% of the mel ones, so comparing RAW differentials would repeat
exactly the scale-invariance failure of that gate. D is dimensionless, and the accuracy is
Phi(D) if d is approximately normal -- which is why Phi(D) is printed next to the observed
accuracy as a mandatory coherence check (contract section 1: diverge by more than ~3 points and the
section 0 model is wrong and MUST BE SAID, not worked around).

PRIMARY TEST, contract section 1: exact paired sign-flip permutation on the STANDARDIZED paired
differences over the SAME 154 trials, one-sided (MFCC > mel-8), alpha = 0.05, K = 1.
Null = 0 by construction (one quantity, two representations, same trials), printed next to
the statistic. K = 1 means ONE test: it is run on the k-fold duo protocol (P2), the protocol
whose 86/154 the section 0 model is built on. D on the paper protocol (P1) is printed as a
DESCRIPTIVE statistic and is not tested -- printing it does not spend a test, running a
second permutation would.

MANDATORY FLOOR, contract section 2: the minimum detectable difference in D at this n with power
0.80, from the OBSERVED variance of the paired differences. A negative without its floor is
not reportable here, because section 0 has already shown n is the binding constraint.

GATES, DECLARED HERE AND CROSSED BEFORE ANY REAL NUMBER IS READ (contract section 3):
  G1 MEL ANCHORS DECISION-IDENTICAL to Exp. 9's: canary_rawsolos 74/154 and canary_duo
     86/154, identical TRIAL BY TRIAL (pred and correct, keyed subject+stim), not merely in
     total. These re-runs are also the mel-8 arm of the primary, so the gate and the
     measurement are the same object: if the wiring moved, the comparison is meaningless.
  G1b (an ADDITION of this file, not a relaxation): the re-run correlations themselves are
     compared to the archived anchors and the max abs deviation is PRINTED. G1 as the
     contract writes it pins the decisions; the primary is computed on the correlations
     BEHIND those decisions, so their agreement is worth a number rather than an assumption.
     Reported, not a hard exit: the hard exit is on the decisions, as pre-registered.
  G2 POSITIVE CONTROL for the MFCC path on the DUO DECISION (Exp. 8/14 construction: every
     EEG replaced by a synthetic linear mixture of that trial's attended representation at
     the model lags + noise), threshold >= 0.95 as the contract raises it -- the file's own
     built-in PASS line still says 0.90, and a candidate between the two counts as FAILED.
Any gate failing => this script writes what it has and exits non-zero.

WHAT IS SPENT: the duos, which the held-out-look ledger declares ALREADY ALL SPENT. Hence
EXPLORATORY BY CONSTRUCTION, in the title and inside the provenance file. NO TRIO is touched:
they are the only untouched holdout, ONE shot, and even a GREEN primary here does not open
them -- that needs a new pre-registration (contract section 6).

Run (CPU, interpreter is NOT optional -- environment trap 1):
  /opt/miniconda3/bin/python src/madeeg_exp15_differential.py --madeeg_dir ~/madeeg
"""
import os
import re
import csv
import sys
import argparse
import subprocess

import numpy as np
from scipy import stats

PY = "/opt/miniconda3/bin/python"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON = os.path.join(ROOT, "src", "madeeg_reconstruction.py")
DIAGNOSE = os.path.join(ROOT, "src", "madeeg_diagnose.py")
RES = os.path.join(ROOT, "runs", "results")

# ---- thresholds and constants, contract section 0-section 3. Not editable after the fact. ----
ANCHORS = {"P1": ("canary_rawsolos", 74, 154), "P2": ("canary_duo", 86, 154)}
CTRL_MIN = 0.95              # contract section 3.2; the file's own built-in line still says 0.90
ALPHA = 0.05                 # contract section 1, one-sided, K = 1
POWER = 0.80                 # contract section 2
N_PERM = 10000               # contract section 1
PERM_SEED = 20260812         # declared here, before the run
COHERENCE_PTS = 3.0          # contract section 1: Phi(D) vs accuracy, in accuracy POINTS
EXP9_BAR = 90                # Exp. 9's absolute bar on 154. NOT LOWERED, NOT MOVED.
SEC0_PRED_ACC = 87.4          # contract section 0's predicted MFCC duo accuracy out of 154
SEC0_PRED_GAIN = 0.161        # contract section 0's predicted relative margin gain, +16.1%

# The decision protocol: every flag identical to the Exp. 9 anchors. Single variable = the
# stimulus representation.
DEC_ARGS = ["--ensemble", "duo", "--estimator", "ridge", "--filters", "pooled",
            "--eeg_clean", "none", "--target_fs", "64", "--band_low", "1", "--band_high", "8",
            "--lags_ms", "250", "--cv_folds", "5", "--seed", "42", "--spatial", "stereo"]
PROTOCOL = {                                     # contract section 4: both Exp. 9 protocols
    "P1": ["--train_on", "raw_solos", "--test_eeg", "raw"],
    "P2": ["--train_on", "duos_kfold", "--test_eeg", "preprocessed"],
}
MEL8 = ["--target", "mel", "--n_mels", "8"]
MFCC13 = ["--target", "mfcc", "--n_mels", "64", "--n_mfcc", "13"]

CAVEAT = [
    "!! READ BEFORE THE NUMBERS (method rule 9).",
    "!! EXPLORATORY BY CONSTRUCTION, AND SAID SO IN THE TITLE. This runs on the DUOS, which",
    "!! the held-out-look ledger declares ALREADY ALL SPENT. No number here is",
    "!! confirmatory, whatever threshold it clears. NO TRIO IS TOUCHED: 92 stereo + 93 mono",
    "!! are the only untouched holdout, ONE shot, and even a GREEN primary here does not open",
    "!! them -- that requires a NEW pre-registration, read before the run (section 6).",
    "!! THE PRIMARY IS THE STANDARDIZED DIFFERENTIAL D = mean(d)/sd(d), NOT THE ACCURACY.",
    "!! Why: the contract's section 0 check ('can this test be won?') was done BEFORE the design,",
    "!! with our own measurements, and predicted +1.4 trials where the Exp. 9 bar needs +4",
    "!! (power 0.370). A pre-registered accuracy primary would have been a pre-registered",
    "!! failure. The bar was NOT lowered; the QUANTITY changed. And the standardization is",
    "!! mandatory: MFCC correlations are ~46% of the mel ones, so comparing RAW differentials",
    "!! would repeat exactly the scale-invariance defect Exp. 13 left open.",
    "!! NULL = 0, EXACT BY CONSTRUCTION, not estimated: the same quantity under two",
    "!! representations on the same trials has expected difference 0 under the null. It is",
    "!! printed next to the statistic.",
    "!! ACCURACY IS SECONDARY AND DESCRIPTIVE. It is reported next to section 0's prediction",
    "!! (87.4/154) and next to the Exp. 9 bar (90/154). MISSING THAT BAR IS THE PREDICTED",
    "!! OUTCOME, NOT A SURPRISE -- and no threshold anywhere has been lowered to meet it.",
    "!! THE POSITIVE-CONTROL NUMBER IS NOT A RESULT. It licenses the wiring of the MFCC path",
    "!! on the duo decision and says NOTHING about the real data: every EEG in it is",
    "!! synthetic. It may never be cited as a result.",
    "!! K = 1. One permutation test, on the k-fold duo protocol (P2), the one section 0's model is",
    "!! built on. D under the paper protocol (P1) is printed as a DESCRIPTIVE statistic and",
    "!! is not tested.",
]


# ----------------------------------------------------------------------------------
def run_recon(madeeg_dir, tag, extra):
    """One madeeg_reconstruction.py run into its own directory (method rule 9: a control
    never writes over a result)."""
    cmd = [PY, RECON, "--madeeg_dir", madeeg_dir, "--training_date", tag] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        sys.exit(f"run {tag} failed with code {r.returncode} -- STOP")
    return os.path.join(RES, tag)


def records(d):
    """The per-trial decision records, keyed subject+stim -- the key `madeeg_diagnose
    --mcnemar` already uses to pair two attention runs."""
    path = os.path.join(d, "madeeg_records.csv") if os.path.isdir(d) else d
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["subject"], r["stim"])] = dict(
            pred=int(r["pred"]), correct=int(r["correct"]),
            r_att=float(r["r_attended"]), r_un=float(r["r_best_unattended"]),
            n_present=int(r["n_present"]))
    return out


def selftest_acc(d):
    # --self_test writes to <training_date>_selftest, never over a real run's directory.
    txt = open(os.path.join(d + "_selftest", "madeeg_selftest_summary.txt")).read()
    m = re.search(r"AAD accuracy=([0-9.]+)", txt)
    assert m, f"no accuracy line under {d}_selftest"
    return float(m.group(1))


def differential(rec, keys):
    """d_i = corr(s_hat_i, s_att,i) - corr(s_hat_i, s_unatt,i), in the trial order `keys`.

    On a duo `r_best_unattended` IS the single competitor (n_present == 2, asserted), so this
    is the contract's d_i exactly and not a max over several distractors."""
    for k in keys:
        assert rec[k]["n_present"] == 2, f"{k}: n_present={rec[k]['n_present']}, not a duo"
    return np.array([rec[k]["r_att"] - rec[k]["r_un"] for k in keys], dtype=np.float64)


def mcnemar(a_csv, b_csv):
    r = subprocess.run([PY, DIAGNOSE, "--mcnemar", a_csv, b_csv], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mcnemar failed: {r.stderr[-2000:]}")
    return r.stdout.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    md = os.path.expanduser(args.madeeg_dir)
    out_path = args.out or os.path.join(ROOT, "docs", "provenance",
                                        "2026-08-12_exp15_mfcc_differential.txt")
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
    say("=== EXP. 15 (EXPLORATORY) -- the standardized attentional differential under MFCC-13 ===")
    say(f"interpreter {PY} · protocol: ridge, pooled, eeg_clean none, 1-8 Hz, 64 Hz,")
    say("lags 0-250 ms, seed 42, 5 folds, stereo -- every one of them identical to the Exp. 9")
    say("mel anchors. Single variable: the stimulus representation.")
    say("  mel-8    : --target mel --n_mels 8")
    say("  MFCC-13  : --target mfcc --n_mels 64 --n_mfcc 13   (DCT-II ortho of log-mel-64,")
    say("             coefficients 1..13, c0 dropped -- the Exp. 13 C3 / Exp. 14 T1 object)")

    # ---------------- G1: the mel anchors, decision-identical, crossed FIRST -------------
    say()
    say("=== GATE 1 -- MEL ANCHORS DECISION-IDENTICAL to Exp. 9 (crossed before anything else) ===")
    say("  Not a total-count check: pred and correct are compared TRIAL BY TRIAL, keyed")
    say("  subject+stim. These re-runs are also the mel-8 arm of the primary.")
    mel_dir, mel_rec, keys = {}, {}, {}
    for p in ("P1", "P2"):
        anchor_tag, exp_k, exp_n = ANCHORS[p]
        d = run_recon(md, f"exp15_gate1_mel8_{p.lower()}", PROTOCOL[p] + DEC_ARGS + MEL8)
        new, old = records(d), records(os.path.join(RES, anchor_tag))
        common = sorted(set(new) & set(old))
        k = sum(new[x]["correct"] for x in common)
        disc = [x for x in common if new[x]["pred"] != old[x]["pred"]
                or new[x]["correct"] != old[x]["correct"]]
        rmax = max(max(abs(new[x]["r_att"] - old[x]["r_att"]),
                       abs(new[x]["r_un"] - old[x]["r_un"])) for x in common) if common else 1.0
        ok = (len(new) == len(old) == len(common) == exp_n and k == exp_k and not disc)
        say(f"  {p} vs runs/results/{anchor_tag}/: n_new={len(new)} n_old={len(old)} "
            f"in common={len(common)} (expected {exp_n})")
        say(f"     {k}/{len(common)} correct (expected {exp_k}/{exp_n})   null 0.500   "
            f"discordant decisions = {len(disc)} (must be 0)")
        say(f"     [G1b, addition] max abs deviation of the correlations behind those "
            f"decisions = {rmax:.3e}")
        say(f"     -> [{'PASSED' if ok else 'FAILED'}]")
        if not ok:
            say("  ANCHOR MOVED -> the wiring is not the one Exp. 9 measured. No MFCC run "
                "(method rules 3/8).")
            dump()
        mel_dir[p], mel_rec[p], keys[p] = d, new, common
    say("  Both anchors are decision-identical, trial by trial. The wiring is Exp. 9's.")

    # ---------------- G2: positive control for the MFCC path on the duo decision --------
    say()
    say(f"=== GATE 2 -- POSITIVE CONTROL, MFCC path on the DUO DECISION, threshold >= "
        f"{CTRL_MIN:.2f} ===")
    say("  Exp. 8/14 construction: every EEG replaced by a synthetic linear mixture of that")
    say("  trial's attended MFCC-13 representation at the model's lags + noise, then the REAL")
    say("  pipeline decides. A PASS licenses the wiring and says NOTHING about the real data.")
    d = run_recon(md, "exp15_gate2_ctrl_mfcc13",
                  ["--self_test"] + PROTOCOL["P2"] + DEC_ARGS + MFCC13)
    acc = selftest_acc(d)
    say(f"  MFCC-13 AAD accuracy on synthetic EEG = {acc:.4f}   chance 0.500   "
        f"threshold >= {CTRL_MIN:.2f}   -> [{'PASSED' if acc >= CTRL_MIN else 'FAILED'}]")
    if acc < CTRL_MIN:
        say("  positive control below the threshold -> no real number is produced "
            "(method rule 3).")
        dump()
    say("  The wiring is licensed. The real numbers may now be read.")

    # ---------------- the real runs ----------------------------------------------------
    say()
    say("=== THE MEASUREMENT -- MFCC-13 on the two Exp. 9 duo protocols ===")
    mf_dir, mf_rec = {}, {}
    for p in ("P1", "P2"):
        d = run_recon(md, f"exp15_mfcc13_{p.lower()}", PROTOCOL[p] + DEC_ARGS + MFCC13)
        mf_dir[p], mf_rec[p] = d, records(d)
        assert set(mf_rec[p]) >= set(keys[p]), f"{p}: MFCC run does not cover the anchor trials"

    # ---------------- PRIMARY: the standardized differential ---------------------------
    say()
    say("=== PRIMARY (K = 1) -- the STANDARDIZED attentional differential, P2 (k-fold duo) ===")
    say("  d_i = corr(s_hat_i, s_att,i) - corr(s_hat_i, s_unatt,i);  D = mean_i d_i / sd_i d_i")
    say("  sd is the sample sd (ddof = 1). On a duo the 'best unattended' is the ONE")
    say("  competitor (n_present == 2, asserted per trial), so d_i is the contract's d_i.")
    say()
    stat = {}
    for p in ("P1", "P2"):
        row = {}
        for name, rec in (("mel-8", mel_rec[p]), ("MFCC-13", mf_rec[p])):
            d_i = differential(rec, keys[p])
            row[name] = dict(d=d_i, mean=float(d_i.mean()), sd=float(d_i.std(ddof=1)),
                             D=float(d_i.mean() / d_i.std(ddof=1)),
                             k=int(sum(rec[x]["correct"] for x in keys[p])), n=len(keys[p]))
        stat[p] = row

    n = len(keys["P2"])
    z_mel = stat["P2"]["mel-8"]["d"] / stat["P2"]["mel-8"]["sd"]
    z_mfc = stat["P2"]["MFCC-13"]["d"] / stat["P2"]["MFCC-13"]["sd"]
    delta = z_mfc - z_mel                        # mean(delta) == D(MFCC) - D(mel), exactly
    t_obs = float(delta.mean())
    # The one invariant that fails loudly if the standardization is wired wrong: the mean of
    # the paired standardized differences IS the difference of the two D's. If this breaks,
    # the test is not testing the pre-registered quantity.
    assert abs(t_obs - (stat["P2"]["MFCC-13"]["D"] - stat["P2"]["mel-8"]["D"])) < 1e-12, \
        "mean(z_MFCC - z_mel) != D(MFCC) - D(mel): the standardization is miswired"
    # Exact sign-flip permutation. 2**154 = 2.3e46 sign patterns: enumeration is not
    # feasible, so it is the Monte Carlo version the contract allows, with the seed declared
    # in this file before the run. The standardization constants (the two sd's) are held
    # FIXED at their observed values, so the permuted statistic is a pure sign-flip of the
    # paired differences and the null is exactly 0.
    rng = np.random.default_rng(PERM_SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_PERM, n))
    t_perm = (signs * delta).mean(axis=1)
    ge = int(np.sum(t_perm >= t_obs))
    p_perm = (1 + ge) / (N_PERM + 1)
    mc_se = float(np.sqrt(p_perm * (1 - p_perm) / (N_PERM + 1)))

    say("  -- raw differentials, for the record and NOT for the comparison (the reason the")
    say("     contract standardizes: these are not on a common scale) --")
    for p in ("P1", "P2"):
        for name in ("mel-8", "MFCC-13"):
            s = stat[p][name]
            say(f"     {p} {name:8s} mean d = {s['mean']:+.6f}   sd d = {s['sd']:.6f}")
    say()
    say("  -- D, standardized, dimensionless (null for the DIFFERENCE = 0 by construction) --")
    for p in ("P1", "P2"):
        tag = "PRIMARY" if p == "P2" else "descriptive, not tested"
        say(f"     {p} ({tag}):")
        for name in ("mel-8", "MFCC-13"):
            s = stat[p][name]
            phi = float(stats.norm.cdf(s["D"]))
            say(f"       D({name:8s}) = {s['D']:+.4f}   Phi(D) = {phi:.4f} = "
                f"{phi * s['n']:.1f}/{s['n']}   observed accuracy = {s['k']}/{s['n']} = "
                f"{s['k'] / s['n']:.4f}   |diff| = {abs(phi - s['k'] / s['n']) * 100:.2f} pts")
        say(f"       D(MFCC-13) - D(mel-8) = {stat[p]['MFCC-13']['D'] - stat[p]['mel-8']['D']:+.4f}"
            f"   null 0 by construction")
    say()
    say(f"  PRIMARY TEST, P2, one-sided (MFCC-13 > mel-8), alpha = {ALPHA}, K = 1:")
    say(f"     paired sign-flip permutation on the STANDARDIZED paired differences,")
    say(f"     n = {n} trials, B = {N_PERM} sign patterns, seed {PERM_SEED} "
        f"(2**{n} = {2.0 ** n:.1e} patterns: enumeration infeasible, Monte Carlo as the")
    say(f"     contract allows). Statistic = mean(z_MFCC - z_mel) = D(MFCC) - D(mel).")
    say(f"     observed statistic = {t_obs:+.4f}   NULL = 0 (exact, by construction)")
    say(f"     permutation mean = {t_perm.mean():+.6f}  sd = {t_perm.std(ddof=1):.6f}  "
        f"(the null distribution is centred on 0, as it must be)")
    say(f"     p = ({ge} + 1)/({N_PERM} + 1) = {p_perm:.4f}   (MC standard error {mc_se:.4f})")
    say(f"     -> {'REJECTS' if p_perm <= ALPHA else 'does NOT reject'} the null at "
        f"alpha = {ALPHA}")

    # ---------------- MANDATORY FLOOR, contract section 2 ------------------------------------
    say()
    say("=== THE DETECTABILITY FLOOR -- mandatory, contract section 2 ===")
    sd_delta = float(delta.std(ddof=1))
    zsum = float(stats.norm.ppf(1 - ALPHA) + stats.norm.ppf(POWER))
    mdd = zsum * sd_delta / np.sqrt(n)
    pred_dD_mult = SEC0_PRED_GAIN * stat["P2"]["mel-8"]["D"]
    pred_dD_acc = float(stats.norm.ppf(SEC0_PRED_ACC / 154.0)
                        - stats.norm.ppf(ANCHORS["P2"][1] / 154.0))
    obs_power = float(stats.norm.sf(stats.norm.ppf(1 - ALPHA)
                                    - pred_dD_mult * np.sqrt(n) / sd_delta))
    say(f"  sd of the paired standardized differences = {sd_delta:.4f}  (observed, n = {n};")
    say(f"  corr(z_mel, z_MFCC) = {float(np.corrcoef(z_mel, z_mfc)[0, 1]):.4f} -- the pairing")
    say(f"  is what buys the power, so it is printed rather than assumed)")
    say(f"  MINIMUM DETECTABLE DIFFERENCE in D at n = {n}, power {POWER}, one-sided "
        f"alpha = {ALPHA}:")
    say(f"     MDD = (z_{1 - ALPHA:.2f} + z_{POWER:.2f}) * sd/sqrt(n) = "
        f"({stats.norm.ppf(1 - ALPHA):.4f} + {stats.norm.ppf(POWER):.4f}) * {sd_delta:.4f}"
        f"/sqrt({n}) = {mdd:.4f}")
    say(f"  THE EFFECT THE CONTRACT'S section 0 PREDICTS, in the same units (two routes, they must")
    say(f"  agree because section 0's accuracy prediction IS Phi of its margin prediction):")
    say(f"     +{SEC0_PRED_GAIN * 100:.1f}% of D(mel-8) = {SEC0_PRED_GAIN:.3f} * "
        f"{stat['P2']['mel-8']['D']:+.4f} = {pred_dD_mult:+.4f}")
    say(f"     Phi^-1({SEC0_PRED_ACC}/154) - Phi^-1({ANCHORS['P2'][1]}/154) = {pred_dD_acc:+.4f}")
    say(f"  power of THIS test against section 0's predicted effect = {obs_power:.3f}")
    say(f"  -> the floor is {'ABOVE' if mdd > pred_dD_mult else 'BELOW'} the predicted effect "
        f"({mdd:.4f} vs {pred_dD_mult:+.4f})")

    # ---------------- coherence check, contract section 1 ------------------------------------
    say()
    say(f"=== COHERENCE CHECK (contract section 1) -- Phi(D) against the observed accuracy ===")
    say(f"  If they diverge by more than ~{COHERENCE_PTS:.0f} accuracy points, the section 0 model is")
    say("  wrong and that MUST BE SAID, not worked around.")
    worst = 0.0
    for p in ("P1", "P2"):
        for name in ("mel-8", "MFCC-13"):
            s = stat[p][name]
            gap = abs(float(stats.norm.cdf(s["D"])) - s["k"] / s["n"]) * 100
            worst = max(worst, gap)
            say(f"  {p} {name:8s}: Phi(D) = {float(stats.norm.cdf(s['D'])):.4f}   observed = "
                f"{s['k'] / s['n']:.4f}   divergence = {gap:.2f} points   "
                f"[{'OK' if gap <= COHERENCE_PTS else 'MODEL WRONG -- SAID, NOT WORKED AROUND'}]")
    say(f"  worst divergence over the four cells = {worst:.2f} points   "
        f"-> [{'COHERENT' if worst <= COHERENCE_PTS else 'THE section 0 MODEL IS WRONG'}]")
    say()
    say("  -- WHAT MAKES IT DIVERGE (added after the first run, because the contract says the")
    say("     divergence must be SAID, and saying it usefully means naming its cause; no")
    say("     threshold, test or branch rule was touched -- method rule 2) --")
    say("     The accuracy IS the fraction of POSITIVE differentials: argmax(corr) == attended")
    say("     is the same event as d_i > 0 (asserted below, exactly). Phi(D) equals that")
    say("     fraction only if d is NORMAL. So every point of divergence is non-normality of")
    say("     d, and its SIGN says which way: d skewed right => the mean is pulled above the")
    say("     median => Phi(D) OVERSTATES how often d is positive, and vice versa.")
    say(f"     {'cell':14s} {'frac(d>0)':>10s} {'accuracy':>9s} {'Phi(D)':>8s} {'skew':>8s} "
        f"{'exc.kurt':>9s} {'Phi(med/sd)':>12s}")
    for p in ("P1", "P2"):
        for name in ("mel-8", "MFCC-13"):
            s = stat[p][name]
            frac = float((s["d"] > 0).mean())
            assert abs(frac - s["k"] / s["n"]) < 1e-12, (
                f"{p} {name}: fraction of positive differentials != accuracy -- the decision "
                "rule and d_i are not the same event, which would break the whole model")
            say(f"     {p + ' ' + name:14s} {frac:10.4f} {s['k'] / s['n']:9.4f} "
                f"{float(stats.norm.cdf(s['D'])):8.4f} {float(stats.skew(s['d'])):8.3f} "
                f"{float(stats.kurtosis(s['d'])):9.3f} "
                f"{float(stats.norm.cdf(np.median(s['d']) / s['sd'])):12.4f}")
    say("     Read the last column against the third: the MEDIAN-based figure is the one that")
    say("     tracks the accuracy, which is what a mean/sd summary of a non-normal d cannot do.")

    # ---------------- SECONDARY, descriptive, contract section 4 -----------------------------
    say()
    say("=== SECONDARY -- ACCURACY, DESCRIPTIVE (contract section 4) ===")
    say(f"  Reported next to section 0's prediction ({SEC0_PRED_ACC}/154) and next to the Exp. 9 bar")
    say(f"  ({EXP9_BAR}/154 = {EXP9_BAR / 154:.4f}). MISSING THAT BAR IS THE PREDICTED OUTCOME,")
    say("  NOT A SURPRISE: section 0 computed the power of the binary test at that bar as 0.370")
    say("  BEFORE any of this ran. No threshold has been lowered. Null 0.500 by construction")
    say("  on a duo (two candidates, argmax).")
    for p in ("P1", "P2"):
        say(f"  -- {p} ({'paper protocol, raw_solos' if p == 'P1' else 'k-fold duo'}) --")
        for name in ("mel-8", "MFCC-13"):
            s = stat[p][name]
            pb = float(stats.binomtest(s["k"], s["n"], 0.5, alternative="greater").pvalue)
            say(f"     {name:8s} {s['k']}/{s['n']} = {s['k'] / s['n']:.4f}   null 0.500   "
                f"one-sided exact binomial p = {pb:.4g}   bar {EXP9_BAR}/154 -> "
                f"{'CLEARED' if s['k'] >= EXP9_BAR else 'not cleared (the PREDICTED outcome)'}")
        say(f"     section 0 predicted for MFCC-13 on P2: {SEC0_PRED_ACC}/154 = "
            f"{SEC0_PRED_ACC / 154:.4f}; observed {stat[p]['MFCC-13']['k']}/154 = "
            f"{stat[p]['MFCC-13']['k'] / 154:.4f}"
            + ("   <- the protocol section 0's model is about" if p == "P2" else ""))

    say()
    say("=== SECONDARY -- McNEMAR against the mel anchors, BOTH directions (descriptive) ===")
    say("  Does not enter the verdict. Both directions are printed so that a representation")
    say("  that is WORSE than the anchor is as readable as one that is better.")
    for p in ("P1", "P2"):
        a = os.path.join(mel_dir[p], "madeeg_records.csv")
        b = os.path.join(mf_dir[p], "madeeg_records.csv")
        say(f"\n  -- {p}: MFCC-13 vs mel-8 --")
        say(mcnemar(a, b))                 # B = MFCC: does MFCC beat mel-8
        say(mcnemar(b, a))                 # B = mel-8: does mel-8 beat MFCC

    # ---------------- verdict, pre-declared, contract section 5 ------------------------------
    say()
    say("=== VERDICT -- the three readings were written before the numbers (contract section 5) ===")
    if p_perm <= ALPHA and t_obs > 0:
        say("  BRANCH 1 (GREEN): D(MFCC) > D(mel) at alpha = 0.05. The representation route")
        say("  works ON THE MECHANISM: the limit is n, not the front end. The constructive")
        say("  claim becomes 'a representation that quadruples candidate separability also")
        say("  raises the attentional differential; seeing it in ACCURACY needs ~N decisions'.")
        say("  N, from the observed effect, is printed below.")
        need = int(np.ceil((zsum * sd_delta / t_obs) ** 2)) if t_obs > 0 else -1
        say(f"     N for power {POWER} at alpha {ALPHA} on THIS effect = {need} paired trials")
        say("  THE TRIOS STILL DO NOT OPEN. Contract section 6: that needs a NEW pre-registration,")
        say("  read before the run. Nothing in this file authorises touching them.")
    elif mdd > pred_dD_mult:
        say("  BRANCH 2 (AMBER): the primary does not reject, and the floor is ABOVE the effect")
        say(f"  section 0 predicts ({mdd:.4f} vs {pred_dD_mult:+.4f}). NOT CONCLUSIVE, FOR POWER, and")
        say("  the floor says it in figures. No conclusion about the representation may be")
        say("  drawn from this: an effect of the predicted size would have been missed here.")
        need = int(np.ceil((zsum * sd_delta / pred_dD_mult) ** 2)) if pred_dD_mult > 0 else -1
        say(f"     paired trials needed to see section 0's predicted effect at power {POWER}: {need}")
        say(f"     (MAD-EEG has 154 stereo duo decisions in total, so this design is short by")
        say(f"     a factor of ~{need / n:.0f}. THIS is what 'the limit is n' means in figures.)")
    else:
        say("  BRANCH 3 (RED, the structural result): flat, and the floor is BELOW the effect")
        say(f"  section 0 predicts ({mdd:.4f} vs {pred_dD_mult:+.4f}). A representation that tracks")
        say("  as well as mel (Exp. 14) and separates 4x better (Exp. 13) does NOT raise the")
        say("  attentional differential: the failure is not in the front end. This closes the")
        say("  PASSIVE route and motivates arm C -- a model that LEARNS selectivity -- as the")
        say("  only road left.")
    if t_obs < 0 and abs(t_obs) > 0.5 * mdd:
        say()
        say("  THE PRE-DECLARED TAXONOMY HAS NO CELL FOR WHAT WAS OBSERVED, AND SAYING SO IS")
        say("  PART OF APPLYING IT. The contract's branches 2 and 3 both begin with the word")
        say("  'flat'. The observed point estimate is NOT flat: it is")
        say(f"  {t_obs:+.4f}, i.e. {abs(t_obs) / mdd:.2f}x the detectability floor, in the")
        say("  OPPOSITE direction to the hypothesis. Branch 2 is applied because that is the")
        say("  rule as written and a rule is not edited after seeing the number (method")
        say("  rule 5), and the qualifier is printed next to it because branch 2's own words")
        say("  ('no conclusion may be drawn') would otherwise hide a large negative estimate.")
        say("  WHAT IS NOT DONE HERE: the sign is NOT flipped (method rule 6). 'mel-8 beats")
        say("  MFCC-13 on attention' is NOT a result of this experiment -- the pre-registered")
        say("  test was one-sided in the other direction, and turning a below-hypothesis")
        say("  outcome into a claim by re-aiming the test is precisely the move the")
        say("  pre-registration exists to prevent. The reverse-direction McNemar above is a")
        say("  DESCRIPTIVE secondary the contract mandates in both directions; a confirmatory")
        say("  claim in that direction needs a NEW pre-registration and fresh material.")
    say()
    say("  NOT DONE, DELIBERATELY: no trio; no split by genre / melody / instrument / subject;")
    say("  no third representation; no sign of anything flipped; no threshold moved; no GPU;")
    say("  no push; no external download.")
    dump(code=0)


if __name__ == "__main__":
    main()
