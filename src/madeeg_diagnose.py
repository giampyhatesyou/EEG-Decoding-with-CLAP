# CHANGED(baseline): NEW FILE — not in the Akama et al. upstream. Post-mortem of a finished
#   MAD-EEG contrastive k-fold run: what did the model key its decision on?
"""Why step C landed BELOW chance, read off the per-trial records.

Step C scored 58/154 = 0.3766 with a 95% CI of [0.300, 0.458] — an interval that excludes
0.50. A model at chance is uninformative; a model whose CI excludes chance in a two-way
decision is using information and inverting it. This script asks the question that separates
the two, and it needs neither GPU nor HDF5: the run's records plus the dataset metadata.

The question: for each test trial, is what the model CHOSE predicted by the attended
instrument that was in the majority among the trials of the same mixture *in that trial's
own training folds*?

Why that prior is a trap rather than a shortcut. MAD-EEG presents all 18 duo mixtures with
both of their instruments as the target, so which stem is attended is not a property of the
stimulus and the prior carries no legitimate information about a held-out trial. Worse:
k-fold removes the test trial from the count, so in a near-balanced mixture the remaining
count tips towards the OTHER instrument. A model that has learned "this stimulus is usually
attended at stem X" is therefore wrong more often than right, by construction — which is
exactly what a below-chance result looks like.

A separate entrypoint on purpose: this is CPU/stdlib work, and `madeeg_contrastive.py`
imports torch at module scope. The vault's rule is that every number can be re-checked
offline, so the check must not need the training environment.

    python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
        --records runs/results/madeeg_clap_kfold/madeeg_contrastive_records.csv
"""
import argparse
import collections
import csv
import os
import re

import numpy as np
import yaml
from scipy import stats


def trial_decision(mean_sims, target_idx):
    """The decision rule, in the form that also holds when three stems are present.

    `mean_sims[s]` is stem s's similarity to the EEG, averaged over the trial's windows.
    The rule is: take the argmax; the trial is correct when the argmax is the attended
    stem. It is returned as a signed margin -- attended minus the best competitor -- so the
    records keep a magnitude and not just a bit. `margin > 0` IS `argmax == target_idx`,
    with an exact tie counted as wrong, which is the conservative reading and has measure
    zero on floats.

    On duos this is identically step C's rule: the mean over windows of
    (sim_attended - sim_competitor) is the same number as
    mean(sim_attended) - mean(sim_competitor). `--check_rule` verifies that against step C's
    archived records rather than taking it on faith.

    It lives in this file, not in `madeeg_contrastive.py`, so that the rule can be re-run
    offline on any machine: this module needs no torch and no HDF5. The training entrypoint
    imports it from here, so there is exactly one implementation.
    """
    competitor = max(v for s, v in enumerate(mean_sims) if s != target_idx)
    return mean_sims[target_idx] - competitor


def load(records_csv, madeeg_dir):
    """Records joined with the metadata that says which instruments a mixture holds."""
    meta = yaml.load(open(os.path.join(madeeg_dir, "madeeg_preprocessed.yaml")),
                     Loader=yaml.FullLoader)
    rows = list(csv.DictReader(open(records_csv)))
    for r in rows:
        m = meta.get(r["subject"], {}).get(r["stim"])
        if m is not None:
            r["instruments"] = list(m["instruments"])
            r["target"] = m["target"]
        else:
            # The mono half of the duo (`--spatial mono`) exists only in the raw release, so
            # its stim key has no entry in the preprocessed metadata at all. Everything this
            # analysis needs is in the key: `pop_mixtape_duo_BsDr_theme2_mono_Dr` -> pair
            # BsDr, target Dr. Every instrument code in the release is two characters.
            r["instruments"] = re.findall("[A-Z][a-z]", r["stim"].split("_")[3])
            r["target"] = r["stim"].rsplit("_", 1)[1]
            assert r["target"] in r["instruments"], f"cannot parse instruments from {r['stim']}"
            # The run also writes the label it used. If the key and the run disagree, the
            # join is silently wrong and every count below would be too.
            if r.get("target_instr"):
                assert r["target_instr"] == r["target"], (
                    f"{r['stim']}: key says target {r['target']}, records say {r['target_instr']}")
        # `..._duo_CoFl_theme1_stereo_Co` -> the mixture, without the attended instrument
        r["mixture"] = r["stim"].rsplit("_", 1)[0]
        r["pair"] = "".join(r["instruments"])
        r["global"] = "all"                              # constant: the whole training set
        r["fold"], r["correct"] = int(r["fold"]), int(r["correct"])
        # What the model actually picked: the target when it was right, the competitor when
        # it was not. For a duo the two are exhaustive, so this inverts losslessly.
        r["chosen"] = (r["target"] if r["correct"]
                       else next(i for i in r["instruments"] if i != r["target"]))
    return rows


def in_training_set(q, r, per_subject):
    """Was trial q in the training set of the model that decided trial r?

    Step C/D train ONE pooled model on a global k-fold, so every trial of another fold is
    in training. The ridge backward model fits a SEPARATE model per subject, so a trial of
    another subject was never seen -- counting it would dilute the prior with data the
    model provably could not use, and understate the effect. Same statistic, different
    training set; the difference is a property of the run, not of the analysis, so it is a
    flag with the pooled behaviour as the default.
    """
    if per_subject and q["subject"] != r["subject"]:
        return False
    return q["fold"] != r["fold"]


def training_counts(rows, r, key="mixture", per_subject=False):
    """How often each of this trial's two instruments is the target in its TRAINING folds."""
    counts = collections.Counter(q["target"] for q in rows
                                 if q[key] == r[key] and in_training_set(q, r, per_subject))
    a, b = r["instruments"]
    return counts[a], counts[b]


def majority(rows, r, key="mixture", per_subject=False):
    """The instrument the training folds attend more often, or None when they tie."""
    ca, cb = training_counts(rows, r, key, per_subject)
    a, b = r["instruments"]
    return None if ca == cb else (a if ca > cb else b)


def base_rate(rows, key="mixture", per_subject=False):
    """P(prior == target): how often following the prior would be right anyway.

    This, NOT 0.5, is the null for "does the model follow the prior". A model that reads the
    EEG and answers with the target agrees with the prior exactly this often -- so testing
    agreement against 0.5 would call a perfectly good decoder a prior-follower whenever the
    prior happens to be right more than half the time. It is a property of the data alone,
    no model involved. The synthetic-EEG control run lands on it at every grain, which is
    what confirms the statistic is measuring the model and not the design.
    """
    decidable = [r for r in rows if majority(rows, r, key, per_subject) is not None]
    return sum(majority(rows, r, key, per_subject) == r["target"] for r in decidable) / len(decidable)


def report(rows, per_subject=False):
    k, n = sum(r["correct"] for r in rows), len(rows)
    print(f"  {k}/{n} = {k / n:.4f} correct, chance 0.500\n")

    if all(r["fold"] < 0 for r in rows):
        # The paper's split (train on solos, test on duets) puts NO duo trial in training,
        # so there is no per-mixture target frequency to follow. That is not a limitation of
        # this script: the shortcut that steps C and D measured is unavailable by
        # construction under the published protocol, and saying so is the finding.
        print("  every trial has fold = -1: this run trained on SOLOS, so no duo label was\n"
              "  ever in the training set. Prior-following is not weakly present here, it is\n"
              "  STRUCTURALLY UNAVAILABLE -- there is no training frequency to follow. The\n"
              "  statistics below need a k-fold over the duos and are skipped.")
        return

    # 1. The effect, stated as a 2x2 with no modelling in between.
    cell = collections.Counter()
    for r in rows:
        mj = majority(rows, r, per_subject=per_subject)
        if mj is not None:
            cell[(mj == r["target"], bool(r["correct"]))] += 1
    tt, tf = cell[(True, True)], cell[(True, False)]
    ft, ff = cell[(False, True)], cell[(False, False)]
    print("  accuracy split by whether the trial's own target was the training majority")
    mixture_grain_ok = bool(tt + tf) and bool(ft + ff)
    if not mixture_grain_ok:
        # Degenerate, and the degeneracy is the finding rather than a failure of the analysis.
        # Under a PER-SUBJECT k-fold a (subject, mixture) group holds at most its two twins,
        # so leaving one out leaves a count of 1-0 pointing at the OTHER instrument -- every
        # single time. The prior at mixture grain is not merely a bad guide here, it is
        # deterministically anti-correlated with the target; and for the 60 groups seen with
        # one target only there is no other trial to count at all. Neither row can fill.
        print(f"    one cell is empty ({tt + tf} vs {ft + ff}): with per-subject folds the")
        print("    mixture-grain prior is DEGENERATE. A (subject, mixture) group holds at most")
        print("    the two twins, so leave-one-out always leaves the competitor in the majority")
        print("    -- the prior points at the wrong instrument by construction, on every trial.")
        print("    Nothing to test at this grain; the coarser grains below still apply.")
    else:
        odds, p_fisher = stats.fisher_exact([[tt, tf], [ft, ff]])
        print(f"    target IS  the majority of its mixture: {tt}/{tt + tf} = {tt / (tt + tf):.3f}")
        print(f"    target is NOT                         : {ft}/{ft + ff} = {ft / (ft + ff):.3f}")
        print(f"    Fisher exact p = {p_fisher:.2e}   odds ratio = {odds:.2f}")
        print("    chance is 0.500 in BOTH cells: every duo mixture is presented with both of")
        print("    its instruments as the target, so the prior is not legitimate information")

    # 2. The same effect from the model's side: does it choose the prior?
    agree = decidable = 0
    for r in rows:
        mj = majority(rows, r, per_subject=per_subject)
        if mj is not None:
            agree, decidable = agree + (r["chosen"] == mj), decidable + 1
    null = base_rate(rows, per_subject=per_subject) if decidable else 0.0
    if not mixture_grain_ok:
        if decidable:
            print(f"\n  the model chose the (always-wrong) majority in {agree}/{decidable} = "
                  f"{agree / decidable:.4f}, against a null P(prior == target) of exactly "
                  f"{null:.0f}.")
            print("  A deterministic prior leaves no hypothesis to test at this grain: the")
            print("  fraction is descriptive only. The coarser grains below are not degenerate.")
        _grain_table(rows, per_subject)
        return
    p = stats.binomtest(agree, decidable, null, alternative="greater").pvalue
    print(f"\n  the model chose the training majority in {agree}/{decidable} = "
          f"{agree / decidable:.4f} of the decidable trials")
    print(f"  null is NOT 0.5 but {null:.4f} = P(prior == target): a model that simply reads")
    print(f"  the EEG agrees with the prior that often by coincidence   ->  p={p:.1e}")
    print(f"  and because that null is below 0.5, following the prior LOSES: the k-fold takes")
    print(f"  the test trial out of the count, tipping a balanced mixture the other way")

    # 3. Dose-response. A confound should get stronger as the prior gets more lopsided;
    #    an artefact of the arithmetic would not.
    buckets = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ca, cb = training_counts(rows, r, per_subject=per_subject)
        gap = abs(ca - cb)
        buckets[min(gap, 3)][1] += 1
        if gap:
            buckets[min(gap, 3)][0] += (r["chosen"] == majority(rows, r, per_subject=per_subject))
    # The null per bucket too: P(majority == target) is not flat across gaps, so the trend
    # has to be read against it and not against a straight line.
    nulls = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ca, cb = training_counts(rows, r, per_subject=per_subject)
        gap = abs(ca - cb)
        nulls[min(gap, 3)][1] += 1
        if gap:
            nulls[min(gap, 3)][0] += (majority(rows, r, per_subject=per_subject) == r["target"])
    print("\n  by how lopsided the training count is (compare each row with its own null):")
    for gap in sorted(buckets):
        hit, total = buckets[gap]
        nh, nt = nulls[gap]
        label = f"gap {gap}" + ("+" if gap == 3 else "")
        note = "   (no majority to follow)" if gap == 0 else f"   null {nh / nt:.3f}"
        print(f"    {label:<7} n={total:<4} follows it {hit}/{total} = {hit / total:.3f}{note}")

    _grain_table(rows, per_subject)


def _grain_table(rows, per_subject):
    """4. At which grain is stimulus identity being read?

    Reachable from both branches of report(): when the mixture grain is degenerate the
    coarser grains are usually still testable, and skipping them would throw away the part
    of the diagnosis that still has a null.
    """
    print("\n  grain of the prior the model is reading:")
    # Each grain gets its OWN null, and they differ a lot -- against a flat 0.5 the coarse
    # grains look innocent and they are not.
    for key, label in [("mixture", "mixture (piece+pair+theme)"),
                       ("pair", "instrument pair, themes pooled"),
                       ("subject", "subject"),
                       ("global", "whole training set")]:
        hit = total = 0
        for r in rows:
            mj = majority(rows, r, key, per_subject)
            if mj is not None:
                hit, total = hit + (r["chosen"] == mj), total + 1
        if not total:
            print(f"    {label:<32} no decidable trial at this grain")
            continue
        null = base_rate(rows, key, per_subject)
        if null in (0.0, 1.0):
            print(f"    {label:<32} {hit}/{total} = {hit / total:.3f}   "
                  f"null {null:.3f}  <- deterministic prior, nothing to test")
            continue
        p = stats.binomtest(hit, total, null, alternative="greater").pvalue
        # A tiny null makes the binomial test uninformative: a model that IGNORES everything
        # and flips a coin agrees with the prior ~0.5 of the time and would also "beat" it
        # with an astronomical p. The three reference points are null (perfect EEG reader),
        # 0.5 (coin flip) and 1.0 (pure prior-follower); prior-following is only demonstrated
        # when agreement sits ABOVE the coin flip. Flag the cases where it does not.
        flag = "" if hit / total > 0.5 else "  <- BELOW a coin flip: not prior-following"
        print(f"    {label:<32} {hit}/{total} = {hit / total:.3f}   "
              f"null {null:.3f}   p={p:.1e}{flag}")
    print("  ⚠ a small null makes p misleading on its own: a coin-flipping model agrees with")
    print("  the prior ~0.500 of the time and beats any null below that. Read agreement")
    print("  against BOTH the null (a perfect decoder scores it) and 0.500 (a coin flip).")
    print("  ⚠ these grains are nested and correlated -- a prior at one grain lifts all the")
    print("  others, so this table says THAT the model follows a training prior, not WHICH")
    print("  grain it reads. Separating them needs a design that decorrelates them.")


def paired_twins(records_csv):
    """The (subject, mixture) pairs that appear with BOTH targets, and their paired statistic.

    Why a paired test at all. The same subject heard the same mixture twice, once attending
    each instrument; the audio is byte-identical and only the instruction differs. Whatever
    makes one instrument easier to reconstruct than the other -- it is louder, it carries the
    melody, it sits in a better frequency range -- is a per-pair constant, and at n=154 that
    nuisance is a large share of the variance.

    The statistic. Write d = r(A) - r(B) for a trial, with A and B in a fixed order. The
    stored margin is r(attended) - r(competitor), so it is +d on the twin that attended A and
    -d on the twin that attended B. Their SUM therefore cancels d entirely:

        D = margin(twin attending A) + margin(twin attending B) = 2 * (attention effect)

    A decoder with a fixed instrument preference and no attention information scores D = 0
    however strong that preference is. No instrument ordering has to be chosen anywhere.
    """
    rows = list(csv.DictReader(open(records_csv)))
    margin_col = "mean_margin" if "mean_margin" in rows[0] else None
    groups = collections.defaultdict(list)
    for r in rows:
        m = (float(r[margin_col]) if margin_col
             else float(r["r_attended"]) - float(r["r_best_unattended"]))
        groups[(r["subject"], r["stim"].rsplit("_", 1)[0])].append((r["stim"], m))
    twins = {k: v for k, v in groups.items() if len(v) == 2 and v[0][0] != v[1][0]}
    singles = len(groups) - len(twins)
    return twins, singles, len(groups)


def paired_report(records_csv):
    twins, singles, total = paired_twins(records_csv)
    D = [a[1] + b[1] for a, b in twins.values()]
    n = len(D)
    print(f"\n  (subject, mixture) pairs: {total} total, {n} with BOTH targets, {singles} with one")
    if n == 0:
        return
    k = sum(d > 0 for d in D)
    p_sign = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    p_wil = stats.wilcoxon(D, alternative="greater").pvalue
    print(f"  paired statistic D > 0 in {k}/{n} pairs  (null 0.500)  sign test p={p_sign:.4f}"
          f"   Wilcoxon signed-rank p={p_wil:.4f}")

    # --- power, computed and not assumed -------------------------------------------------
    # margin = delta +/- (b_pair + eps).  D = 2*delta + eps_A - eps_B  -> Var(D)   = 2 s_eps^2
    #                                     S = margin_A - margin_B      -> Var(S)   = 4 s_b^2 + 2 s_eps^2
    # so both variance components are identified by the twins themselves.
    S = [a[1] - b[1] for a, b in twins.values()]
    s_eps2 = max(np.var(D, ddof=1) / 2.0, 1e-18)
    s_b2 = max((np.var(S, ddof=1) - 2 * s_eps2) / 4.0, 0.0)
    print(f"  variance decomposition from the twins: per-pair instrument bias s_b={np.sqrt(s_b2):.4f}, "
          f"trial noise s_eps={np.sqrt(s_eps2):.4f}  ->  bias is "
          f"{100 * s_b2 / (s_b2 + s_eps2):.0f}% of the single-trial variance")

    def crit(nn):                       # smallest k with a one-sided binomial p <= 0.05
        return next(c for c in range(nn + 1)
                    if stats.binomtest(c, nn, 0.5, alternative="greater").pvalue <= 0.05)
    c154, c47 = crit(154), crit(n)
    print(f"  significance thresholds: unpaired {c154}/154 = {c154 / 154:.4f} · "
          f"paired {c47}/{n} = {c47 / n:.4f}")
    print("  power at alpha=0.05 one-sided, as a function of the true attention effect delta")
    print("  (delta in the same units as the margin; p_acc and p_pair follow from the two")
    print("   variance components above, so nothing here is assumed):")
    print(f"    {'delta':>8} {'p(acc)':>8} {'power154':>9} {'p(pair)':>8} {'power' + str(n):>9}   winner")
    for delta in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050):
        p_acc = stats.norm.cdf(delta / np.sqrt(s_b2 + s_eps2))
        p_pair = stats.norm.cdf(delta * np.sqrt(2.0) / np.sqrt(s_eps2))
        pow_u = stats.binom.sf(c154 - 1, 154, p_acc)
        pow_p = stats.binom.sf(c47 - 1, n, p_pair)
        print(f"    {delta:8.3f} {p_acc:8.3f} {pow_u:9.3f} {p_pair:8.3f} {pow_p:9.3f}   "
              f"{'paired' if pow_p > pow_u else 'unpaired'}")


def _demo():
    """Self-check on synthetic records. Run: python src/madeeg_diagnose.py --demo

    Pins the two facts the whole analysis rests on, one of which cost a wrong conclusion
    before it was checked against the synthetic-EEG control run:
      * a pure PRIOR-follower agrees with the prior always, and scores `base_rate`;
      * a pure TARGET-follower scores 1.0, and still agrees with the prior `base_rate` of
        the time -- so `base_rate`, not 0.5, is the null for "follows the prior".

    Three balanced mixtures (4+4, where leave-one-out always tips the count the other way,
    which is MAD-EEG's design) and three lopsided ones (6+2), so base_rate is strictly
    between 0 and 1 and neither assertion can pass by degeneracy.
    """
    rows = []
    for mix in range(3):
        for targets in (["Aa"] * 4 + ["Bb"] * 4, ["Aa"] * 6 + ["Bb"] * 2):
            tag = f"mix{mix}_{len(rows)}"
            for i, target in enumerate(targets):
                rows.append({"instruments": ["Aa", "Bb"], "target": target, "mixture": tag,
                             "pair": "AaBb", "subject": f"s{i % 3}", "global": "all",
                             "fold": i})
    decidable = [r for r in rows if majority(rows, r) is not None]
    null = base_rate(rows)
    assert 0.0 < null < 1.0, f"degenerate demo: base_rate is {null}"

    for r in rows:                                        # a pure prior-follower
        r["chosen"] = majority(rows, r) or "Aa"
        r["correct"] = int(r["chosen"] == r["target"])
    agree = sum(r["chosen"] == majority(rows, r) for r in decidable)
    acc = sum(r["correct"] for r in decidable) / len(decidable)
    assert agree == len(decidable), "the follower must be detected on every decidable trial"
    assert abs(acc - null) < 1e-9, f"a prior-follower scores base_rate: {acc} vs {null}"

    for r in rows:                                        # a pure target-follower
        r["chosen"], r["correct"] = r["target"], 1
    agree = sum(r["chosen"] == majority(rows, r) for r in decidable)
    assert abs(agree / len(decidable) - null) < 1e-9, (
        f"a perfect decoder still agrees with the prior base_rate={null:.3f} of the time, "
        f"got {agree}/{len(decidable)} -- this is why the null is not 0.5")
    print(f"[demo] PASS  null (base_rate) = {null:.3f}; prior-follower agrees 100% and "
          f"scores {null:.3f}; target-follower scores 1.000 and still agrees {null:.3f} "
          f"-- so agreement must be tested against {null:.3f}, not 0.5")


def _check_rule(records_csv):
    """Gate [1c]: the argmax rule must reproduce step C's 58/154 on its ARCHIVED records.

    Two separate things are checked, because neither alone is enough.

    1. Against the archive. The stored `correct` column must come back out of
       `trial_decision`, on all 154 rows, totalling 58. The archive keeps the per-trial
       margin and not the two per-stem means, so the means are reconstructed as
       [margin, 0] -- faithful, because an argmax does not move when both means shift by
       the same constant. This pins that the file is the one that produced 58/154 and that
       the rule agrees with it trial by trial. It does NOT show that averaging each stem
       and subtracting is the same operation as averaging the margin: that is (2).

    2. Against the step-C rule itself, on random windows. mean_i(a_i - b_i) and
       mean_i(a_i) - mean_i(b_i) are the same number, so on duos the two rules can never
       disagree. Checked rather than asserted, on trials of 1 to 40 windows.
    """
    import random

    rows = list(csv.DictReader(open(records_csv)))
    k = 0
    for r in rows:
        got = int(trial_decision([float(r["mean_margin"]), 0.0], 0) > 0)
        assert got == int(r["correct"]), (
            f"{r['subject']}/{r['stim']}: argmax rule says {got}, record says {r['correct']}")
        k += got
    assert (k, len(rows)) == (58, 154), (
        f"expected step C's 58/154, got {k}/{len(rows)} -- either the rule is not equivalent "
        "or these are not the step C records")
    print(f"[check_rule] {k}/{len(rows)} = {k / len(rows):.4f} -- the argmax rule reproduces "
          f"step C exactly, and every one of the {len(rows)} stored decisions")

    rng = random.Random(0)
    for _ in range(2000):
        w = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(rng.randint(1, 40))]
        step_c = sum(a - b for a, b in w) / len(w) > 0
        step_d = trial_decision([sum(a for a, _ in w) / len(w),
                                 sum(b for _, b in w) / len(w)], 0) > 0
        assert step_c == step_d, f"the two rules disagree on {w}"
    print("[check_rule] and on 2000 random duo trials the step-C rule (mean of the margins) "
          "and the step-D rule (argmax of the per-stem means) never disagree")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", help="madeeg_contrastive_records.csv from a k-fold run")
    ap.add_argument("--madeeg_dir", help="dir with madeeg_preprocessed.yaml")
    ap.add_argument("--folds", default="global", choices=["global", "per_subject"],
                    help="how the run split its data. 'global' (default) = one pooled model on a "
                         "k-fold over all trials -- steps C and D. 'per_subject' = a separate "
                         "model per subject, as madeeg_reconstruction.py's duos_kfold does, so a "
                         "trial of another subject was never in training and must not be counted")
    ap.add_argument("--paired", action="store_true",
                    help="paired test over the (subject, mixture) pairs seen with BOTH targets, "
                         "plus the power calculation that says whether it beats the plain "
                         "binomial on all trials. Needs no madeeg_dir")
    ap.add_argument("--demo", action="store_true", help="self-check on synthetic records, then exit")
    ap.add_argument("--check_rule", metavar="RECORDS_CSV",
                    help="regression gate: the step-D argmax decision rule must return step "
                         "C's 58/154 on step C's archived records. Needs no madeeg_dir")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.check_rule:
        _check_rule(args.check_rule)
        return
    if args.paired:
        if not args.records:
            ap.error("--paired needs --records")
        print(f"\n[paired] {args.records}")
        paired_report(args.records)
        return
    if not args.records or not args.madeeg_dir:
        ap.error("--records and --madeeg_dir are both required (or use --demo)")
    print(f"\n[diagnose] {args.records}   folds={args.folds}")
    report(load(args.records, args.madeeg_dir), per_subject=args.folds == "per_subject")


if __name__ == "__main__":
    main()
