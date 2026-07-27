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

import yaml
from scipy import stats


def load(records_csv, madeeg_dir):
    """Records joined with the metadata that says which instruments a mixture holds."""
    meta = yaml.load(open(os.path.join(madeeg_dir, "madeeg_preprocessed.yaml")),
                     Loader=yaml.FullLoader)
    rows = list(csv.DictReader(open(records_csv)))
    for r in rows:
        m = meta[r["subject"]][r["stim"]]
        r["instruments"] = list(m["instruments"])
        r["target"] = m["target"]
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


def training_counts(rows, r, key="mixture"):
    """How often each of this trial's two instruments is the target in its TRAINING folds."""
    counts = collections.Counter(q["target"] for q in rows
                                 if q[key] == r[key] and q["fold"] != r["fold"])
    a, b = r["instruments"]
    return counts[a], counts[b]


def majority(rows, r, key="mixture"):
    """The instrument the training folds attend more often, or None when they tie."""
    ca, cb = training_counts(rows, r, key)
    a, b = r["instruments"]
    return None if ca == cb else (a if ca > cb else b)


def base_rate(rows, key="mixture"):
    """P(prior == target): how often following the prior would be right anyway.

    This, NOT 0.5, is the null for "does the model follow the prior". A model that reads the
    EEG and answers with the target agrees with the prior exactly this often -- so testing
    agreement against 0.5 would call a perfectly good decoder a prior-follower whenever the
    prior happens to be right more than half the time. It is a property of the data alone,
    no model involved. The synthetic-EEG control run lands on it at every grain, which is
    what confirms the statistic is measuring the model and not the design.
    """
    decidable = [r for r in rows if majority(rows, r, key) is not None]
    return sum(majority(rows, r, key) == r["target"] for r in decidable) / len(decidable)


def report(rows):
    k, n = sum(r["correct"] for r in rows), len(rows)
    print(f"  {k}/{n} = {k / n:.4f} correct, chance 0.500\n")

    # 1. The effect, stated as a 2x2 with no modelling in between.
    cell = collections.Counter()
    for r in rows:
        mj = majority(rows, r)
        if mj is not None:
            cell[(mj == r["target"], bool(r["correct"]))] += 1
    tt, tf = cell[(True, True)], cell[(True, False)]
    ft, ff = cell[(False, True)], cell[(False, False)]
    odds, p_fisher = stats.fisher_exact([[tt, tf], [ft, ff]])
    print("  accuracy split by whether the trial's own target was the training majority")
    print(f"    target IS  the majority of its mixture: {tt}/{tt + tf} = {tt / (tt + tf):.3f}")
    print(f"    target is NOT                         : {ft}/{ft + ff} = {ft / (ft + ff):.3f}")
    print(f"    Fisher exact p = {p_fisher:.2e}   odds ratio = {odds:.2f}")
    print("    chance is 0.500 in BOTH cells: every duo mixture is presented with both of")
    print("    its instruments as the target, so the prior is not legitimate information")

    # 2. The same effect from the model's side: does it choose the prior?
    agree = decidable = 0
    for r in rows:
        mj = majority(rows, r)
        if mj is not None:
            agree, decidable = agree + (r["chosen"] == mj), decidable + 1
    null = base_rate(rows)
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
        ca, cb = training_counts(rows, r)
        gap = abs(ca - cb)
        buckets[min(gap, 3)][1] += 1
        if gap:
            buckets[min(gap, 3)][0] += (r["chosen"] == majority(rows, r))
    # The null per bucket too: P(majority == target) is not flat across gaps, so the trend
    # has to be read against it and not against a straight line.
    nulls = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ca, cb = training_counts(rows, r)
        gap = abs(ca - cb)
        nulls[min(gap, 3)][1] += 1
        if gap:
            nulls[min(gap, 3)][0] += (majority(rows, r) == r["target"])
    print("\n  by how lopsided the training count is (compare each row with its own null):")
    for gap in sorted(buckets):
        hit, total = buckets[gap]
        nh, nt = nulls[gap]
        label = f"gap {gap}" + ("+" if gap == 3 else "")
        note = "   (no majority to follow)" if gap == 0 else f"   null {nh / nt:.3f}"
        print(f"    {label:<7} n={total:<4} follows it {hit}/{total} = {hit / total:.3f}{note}")

    # 4. At which grain is stimulus identity being read?
    print("\n  grain of the prior the model is reading:")
    # Each grain gets its OWN null, and they differ a lot -- against a flat 0.5 the coarse
    # grains look innocent and they are not.
    for key, label in [("mixture", "mixture (piece+pair+theme)"),
                       ("pair", "instrument pair, themes pooled"),
                       ("subject", "subject"),
                       ("global", "whole training set")]:
        hit = total = 0
        for r in rows:
            mj = majority(rows, r, key)
            if mj is not None:
                hit, total = hit + (r["chosen"] == mj), total + 1
        null = base_rate(rows, key)
        p = stats.binomtest(hit, total, null, alternative="greater").pvalue
        print(f"    {label:<32} {hit}/{total} = {hit / total:.3f}   "
              f"null {null:.3f}   p={p:.1e}")
    print("  ⚠ these grains are nested and correlated -- a prior at one grain lifts all the")
    print("  others, so this table says THAT the model follows a training prior, not WHICH")
    print("  grain it reads. Separating them needs a design that decorrelates them.")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", help="madeeg_contrastive_records.csv from a k-fold run")
    ap.add_argument("--madeeg_dir", help="dir with madeeg_preprocessed.yaml")
    ap.add_argument("--demo", action="store_true", help="self-check on synthetic records, then exit")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if not args.records or not args.madeeg_dir:
        ap.error("--records and --madeeg_dir are both required (or use --demo)")
    print(f"\n[diagnose] {args.records}")
    report(load(args.records, args.madeeg_dir))


if __name__ == "__main__":
    main()
