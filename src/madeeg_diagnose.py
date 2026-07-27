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
    p = stats.binomtest(agree, decidable, 0.5, alternative="greater").pvalue
    print(f"\n  the model chose the training majority in {agree}/{decidable} = "
          f"{agree / decidable:.4f} of the decidable trials (one-sided binomial p={p:.1e})")
    right = sum(majority(rows, r) == r["target"] for r in rows if majority(rows, r))
    print(f"  and that majority is itself the correct answer only {right}/{decidable} = "
          f"{right / decidable:.4f} of the time -> following it LOSES")

    # 3. Dose-response. A confound should get stronger as the prior gets more lopsided;
    #    an artefact of the arithmetic would not.
    buckets = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ca, cb = training_counts(rows, r)
        gap = abs(ca - cb)
        buckets[min(gap, 3)][1] += 1
        if gap:
            buckets[min(gap, 3)][0] += (r["chosen"] == majority(rows, r))
    print("\n  and it follows the prior harder the more lopsided the training count is:")
    for gap in sorted(buckets):
        hit, total = buckets[gap]
        label = f"gap {gap}" + ("+" if gap == 3 else "")
        note = "   (no majority to follow)" if gap == 0 else ""
        print(f"    {label:<7} n={total:<4} follows it {hit}/{total} = {hit / total:.3f}{note}")

    # 4. At which grain is stimulus identity being read?
    print("\n  grain of the prior the model is reading:")
    # `global` is the control that matters: if the model merely preferred the instruments
    # that are attended most often overall, it would show up there and not at mixture level.
    for key, label in [("mixture", "mixture (piece+pair+theme)"),
                       ("pair", "instrument pair, themes pooled"),
                       ("subject", "subject"),
                       ("global", "whole training set (control)")]:
        hit = total = 0
        for r in rows:
            mj = majority(rows, r, key)
            if mj is not None:
                hit, total = hit + (r["chosen"] == mj), total + 1
        p = stats.binomtest(hit, total, 0.5, alternative="greater").pvalue
        print(f"    {label:<32} {hit}/{total} = {hit / total:.3f}   p={p:.1e}")

    # 5. Does the mixture prior survive the subject prior? The two are correlated, because a
    #    subject heard a given mixture at most twice.
    hit = total = 0
    for r in rows:
        mm, ms = majority(rows, r), majority(rows, r, "subject")
        if mm is not None and ms is not None and mm != ms:
            hit, total = hit + (r["chosen"] == mm), total + 1
    p = stats.binomtest(hit, total, 0.5, alternative="greater").pvalue
    print(f"\n  where the mixture prior and the subject prior DISAGREE, the model follows")
    print(f"  the mixture in {hit}/{total} = {hit / total:.3f} (p={p:.3f})")


def _demo():
    """Self-check on synthetic records: a perfect prior-follower must be detected as one,
    and a model that ignores the prior must not be. Run: python src/madeeg_diagnose.py --demo"""
    # A mixture balanced over its two targets -- which is what MAD-EEG's design guarantees --
    # under folds fine enough that removing the test trial tips the remaining count to the
    # other instrument. This is the whole mechanism in eight rows.
    rows = []
    for mix in range(6):
        for i, target in enumerate(["Aa"] * 4 + ["Bb"] * 4):
            rows.append({"instruments": ["Aa", "Bb"], "target": target,
                         "mixture": f"mix{mix}", "pair": "AaBb", "subject": f"s{i % 3}",
                         "fold": i})
    for r in rows:
        r["chosen"] = majority(rows, r) or "Aa"          # a pure prior-follower
        r["correct"] = int(r["chosen"] == r["target"])
    k = sum(r["correct"] for r in rows)
    assert k == 0, f"on a balanced mixture the prior is always the wrong answer, got {k}"
    agree = sum(r["chosen"] == majority(rows, r) for r in rows if majority(rows, r))
    decidable = sum(majority(rows, r) is not None for r in rows)
    assert agree == decidable, "the follower must be detected as following on every trial"

    for i, r in enumerate(rows):                          # now a model that ignores the prior
        r["chosen"] = r["instruments"][i % 2]
        r["correct"] = int(r["chosen"] == r["target"])
    agree = sum(r["chosen"] == majority(rows, r) for r in rows if majority(rows, r))
    assert agree / decidable < 0.7, f"a prior-blind model must not look like a follower ({agree}/{decidable})"
    print(f"[demo] PASS  prior-follower scores {k}/{len(rows)} = {k / len(rows):.3f} (below "
          f"chance by construction) and is detected on {decidable}/{decidable} trials; "
          f"a prior-blind model is not")


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
