"""Exact paired p for own-vs-other (T = n1 + 2*Binom(n0+n2, 1/2)) and the
Bonferroni McNemar threshold at alpha = 0.05/8 = 0.00625.

POSITIVE CONTROLS (declared before looking at any candidate number):
  * on runs/results/ovo_sweep_REF/madeeg_ownvsother.csv the script must give
    n0/n1/n2 = 36/96/56 and exact paired p = 0.0235 (report of 2026-08-11, par. 3);
  * the Bonferroni threshold table must reproduce the contract par.3 rows:
    60->41, 80->52, 100->63, 120->75, 131->81, 150->91, 180->108.
Only stdlib: math.comb, csv.
"""
import csv, math, sys
from collections import defaultdict

ALPHA_BONF = 0.05 / 8  # K=8, fixed by the contract


def binom_sf_ge(k, n):
    """P(X >= k), X ~ Binom(n, 1/2), exact."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def paired_exact_p(path):
    """One-sided exact p for accuracy > 0.5 accounting for the 188-pair structure.
    A pair with 1 correct of its 2 directions is invariant under the own/other swap;
    a pair with 0 or 2 flips. So T = n1 + 2*Binom(n0+n2, 1/2) and
    P(T >= T_obs) = P(X >= n2 | n = n0+n2, p = 1/2)."""
    per_pair = defaultdict(int)
    n_rows = 0
    for r in csv.DictReader(open(path)):
        a, b = r["seg"], r["other"]
        per_pair[(r["subject"], r["fold"], min(a, b, key=int), max(a, b, key=int))] += int(r["correct"])
        n_rows += 1
    counts = [0, 0, 0]
    for c in per_pair.values():
        counts[c] += 1
    n0, n1, n2 = counts
    k = n1 + 2 * n2
    return dict(n=n_rows, n_pairs=len(per_pair), n0=n0, n1=n1, n2=n2,
                acc=k / n_rows, k=k, p_exact=binom_sf_ge(n2, n0 + n2))


def bonferroni_threshold(n_disc):
    """Smallest integer c with P(X>=c | n_disc, 1/2) < ALPHA_BONF."""
    return next(c for c in range(n_disc + 1) if binom_sf_ge(c, n_disc) < ALPHA_BONF)


if __name__ == "__main__":
    # control 1: the contract's own table
    expected = {60: 41, 80: 52, 100: 63, 120: 75, 131: 81, 150: 91, 180: 108}
    for nd, thr in expected.items():
        got = bonferroni_threshold(nd)
        assert got == thr, f"threshold table mismatch at n_disc={nd}: got {got}, contract says {thr}"
    print("control 1 PASS: Bonferroni thresholds reproduce the contract table")
    # control 2: the REF csv
    ref = paired_exact_p(sys.argv[1])
    assert (ref["n0"], ref["n1"], ref["n2"]) == (36, 96, 56), f"REF pair counts off: {ref}"
    assert abs(ref["p_exact"] - 0.0235) < 5e-4, f"REF exact paired p off: {ref['p_exact']:.4f}"
    print(f"control 2 PASS: REF n0/n1/n2 = 36/96/56, exact paired p = {ref['p_exact']:.4f} (expected: 0.0235)")
    for path in sys.argv[2:]:
        r = paired_exact_p(path)
        print(f"{path}: acc {r['k']}/{r['n']} = {r['acc']:.4f}  n_pairs={r['n_pairs']}  "
              f"n0/n1/n2 = {r['n0']}/{r['n1']}/{r['n2']}  exact paired p = {r['p_exact']:.4f}")
