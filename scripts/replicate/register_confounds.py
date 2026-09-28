"""The two confounds of the stereo register test (Exp. 11), counted from the metadata.

ANALYSIS OF 2026-09-28, on metadata only: no EEG is read. The counts were first made by hand
on 11 August 2026 while writing the Exp. 11 analysis plan and never saved; this script makes
them reproducible. The 11 August p for the order count (0.39) was P(X >= 25 | 47), one short
of the observed 26: the exact one-sided value is printed below.

POSITIVE CONTROL, crossed first (exit 1 otherwise): the 47 stereo twin pairs rebuilt with the
experiment's own `build_index`, and the gain count of the analysis plan, 27 / 13 / 7.
THEN: the presentation order, from the first timestamp of each trial in
madeeg_sequences_raw.yaml, cross-checked against the first sample index of the same trial.

Usage: /opt/miniconda3/bin/python scripts/replicate/register_confounds.py --madeeg_dir ~/madeeg
"""
import argparse
import os
import sys

import h5py
import yaml
from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import madeeg_spectral_attention as M  # noqa: E402

OUT = os.path.join(ROOT, "docs", "provenance", "2026-09-28_register_confounds.txt")


def seconds(ts):
    h, m, s, ms = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s + ms / 1000


def raw_key(seq_subj, stim):
    """Preprocessed `..._stereo_Co` -> raw `..._stereo_lcr_Co` (the only stereo layout)."""
    pre, target = stim.split("_stereo_")
    hits = [k for k in seq_subj if k.startswith(pre + "_stereo_") and k.endswith("_" + target)]
    assert len(hits) == 1, f"{stim}: {len(hits)} raw keys -- STOP"
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", default=os.path.expanduser("~/madeeg"))
    md = ap.parse_args().madeeg_dir
    f = h5py.File(os.path.join(md, "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.safe_load(open(os.path.join(md, "madeeg_preprocessed.yaml")))
    seq = yaml.safe_load(open(os.path.join(md, "madeeg_sequences_raw.yaml")))
    trials, pairs = M.build_index(f, meta, M.stem_centroids(f, meta))
    ster = [p for p in pairs if meta[p["subj"]][trials[p["i_hi"]]["stim"]]["spatial"] == "stereo"]
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("!! ANALYSIS OF 2026-09-28 on metadata only (no EEG): the two confounds of the stereo")
    say("!! register test (Exp. 11), first counted by hand on 11 August 2026 and never saved.")
    say("")
    louder = sum(p["gain_hi"] > p["gain_lo"] for p in ster)
    softer = sum(p["gain_hi"] < p["gain_lo"] for p in ster)
    equal = len(ster) - louder - softer
    ok = len(ster) == 47 and (louder, softer, equal) == (27, 13, 7)
    say("=== POSITIVE CONTROL (analysis plan of 11 August, crossed first) ===")
    say(f"  stereo twin pairs             {len(ster)}   plan 47")
    say(f"  higher register louder/softer/equal  {louder}/{softer}/{equal}   plan 27/13/7   "
        f"[{'OK' if ok else 'FAIL'}]")
    if not ok:
        sys.exit(1)

    first_ts = first_n = agree = 0
    for p in ster:
        s = seq[p["subj"]]
        a = s[raw_key(s, trials[p["i_hi"]]["stim"])]
        b = s[raw_key(s, trials[p["i_lo"]]["stim"])]
        by_ts = seconds(a["timestamps"][0]) < seconds(b["timestamps"][0])
        by_n = a["n_ech"][0] < b["n_ech"][0]
        first_ts += by_ts
        first_n += by_n
        agree += by_ts == by_n
    n = len(ster)
    say("")
    say("=== PRESENTATION ORDER (the higher-register trial of each pair presented first) ===")
    say(f"  by first timestamp {first_ts}/{n} · by first sample index {first_n}/{n} · agree {agree}/{n}")
    say(f"  null {n / 2:.1f}/{n} = 0.500   exact binomial p one-sided {binomtest(first_ts, n, alternative='greater').pvalue:.4f}"
        f" · two-sided {binomtest(first_ts, n).pvalue:.4f}")
    say(f"  (the 11 August figure 0.39 is P(X >= {first_ts - 1} | {n}) = "
        f"{binomtest(first_ts - 1, n, alternative='greater').pvalue:.4f}: one short of the count)")
    say("")
    say("=== LOUDNESS (the higher-register instrument louder, among unequal pairs) ===")
    m = louder + softer
    say(f"  {louder}/{m}   null 0.500   exact binomial p one-sided "
        f"{binomtest(louder, m, alternative='greater').pvalue:.4f} · two-sided {binomtest(louder, m).pvalue:.4f}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
