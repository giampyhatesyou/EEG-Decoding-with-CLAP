"""Leave-song-out, fold by fold, for the four models of Tab. 7.1.

ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed. It reads the
test records of the folds pinned in results_manifest.tsv (song_out rows) from the run
archive; nothing is retrained.

POSITIVE CONTROL, crossed first (exit 1 otherwise): sweeps/report.py's own loader and MACRO
  reproduce Tab. 7.1 -- audio_only_raw 0.181, eeg_only 0.247, contrastive_clap 0.268 over 20
  folds and 0.258 over the 14 folds of the current code revision, contrastive_raw 0.142.
THEN: the accuracy of every fold with its held-out song and code revision (vintage). Each
  fold holds out one song, and every trial of a song has the same target slot, so a fold's
  accuracy is how often the model picks that song's slot; the null is 0.25 (4 fixed slots).

Usage: /opt/miniconda3/bin/python scripts/replicate/song_out_per_fold.py [--results_dir DIR] [--fig PDF]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "sweeps"))
from report import load_fold, read_manifest  # noqa: E402

OUT = "2026-09-27_song_out_per_fold.txt"
MODELS = ["audio_only_raw", "eeg_only", "contrastive_clap", "contrastive_raw"]
PUBLISHED = {"audio_only_raw": 0.181, "eeg_only": 0.247, "contrastive_clap": 0.268,
             "contrastive_raw": 0.142, "contrastive_clap, 2026-07 only": 0.258}


def macro(dfs):
    """report.py's MACRO: per-task accuracy over the pooled records, then the mean."""
    import pandas as pd
    return pd.concat(dfs).groupby("task")["correct"].mean().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "..", "_baldo_archive_2026-07-18", "results"))
    ap.add_argument("--fig", default="")
    args = ap.parse_args()
    out_path = os.path.join(ROOT, "docs", "provenance", OUT)
    L = ["!! ANALYSIS OF 2026-09-27, requested by the supervisor, on data already analysed:",
         "!! the pinned leave-song-out folds of Tab. 7.1, read back from the run archive.",
         "!! Null 0.25 per fold (4 fixed slots). A fold has ONE target slot (one held-out song),",
         "!! so its accuracy is the rate at which the model picks that slot.",
         ""]

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    rows = [r for r in read_manifest(os.path.join(ROOT, "results_manifest.tsv")) if r["table"] == "song_out"]
    problems, multi, folds = [], [], {m: [] for m in MODELS}
    for r in rows:
        df = load_fold(args.results_dir, r, problems, multi)
        if df is not None:
            folds[r["model"]].append((int(r["held_out"]), r["vintage"], r["run_tag"], df))
    if problems:
        say("POSITIVE CONTROL FAILED, the loader reported: " + "; ".join(problems))
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)

    say("=== POSITIVE CONTROL: report.py's loader and MACRO against Tab. 7.1 ===")
    got = {m: macro([f[3] for f in folds[m]]) for m in MODELS}
    got["contrastive_clap, 2026-07 only"] = macro([f[3] for f in folds["contrastive_clap"] if f[1] == "2026-07"])
    ok = True
    for k, v in PUBLISHED.items():
        n = len(folds[k.split(",")[0]]) if "," not in k else sum(f[1] == "2026-07" for f in folds["contrastive_clap"])
        good = round(got[k], 3) == v
        ok &= good
        say(f"  {k:32s} {n:2d} folds  MACRO {got[k]:.4f}  published {v:.3f}  [{'OK' if good else 'FAIL'}]")
    if not ok:
        say("POSITIVE CONTROL FAILED: nothing below this line exists.")
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)
    one = all(f[3]["task"].nunique() == 1 for m in MODELS for f in folds[m])
    say(f"  every fold has a single target slot: {'yes' if one else 'NO'}  [{'OK' if one else 'FAIL'}]")
    if not one:
        say("POSITIVE CONTROL FAILED: nothing below this line exists.")
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)
    if multi:
        say(f"  (tags with more than one version, newest taken as report.py does: {', '.join(multi)})")

    say()
    say("=== ACCURACY OF EVERY FOLD (null 0.25) ===")
    say(f"  {'song':>5s} " + " ".join(f"{m:>22s}" for m in MODELS))
    songs = sorted({f[0] for m in MODELS for f in folds[m]})
    acc = {m: {f[0]: (f[3]["correct"].mean(), len(f[3]), f[1]) for f in folds[m]} for m in MODELS}
    for s in songs:
        cells = []
        for m in MODELS:
            a, n, v = acc[m].get(s, (np.nan, 0, "-"))
            cells.append(f"{a:.3f} (n={n:3d}, {v:9s})")
        say(f"  {s:>5d} " + " ".join(f"{c:>22s}" for c in cells))
    say()
    say("  per model, over folds (each fold weighted once; MACRO above weights trials):")
    for m in MODELS:
        a = np.array([acc[m][s][0] for s in songs if s in acc[m]])
        say(f"  {m:18s} folds {len(a):2d}  median {np.median(a):.3f}  min {a.min():.3f}  max {a.max():.3f}  "
            f"folds above 0.25: {int((a > 0.25).sum())}/{len(a)}  at 0: {int((a == 0).sum())}/{len(a)}")
    open(out_path, "w").write("\n".join(L) + "\n")
    print(f"\n-> {out_path}")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        rng = np.random.RandomState(0)          # horizontal jitter only
        for i, m in enumerate(MODELS):
            for s in songs:
                if s not in acc[m]:
                    continue
                a, _, v = acc[m][s]
                ax.plot(i + rng.uniform(-0.18, 0.18), a, "o", ms=3.5,
                        mfc="none" if v != "2026-07" else "0.25", mec="0.25")
        ax.axhline(0.25, color="tab:red", lw=1, ls="--", label="null 0.25")
        ax.set_xticks(range(len(MODELS)), ["audio only", "EEG only", "contrastive,\npre-trained audio",
                                           "contrastive,\nlearned audio"])  # the names of Table 7.1
        ax.set_ylabel("fold accuracy (one held-out song)")
        ax.set_ylim(-0.03, 1.03)
        ax.plot([], [], "o", mfc="0.25", mec="0.25", ms=3.5, label="current code (2026-07)")
        ax.plot([], [], "o", mfc="none", mec="0.25", ms=3.5, label="other revision")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.fig)
        print(f"-> {args.fig}")


if __name__ == "__main__":
    main()
