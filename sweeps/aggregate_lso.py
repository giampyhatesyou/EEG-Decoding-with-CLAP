#!/usr/bin/env python3
# CHANGED(baseline): new file -- analysis helper (not in upstream). Aggregates the
#                    leave-song-out / within results into a per-model MACRO table.
"""Per-model accuracy summary over results/.

Run inside the eeg_attention env (from anywhere -- results/ is resolved from the
repo root, not from the current directory):

    python sweeps/aggregate_lso.py

Prints MACRO (mean over the held-out classes; the honest metric), GLOBAL and the
per-class accuracy for every (objective, audio_repr, eeg_repr, cv_mode) found. Old
runs predating the eeg_repr flag are treated as eeg_repr=raw. cv_mode is part of the
key, so leave_song_out, within and leave_subject_out never get mixed together.
"""
import glob
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas -- run inside the eeg_attention env")

TASK = {0: "vocal", 1: "drum", 2: "bass", 3: "others"}
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def hparams(version_dir):
    cfg = {"objective": "contrastive", "audio_repr": "?", "eeg_repr": "raw",
           "cv_mode": "?", "cv_held_out_id": "?"}
    try:
        for line in open(os.path.join(version_dir, "hparams.yaml")):
            s = line.strip()
            for key in cfg:
                if s.startswith(key + ":"):
                    cfg[key] = s.split(":", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return cfg


# Deduplicate by (objective, audio_repr, eeg_repr, cv_mode, held-out id): if a fold was
# run more than once, keep the last test_records.csv found for it.
by_fold = {}
for f in glob.glob(f"{RESULTS}/*/nmed-CL-*/version_*/test_records.csv"):
    c = hparams(os.path.dirname(f))
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if "correct" not in df or "task" not in df or len(df) == 0:
        continue
    by_fold[(c["objective"], c["audio_repr"], c["eeg_repr"],
             c["cv_mode"], c["cv_held_out_id"])] = df

groups = {}
for (obj, arepr, erepr, cvmode, held), df in by_fold.items():
    groups.setdefault((obj, arepr, erepr, cvmode), []).append(df)

header = "{:42}{:>6}{:>8}{:>8}   {}".format(
    "objective / audio / eeg / eval", "# of runs", "MACRO", "GLOBAL", "per-class [v d b o]")
print(header)
print("-" * len(header))
for key in sorted(groups):
    dfs = groups[key]
    alld = pd.concat(dfs)
    per = alld.groupby("task")["correct"].mean()
    pc = " ".join(("%.2f" % per[t]) if t in per.index else " -  " for t in range(4))
    print("{:42}{:>6}{:>8.3f}{:>8.3f}   [{}]".format(
        " / ".join(key), len(dfs), per.mean(), alld["correct"].mean(), pc))

print("\nchance = 0.25 | MACRO is the honest metric "
      "(GLOBAL is inflated by the ~49% vocal prior).")
print("The SpectraCLIP question: compare eeg=raw vs eeg=spectra at eval=leave_song_out.")
