#!/usr/bin/env python3
# CHANGED(baseline): new file -- analysis helper (not in upstream). Replaces
#                    aggregate_lso.py, which globbed results/ and deduplicated by a
#                    partial key: two runs of the same fold collided and whichever the
#                    filesystem returned last silently won. That moved a reported number
#                    from 0.142 to 0.154 with no new evidence. This one reads a pinned
#                    manifest instead, so a number can only change if a human edits it.
"""Render THE results table from results_manifest.tsv.

    python sweeps/report.py                      # table to stdout + RESULTS.md
    python sweeps/report.py --results-dir DIR    # e.g. the offline archive mirror
    python sweeps/report.py --check              # verify pins only, write nothing

Every row of the manifest names one fold. This script reads only those folds, checks
each one against its own hparams.yaml, and aggregates. Nothing is discovered from disk,
so adding a run directory cannot change a published number: pin it, or it does not count.

MACRO (mean over the held-out classes) is the honest metric; GLOBAL is inflated by the
~49% vocal prior. Chance = 0.25 for the four-way task.
"""
import argparse
import glob
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas -- run inside the eeg_attention env")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "results_manifest.tsv")
TASKS = ["vocal", "drum", "bass", "others"]
# manifest model name -> the hparams a run must actually have, so a wrong pin is caught
MODELS = {
    "contrastive_raw": ("contrastive", "raw", "raw"),
    "contrastive_clap": ("contrastive", "clap", "raw"),
    "audio_only_raw": ("classify_audio", "raw", "raw"),
    "audio_only_clap": ("classify_audio", "clap", "raw"),
    "eeg_only": ("classify_eeg", "raw", "raw"),
}


def hparams(version_dir):
    cfg = {"objective": "contrastive", "audio_repr": "?", "eeg_repr": "raw",
           "cv_mode": "?", "cv_held_out_id": "?"}
    try:
        for line in open(os.path.join(version_dir, "hparams.yaml")):
            s = line.strip()
            for key in cfg:
                if s.startswith(key + ":"):
                    cfg[key] = s.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return cfg


def read_manifest(path):
    rows = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "table":          # header
                continue
            if len(parts) != 6:
                sys.exit(f"{path}:{n}: expected 6 tab-separated fields, got {len(parts)}")
            rows.append(dict(zip(("table", "model", "eval", "held_out", "run_tag", "vintage"), parts)))
    return rows


def load_fold(results_dir, row, problems, multi):
    """Return the pinned fold's records, or None (appending to `problems`)."""
    versions = sorted(glob.glob(os.path.join(
        results_dir, row["run_tag"], "nmed-CL-*", "version_*", "test_records.csv")))
    if not versions:
        problems.append(f"{row['run_tag']}: no test_records.csv under {results_dir}")
        return None
    if len(versions) > 1:
        # Re-tests of the same config under one tag. Take the newest and say so: silence
        # about which of two runs won is what let a converged fold masquerade as a capped
        # one for three days. A *different* config would need its own tag and its own row.
        multi.append(f"{row['run_tag']}->{os.path.basename(os.path.dirname(versions[-1]))}")
    version_dir = os.path.dirname(versions[-1])

    # The pin must describe the run it points at, otherwise the manifest is fiction.
    if row["model"] in MODELS:
        want = MODELS[row["model"]]
        cfg = hparams(version_dir)
        got = (cfg["objective"], cfg["audio_repr"], cfg["eeg_repr"])
        if got != want:
            problems.append(f"{row['run_tag']}: pinned as {row['model']} {want} but hparams say {got}")
            return None
        if cfg["cv_mode"] != row["eval"]:
            problems.append(f"{row['run_tag']}: pinned as {row['eval']} but hparams say {cfg['cv_mode']}")
            return None
        if cfg["cv_held_out_id"] != row["held_out"]:
            problems.append(f"{row['run_tag']}: pinned held_out={row['held_out']} "
                            f"but hparams say {cfg['cv_held_out_id']}")
            return None

    df = pd.read_csv(versions[-1])
    if "correct" not in df or "task" not in df or df.empty:
        problems.append(f"{row['run_tag']}: test_records.csv has no usable rows")
        return None
    return df


def render(rows, results_dir, problems, multi):
    out = []
    groups = {}
    for row in rows:
        df = load_fold(results_dir, row, problems, multi)
        if df is not None:
            groups.setdefault((row["table"], row["model"]), []).append((row, df))

    for table in dict.fromkeys(r["table"] for r in rows):
        keys = [k for k in groups if k[0] == table]
        if not keys:
            continue
        out.append(f"\n## {table}\n")
        out.append("| model | folds | MACRO | GLOBAL | per-class [v d b o] | vintage |")
        out.append("|---|---:|---:|---:|---|---|")
        for key in sorted(keys):
            entries = groups[key]
            alld = pd.concat([df for _, df in entries])
            per = alld.groupby("task")["correct"].mean()
            pc = " ".join(f"{per[t]:.2f}" if t in per.index else "  -  " for t in range(4))
            vint = sorted({r["vintage"] for r, _ in entries})
            mix = vint[0] if len(vint) == 1 else " + ".join(
                f"{v}x{sum(1 for r, _ in entries if r['vintage'] == v)}" for v in vint)
            out.append(f"| {key[1]} | {len(entries)} | **{per.mean():.3f}** | "
                       f"{alld['correct'].mean():.3f} | {pc} | {mix} |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None,
                    help="where the run directories live (default: the repo's runs/results)")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--check", action="store_true", help="verify the pins, write nothing")
    args = ap.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        # Load paths.py by file, not as `utils.paths`: the package __init__ pulls in the
        # training deps (pytz, torch, ...) and this script only needs one path helper.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_paths", os.path.join(REPO, "src", "utils", "paths.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        results_dir = mod.resolve_log_dir()

    rows = read_manifest(args.manifest)
    problems, multi = [], []
    body = render(rows, results_dir, problems, multi)

    header = [f"# Results — {len(rows)} pinned folds",
              "",
              f"Generated by `python sweeps/report.py` from `results_manifest.tsv`.",
              "Do not edit by hand: edit the manifest and re-run.",
              "",
              "Chance = 0.25. MACRO (mean over held-out classes) is the honest metric;",
              "GLOBAL is inflated by the ~49% vocal prior.",
              "",
              "`vintage` says which code revision produced each fold — see the manifest header.",
              "A row mixing vintages is reported as such rather than silently averaged away."]
    text = "\n".join(header + body) + "\n"
    print(text)

    if multi:
        print(f"[note] {len(multi)} tag(s) hold several versions (re-tests of the same "
              f"config); using the newest: {', '.join(multi)}", file=sys.stderr)
    for p in problems:
        print("[BAD PIN] " + p, file=sys.stderr)

    fatal = problems
    if args.check:
        return 1 if fatal else 0
    if fatal:
        print(f"\n{len(fatal)} bad pin(s): RESULTS.md not written.", file=sys.stderr)
        return 1
    with open(os.path.join(REPO, "RESULTS.md"), "w") as fh:
        fh.write(text)
    print(f"wrote {os.path.join(REPO, 'RESULTS.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
