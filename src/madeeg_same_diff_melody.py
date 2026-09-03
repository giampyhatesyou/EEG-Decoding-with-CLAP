#!/usr/bin/env python3
"""Same/diff-melody split of an arm A run (descriptive).

Rewritten on 2026-09-01. The original script was never saved: the split had been
computed ad hoc on 2026-08-11 and only its OUTPUT survived, in
docs/provenance/2026-08-11_armA_same_diff_melody.txt. This file reproduces that
output from the same two inputs, so the figure quoted in the thesis has code
behind it and not only a text file.

WHAT IT DOES. Every duo decision of an arm A run is labelled

    same  the subject's training solos of the ATTENDED instrument include one
          with the duo's (piece, theme) - the decoder has heard that melody,
          played by that instrument, in training;
    diff  they do not.

Solos are pooled per instrument, exactly as --filters per_instrument pools them
when the decoders are trained, so this is a property of the training material and
not a new analysis choice.

WHAT IT COSTS. Nothing. It re-reads the records of a run that has already been
made and the dataset's session metadata. It never opens EEG, never opens
['response'], and spends no data-looking budget.

WHAT IT IS. Descriptive. No formal test of the same-vs-diff difference is
computed here and none may be quoted: the two groups are unequal (89 vs 61),
the duos were already spent when the split was first computed, and the reading
it supports is about WHERE the published advantage lives in frequency, not about
its significance.

STRUCTURAL CONTROL. The split must come out 89 same and 61 diff on the full arm A
package. If it does not, the parse of the stimulus keys has drifted and the
figures are not comparable with the pinned ones: the script exits non-zero and
prints nothing quotable.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import yaml

# A stimulus key looks like
#   classique_morceau1_solo_Co_theme2_mono
#   classique_morceau1_duo_CoFl_theme1_stereo_Co        (records)
#   classique_morceau1_duo_CoFl_theme1_stereo_lcr_Co    (sequences yaml)
# The piece is everything before _solo_/_duo_; the theme is its own field. The
# spatial tag differs between the two files and is deliberately not parsed.
SOLO = re.compile(r"^(?P<piece>.+?)_solo_(?P<instr>[A-Za-z]+)_(?P<theme>theme\d+)_")
DUO = re.compile(r"^(?P<piece>.+?)_duo_(?P<pair>[A-Za-z]+)_(?P<theme>theme\d+)_")


def solos_per_subject(sequences_yaml: str) -> dict[str, set[tuple[str, str, str]]]:
    """(piece, theme, instrument) of every solo each subject was played."""
    sessions = yaml.safe_load(open(sequences_yaml, encoding="utf-8"))
    out: dict[str, set[tuple[str, str, str]]] = collections.defaultdict(set)
    for subject, trials in sessions.items():
        for key in trials:
            m = SOLO.match(key)
            if m:
                out[subject].add((m["piece"], m["theme"], m["instr"]))
    return out


def split(records_csv: str, solos) -> dict[str, list[int]]:
    """[n_correct, n_total] for each group, over the duo rows of a run."""
    counts = {"same": [0, 0], "diff": [0, 0]}
    for row in csv.DictReader(open(records_csv, encoding="utf-8")):
        if row.get("ensemble") != "duo":
            continue
        m = DUO.match(row["stim"])
        if not m:
            sys.exit(f"stimulus key not parsed, refusing to guess: {row['stim']}")
        heard = (m["piece"], m["theme"], row["target_instr"]) in solos[row["subject"]]
        group = counts["same" if heard else "diff"]
        group[0] += int(row["correct"])
        group[1] += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True, nargs="+",
                   help="madeeg_records.csv of one or more arm A runs")
    p.add_argument("--madeeg_dir", default=os.path.expanduser("~/madeeg"))
    p.add_argument("--sequences", default=None,
                   help="default: <madeeg_dir>/madeeg_sequences_raw.yaml")
    p.add_argument("--expect_same", type=int, default=89)
    p.add_argument("--expect_diff", type=int, default=61)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sequences = a.sequences or os.path.join(a.madeeg_dir, "madeeg_sequences_raw.yaml")
    solos = solos_per_subject(sequences)

    lines = ["ARM A -- same/diff-melody split (descriptive; no new looks, no test)",
             f"sequences : {sequences}",
             '"same" = the subject\'s training solos of the ATTENDED instrument include '
             "one with the duo's (piece, theme)",
             ""]
    failed = False
    for records in a.records:
        c = split(records, solos)
        same, diff = c["same"], c["diff"]
        ok = same[1] == a.expect_same and diff[1] == a.expect_diff
        failed |= not ok
        lines.append(f"{os.path.basename(os.path.dirname(records)) or records}")
        lines.append(f"  same {same[0]}/{same[1]} = {same[0] / max(same[1], 1):.4f}"
                     f"   diff {diff[0]}/{diff[1]} = {diff[0] / max(diff[1], 1):.4f}")
        lines.append(f"  STRUCTURAL CONTROL {'PASSED' if ok else 'FAILED'}: "
                     f"expected {a.expect_same} same / {a.expect_diff} diff, "
                     f"got {same[1]} / {diff[1]}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if failed:
        print("REFUSING: the split does not have the shape it had when the pinned "
              "numbers were produced. Nothing above is comparable with them.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
