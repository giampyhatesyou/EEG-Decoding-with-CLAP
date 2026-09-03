#!/usr/bin/env bash
# =============================================================================
# Arm A - same/diff-melody split of the faithful-package runs (DESCRIPTIVE).
#
# WHAT IT MEASURES: where the same-melody advantage reported by the paper lives
#   in frequency. Every duo decision of an arm A run is labelled "same" when the
#   subject's training solos of the ATTENDED instrument include one with the
#   duo's (piece, theme), and "diff" otherwise; accuracy is reported per group.
# COST       : ~1 s CPU. It re-reads records already produced by
#              armA_paper_protocol.sh plus the dataset's session metadata.
# BUDGET     : ZERO new looks. No EEG is opened, ['response'] is never touched.
#              The underlying runs were already spent: the split is EXPLORATORY
#              BY CONSTRUCTION and DESCRIPTIVE - no test of the difference is
#              computed here, and none may be quoted.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   band 1-8 Hz    (armA_paper_package)   same 46/89 = 0.5169  diff 33/61 = 0.5410
#   band 0.2-40 Hz (armA_paper_band02_40) same 51/89 = 0.5730  diff 30/61 = 0.4918
#   READING: the paper's same-melody advantage - its own stimulus-following
#   component - lives in the slow (<1 Hz) contour. A 1-Hz high-pass removes it;
#   lowering the high-pass to 0.2 Hz re-opens it while diff-melody drops to chance.
#
# CANARY     : the structural control is inside the driver - the split must come
#              out 89 same / 61 diff, and the driver exits non-zero if it does
#              not, because a different shape is not comparable with the pinned
#              numbers. Do NOT relax it to "about 89".
# PRE-REGISTRATION: none (descriptive split of runs already made).
# PROVENANCE : docs/provenance/2026-08-11_armA_same_diff_melody.txt (pinned,
#              produced ad hoc on 2026-08-11 by a script that was never saved).
#              This file is the rewrite: 2026-09-01, verified to reproduce the
#              pinned 0.2-40 Hz figures exactly before being committed.
#
# INPUT      : runs/results/armA_paper_{package,band02_40}/madeeg_records.csv.
#              They are gitignored: run scripts/replicate/armA_paper_protocol.sh
#              first if this is a fresh clone. The 0.2-40 Hz records are also in
#              docs/provenance/ and this script falls back to them.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # pinned provenance is never touched
OUT="$OUT_DIR/armA_same_diff_melody.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

RECORDS=()
for R in armA_paper_package armA_paper_band02_40; do
  F="$REPO/runs/results/$R/madeeg_records.csv"
  [ -f "$F" ] && RECORDS+=("$F")
done
# Fresh clone: the run outputs are gitignored, the pinned 0.2-40 Hz records are not.
if [ ${#RECORDS[@]} -eq 0 ]; then
  echo "no arm A run found under runs/results/; falling back to the pinned records"
  RECORDS+=("$REPO/docs/provenance/2026-08-17_armA_paper_band02_40_RESULT_madeeg_records.csv")
fi

"$PY" src/madeeg_same_diff_melody.py --madeeg_dir "$MADEEG_DIR" \
      --records "${RECORDS[@]}" --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "same|CONTROL" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-11_armA_same_diff_melody.txt"
