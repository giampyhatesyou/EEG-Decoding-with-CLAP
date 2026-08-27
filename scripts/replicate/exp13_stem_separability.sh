#!/usr/bin/env bash
# =============================================================================
# Exp. 13 - AUDIO-ONLY separability gate for arm C.
#
# WHAT IT MEASURES: under which representation (6 candidates) the two competing
#   stems of a duo become as decorrelated as stems from different songs.
# COST       : ~10 s CPU (measured: 9.81 s wall, Mac arm64, 2026-08-12).
# BUDGET     : NO held-out looks - zero. It opens the HDF5 only for `soli` and
#              the metadata: never ['response'], never a trial, never a label.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   CANARY  mel-8  mean 0.1772  median 0.1442
#   CANARY  flux   mean 0.2874  median 0.2417   flux > mel on 30/36
#   C3 MFCC-13  within 0.0533 / cross-song floor 0.0256 / 32/36 / gap +0.0277
#   VERDICT: GO, the only candidate passing both criteria = C3 MFCC-13.
#   The canary is crossed BEFORE anything else BY THE DRIVER ITSELF, which exits
#   with an error if it fails: no candidate is computed unless it passes.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 13: audio-only separability gate for arm C,
#              pre-registered criterion (2026-08-12)"
# PROVENANCE : docs/provenance/2026-08-12_exp13_stem_separability.txt (pinned)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # pinned provenance is never touched
OUT="$OUT_DIR/exp13_stem_separability.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "CANARY|PASSED|FAILED|C3 MFCC-13|VERDICT" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-12_exp13_stem_separability.txt"
