#!/usr/bin/env bash
# =============================================================================
# Exp. 14 - the TRACKING side of the gate: MFCC-13 and mel-64 on own-vs-other.
#
# WHAT IT MEASURES: whether the two representations promoted by the audio-only gate
#   of Exp. 13 are also TRACKED by the EEG, i.e. whether discarding energy (MFCC
#   excludes coefficient 0) costs tracking. Pre-registered bar: >= 208/376.
# COST       : ~3-4 min CPU (measured from the 2026-08-12 timestamps: gate 12:49:33
#              -> last run 12:52:55 = 3m22s for 5 runs).
# BUDGET     : NO - own-vs-other runs on the SOLOS, training material declared free
#              by the held-out-look accounting. No duo, no trio.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   T1 MFCC-13  214/376 = 0.5691  exact p 0.004224  -> TRACKS
#   T2 mel-64   212/376 = 0.5638  exact p 0.007623  -> TRACKS
#   mel-8 (ref) 208/376 = 0.5532  (= the bar itself)   null 0.500 exact
#   OUTCOME: branch 1 of the criterion - both track; the arm C front end is named.
#   But the descriptive McNemar does NOT distinguish them from mel-8.
#   band_pearson(own): MFCC 0.0165, mel-64 0.0341, mel-8 0.0360 (the scale
#   contraction Exp. 13 had left open).
#
# CANARIES   : three, all INSIDE the driver and crossed BEFORE the real numbers;
#              the driver exits with an error if one fails.
#              G1 regression: --target mel --n_mels 8 -> 208/376 with md5
#                 2eaa926244de340d31907c6deeb04b0c (identical to ovo_ridge_ica).
#              G2 representation identity with Exp. 13's candidate C3:
#                 maximum difference 0.000e+00.
#              G3 synthetic positive control per candidate, threshold >= 0.95:
#                 1.0000 for both.
# PRE-REGISTRATION: "Chapter 2 - Exp. 14: the tracking side of the gate - MFCC and
#              mel-64 on own-vs-other, pre-registered criterion (2026-08-12)"
# PROVENANCE : docs/provenance/2026-08-12_exp14_mfcc_tracking.txt (pinned),
#              runs/results/exp14_*/
#
# The driver does NOT accept --log_dir: it rewrites runs/results/exp14_* (those are
#    this experiment's own records, and the gates stop first if something is off).
#    The PINNED provenance file is not touched: --out goes to $OUT_DIR. If the run
#    records are needed intact too, copy the directories first.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp14_mfcc_tracking.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp14_tracking.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "PASSED|FAILED|TRACKS|does not|BRANCH|VERDICT" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-12_exp14_mfcc_tracking.txt"
