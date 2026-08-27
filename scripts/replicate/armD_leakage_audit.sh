#!/usr/bin/env bash
# =============================================================================
# Arm D - per-window leakage audit on MAD-EEG.
#
# WHAT IT MEASURES: how much of the accuracy at 1 s windows comes from TRIAL
#   FINGERPRINTING rather than from attention. An LDA on log-variance in 5 bands
#   decodes PSEUDO-RANDOM per-trial labels: under per-window CV (windows of the same
#   trial straddling the split) against per-trial CV. It is a claim about the
#   PROTOCOL, not about the EEG.
# COST       : ~2-3 min CPU (estimate; one pass over 8 subjects, LDA on 1 s windows).
#              TO BE CONFIRMED on first run: no measured timing on record.
# BUDGET     : YES - it reads the stereo duo trials (EEG + true labels). Duos all
#              already spent: EXPLORATORY BY CONSTRUCTION. TRIOS untouched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PSEUDO labels (balanced): window-CV 0.6209   against null 0.5136
#                             trial-CV  0.4847   against null 0.500 exact
#   TRUE labels:              window-CV 0.3803   against null 0.2277
#                             trial-CV  0.0895   (majority-class mean)
#   OUTCOME: [LEAKAGE DEMONSTRATED] - rule declared IN THE CODE before the run:
#   leakage if pseudo window-CV > 0.60 AND pseudo trial-CV in [0.40, 0.60].
#
#   THE NULL PAIRS WITH THE UNIT OF THE ACCURACY NEXT TO IT. Under per-window CV the
#   null is 0.5136 by construction and not 0.500, because trials contribute unequal
#   numbers of windows; under per-trial CV it is 0.500 exact. Quoting one null for
#   both accuracies is the error this line exists to prevent.
#
# CANARY     : the PSEUDO-LABEL arm IS the audit's positive control (labels with ZERO
#              attentional content: any accuracy above its null is pure fingerprint).
#              The null is PRINTED next to it: that is how the mis-calibrated v1
#              (unbalanced pseudo-labels, null 0.64) was caught - kept on record in
#              runs/results/armD_leakage_audit/*_v1_unbalanced.*
#              If the printed null is not ~0.5136, do NOT read the number.
# PRE-REGISTRATION: interpretation rule written in the code before the run.
# PROVENANCE : runs/results/armD_leakage_audit/leakage_audit_summary.txt +
#              leakage_audit_per_subject.csv, and
#              docs/provenance/2026-08-17_armD_leakage_audit_RESULT_*
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # the pinned record is never overwritten
SEED=${SEED:-42}

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_leakage_audit.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed "$SEED"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
cat "$OUT_DIR/armD_leakage_audit/leakage_audit_summary.txt"
echo
echo "replication outputs : $OUT_DIR/armD_leakage_audit/"
echo "pinned record (2026-08-11): runs/results/armD_leakage_audit/"
