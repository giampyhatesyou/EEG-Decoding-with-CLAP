#!/usr/bin/env bash
# =============================================================================
# Exp. 11 (EXPLORATORY) - the attended spectral register, STEREO duos.
#
# WHAT IT MEASURES: whether the spatial-spectral pattern of EEG power (Morlet
#   1-40 Hz, 25 frequencies, shrinkage LDA, LOSO) says WHICH OF THE TWO REGISTERS
#   is attended, comparing the two trials of the SAME mixture (47 paired couples).
#   It is the frequency-decomposition proposal (de Vries et al. 2021) carried out
#   in full.
# COST       : ~3 min CPU (estimated from the 2026-08-11 timestamps: first control
#              15:59:38, primary 16:00:25; the optional sweep adds ~2 min).
# BUDGET     : YES - it decides on the STEREO duos. All 309 duos are ALREADY SPENT:
#              rerunning them opens no new material, but the number stays
#              EXPLORATORY BY CONSTRUCTION. The TRIOS are never touched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY    attended register 24/47 = 0.5106  (p = 0.5000)
#              threshold written in advance 30/47 = 0.6383 -> [NOT CROSSED]
#   null 0.500 EXACT by symmetry (the two trials of a pair are exchangeable)
#   positive control a=1.0  47/47, dose-response a=0.5  47/47
#   negative control (permuted labels, 20 seeds) 0.4968, band [0.40,0.60] OK
#   dose-response: far apart 11/22 = 0.5000, close 13/25 = 0.5200  (flat)
#   intensity-confound probe (n=13, descriptive): 3/13 = 0.2308
#   with CONTROL_SWEEP=1: a=0.40 47/47, 0.20 47/47, 0.10 38/47,
#              0.05 26/47, 0.02 23/47, 0.01 22/47 -> sensitivity floor at a = 0.05
#
# CANARY     : the positive control (label-linked 8-13 Hz modulation injected into
#              the REAL EEG, threshold >= 0.90) is crossed FIRST BY THE DRIVER
#              ITSELF, which exits with an error if it does not pass: the real
#              number is never computed. It also asserts n_pairs == 47.
# PRE-REGISTRATION: "Chapter 2 - Exp. 11 (EXPLORATORY): the attended spectral
#              register, pre-registered criterion (2026-08-11)"
# PROVENANCE : runs/results/exp11_{primary,control_a10,control_a05,control_sweep}/
#              and docs/provenance/2026-08-17_exp11_*_RESULT_*
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # pinned records are never overwritten
CONTROL_SWEEP=${CONTROL_SWEEP:-0}             # 1 = amplitude sweep only (sensitivity floor)

SRC="$REPO/src/madeeg_spectral_attention.py"
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ "$CONTROL_SWEEP" = 1 ]; then
  "$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
        --log_dir "$OUT_DIR" --seed 42 --control_sweep
  echo; echo "=== OUTCOME (controls only, NOT a result) ==="
  cat "$OUT_DIR/exp11_control_sweep/summary.txt"
  exit 0
fi

"$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed 42

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
cat "$OUT_DIR/exp11_primary/summary.txt"
echo
echo "replication outputs   : $OUT_DIR/exp11_*"
echo "pinned records (2026-08-11): runs/results/exp11_*"
echo "sensitivity floor (amplitude sweep): CONTROL_SWEEP=1 bash $0"
