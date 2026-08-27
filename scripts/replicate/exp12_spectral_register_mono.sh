#!/usr/bin/env bash
# =============================================================================
# Exp. 12 - the spectral register on the MONO duos (fresh material for the variable).
#
# WHAT IT MEASURES: the same hypothesis as Exp. 11 (which register is attended) on
#   the 42 MONO duos, which exist ONLY in the raw release and have to be
#   reconstructed from there; plus stereo-from-raw as a pipeline-comparability
#   check and the pooled set at 89 pairs.
# COST       : ~3 min CPU (estimated from the 2026-08-11 timestamps: control
#              16:24:42, primary 16:26:28). It reads the RAW release: more I/O than
#              Exp. 11.
# BUDGET     : YES - it decides on the MONO duos. All 309 duos are ALREADY SPENT:
#              exploratory by construction. The TRIOS are never touched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY    mono 22/42 = 0.5238  (p = 0.4388)
#              threshold written 2026-08-09: 27/42 = 0.6429 -> [NOT CROSSED]
#   null 0.500 EXACT by symmetry
#   SECONDARY  stereo-from-raw 24/47 = 0.5106 (threshold 30/47) - coincides with
#              Exp. 11 on the preprocessed release: the pipeline is irrelevant
#              pooled 89: 46/89 = 0.5169 (threshold 53/89)
#              dose-response: far apart 10/20 = 0.5000, close 12/22 = 0.5455
#              5/5 concentration (n=16, descriptive): 10/16 = 0.6250
#   positive control a=1.0  42/42, negative (20 seeds) 0.5226, band OK
#   sensitivity floor: a=0.20 38/42, a=0.10 28/42, a=0.05 24/42
#
# CANARY     : (a) the raw-path ALIGNMENT canary, already crossed and written into
#              the criterion (sharp peak at offset 0, destroyed by +/-0.12 s);
#              (b) the positive control with threshold >= 0.90 is run FIRST BY THE
#              DRIVER, which exits with an error if it does not pass.
# PRE-REGISTRATION: "Chapter 2 - Exp. 12: the spectral register on the MONO duos
#              (fresh material), pre-registered criterion (2026-08-11)"
# PROVENANCE : runs/results/exp12_{primary,control_a10}/ and
#              docs/provenance/2026-08-17_exp12_*_RESULT_*
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the RAW release is needed in the same dir
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # pinned records are never overwritten

SRC="$REPO/src/madeeg_spectral_attention.py"
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

mkdir -p "$OUT_DIR"
cd "$REPO"

# --source raw = Exp. 12. The default (preprocessed) is Exp. 11 and is not touched.
"$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed 42 --source raw

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
cat "$OUT_DIR/exp12_primary/summary.txt"
echo
echo "replication outputs   : $OUT_DIR/exp12_*"
echo "pinned records (2026-08-11): runs/results/exp12_*"
