#!/usr/bin/env bash
# =============================================================================
# run_all_cheap.sh - in sequence, ONLY the experiments that spend no held-out
# looks and need no GPU.
#
# What runs, and why it is free:
#   exp13  audio-only separability   - no EEG file is ever opened
#   exp07  own-vs-other sweep        - held-out SOLO segments only (asserts raw_solos)
#   exp08  spectral flux on own-vs-other  - same
#   exp14  MFCC/mel-64 on own-vs-other    - same
#
# What does NOT run, and has to be launched by hand knowing what it costs:
#   exp09 / exp11 / exp12 / exp15 / arm A / arm D  -> they decide on the DUOS.
#   All 309 duos are already spent: rerunning them opens no new material, but
#   every duo number is EXPLORATORY BY CONSTRUCTION and must be labelled as such.
#   No script in this directory touches the TRIOS (92 stereo + 93 mono).
#
# TOTAL COST: ~25-50 min CPU, dominated by Exp. 7 (C5 and C7 run at
#   --target_fs 256, ~6 GB each). See the estimates in each script's header.
#
# Every script stops on its own if its canary fails, and `set -e` here stops the
# sequence: a red canary must NOT produce the numbers that follow it.
#
#   bash scripts/replicate/run_all_cheap.sh
#   ONLY="exp13 exp14" bash scripts/replicate/run_all_cheap.sh   # a subset
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ONLY=${ONLY:-"exp13 exp07 exp08 exp14"}

# case, not an associative array: macOS ships bash 3.2, which has no `declare -A`.
script_for() {
  case "$1" in
    exp13) echo exp13_stem_separability.sh ;;
    exp07) echo exp07_ovo_config_sweep.sh ;;
    exp08) echo exp08_flux_ovo.sh ;;
    exp14) echo exp14_mfcc_tracking.sh ;;
    *)     echo "" ;;
  esac
}

for k in $ONLY; do
  s=$(script_for "$k")
  [ -n "$s" ] || { echo "unknown: $k (valid: exp13 exp07 exp08 exp14)"; exit 2; }
  echo
  echo "############################################################"
  echo "# $k  ->  $s   $(date '+%H:%M:%S')"
  echo "############################################################"
  bash "$HERE/$s"
done

echo
echo "############################################################"
echo "# all free experiments passed  $(date '+%H:%M:%S')"
echo "# no held-out looks spent / no GPU / trios untouched"
echo "############################################################"
