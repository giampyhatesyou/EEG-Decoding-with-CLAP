#!/usr/bin/env bash
# =============================================================================
# run_all_cheap.sh - in sequence, EVERY experiment that spends no held-out look
# and needs nothing but the dataset. Cheapest first, so a failure surfaces early.
#
# What runs, and why it is free:
#   exp13  audio-only separability        - no EEG file is ever opened      ~10 s
#   exp08  spectral flux on own-vs-other  - held-out SOLO segments only     ~5 min
#   exp14  MFCC/mel-64 on own-vs-other    - same                            ~4 min
#   exp16b CCA with band power as a view  - same                            ~7 min
#   exp06  the own-vs-other gate itself   - same; produces the project's
#          reference 208/376 and its md5, which every other row is judged against
#                                                                          ~15 min
#   exp07  own-vs-other configuration sweep - same                       ~15-40 min
#
# These six are every experiment `run.py exp --list` marks `looks = no` EXCEPT the
# five named at the bottom: 6 + 5 = 11, and the count is meant to be checked.
#
# 'Free' is defined by the ledger of held-out looks, not by runtime: own-vs-other
# runs on held-out SOLO segments (the code asserts `--train_on raw_solos`), so no
# duo is loaded and no attention decision is taken.
#
# What does NOT run, and why:
#   exp09 / exp11 / exp12 / exp15 / arm A / arm D -> they decide on the DUOS.
#     All 309 duos are already spent: rerunning them opens no new material, but
#     every duo number is EXPLORATORY BY CONSTRUCTION and must be labelled so.
#   exp16a / exp17 -> free, but they need the CLAP embedding arrays, which are
#     not in the repository (~9 min of cluster CPU to regenerate).
#   exp18 / exp19  -> free, but their gates need torch; they are GPU experiments.
#   exp05          -> free because it EXECUTES NOTHING. It is an archived
#     pre-registration, kept so that a replication directory missing Exp. 5 does not
#     read as an oversight; `exp all` still walks it, and it prints no number.
#   No script in this directory touches the TRIOS (92 stereo + 93 mono).
#
# TOTAL COST: ~45-70 min CPU, dominated by Exp. 7 (C5 and C7 run at
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
ONLY=${ONLY:-"exp13 exp08 exp14 exp16b exp06 exp07"}

# case, not an associative array: macOS ships bash 3.2, which has no `declare -A`.
script_for() {
  case "$1" in
    exp13)  echo exp13_stem_separability.sh ;;
    exp08)  echo exp08_flux_ovo.sh ;;
    exp14)  echo exp14_mfcc_tracking.sh ;;
    exp16b) echo exp16b_ccaviews.sh ;;
    exp06)  echo exp06_ovo_gate.sh ;;
    exp07)  echo exp07_ovo_config_sweep.sh ;;
    *)      echo "" ;;
  esac
}

for k in $ONLY; do
  s=$(script_for "$k")
  [ -n "$s" ] || { echo "unknown: $k (valid: exp13 exp08 exp14 exp16b exp06 exp07)"; exit 2; }
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
