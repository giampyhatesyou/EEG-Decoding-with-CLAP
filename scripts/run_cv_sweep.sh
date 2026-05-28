#!/bin/bash
# Sequential CV sweep: trains + tests one fold per held-out id.
#
# Usage:
#   bash scripts/run_cv_sweep.sh <cv_mode> [<id1> <id2> ...]
#
# If no ids are given, defaults to:
#   leave_subject_out  -> 0 1 2 3 4 5 6 7  (all 8 subjects)
#   leave_song_out     -> the 8 most common test songs (a reasonable smoke-set;
#                         override with an explicit list for a full 63-song
#                         sweep)
#
# Designed for sequential execution on one GPU. If you want a SLURM array
# instead, dispatch `bash scripts/train_cv.sh <cv_mode> <id>` per array task.
#
# Aggregation: each fold writes its own results/cv_<mode>_<id>/ tree; once
# all folds are done you can mean/median the per-fold global accuracies
# across results/cv_<mode>_*/.../test_breakdown_summary.txt.

set -e

CV_MODE="${1:?usage: $0 <cv_mode> [<id> ...]}"
shift || true

case "$CV_MODE" in
    leave_song_out|leave_subject_out) ;;
    *) echo "[run_cv_sweep.sh] cv_mode must be 'leave_song_out' or 'leave_subject_out'"; exit 2 ;;
esac

if [ "$#" -gt 0 ]; then
    IDS=( "$@" )
elif [ "$CV_MODE" = "leave_subject_out" ]; then
    IDS=( 0 1 2 3 4 5 6 7 )
else
    # Smoke-set of held-out songs (each also lives in the legacy within-subject
    # test split, so we know they were heard by at least one subject). Replace
    # with the full 63 song ids for a final sweep.
    IDS=( 100 105 120 123 137 36 44 45 )
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${TAG:-cv}"
echo "[$(date +%H:%M:%S)] CV sweep: mode=${CV_MODE} ids=(${IDS[*]}) tag=${TAG}"

for id in "${IDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] fold start: ${CV_MODE} id=${id}"
    echo "============================================================"
    bash "$PROJECT_ROOT/scripts/train_cv.sh" "$CV_MODE" "$id" "$TAG"
done

echo ""
echo "[$(date +%H:%M:%S)] All folds done."
echo "Aggregate per-fold accuracies with e.g.:"
echo "  grep 'global accuracy' results/${TAG}_${CV_MODE}_*/nmed-CL-*/version_*/test_breakdown_summary.txt"
