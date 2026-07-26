#!/usr/bin/env bash
# Train one model on one split. Replaces train.sh / train_clap.sh / train_supervised.sh
# / train_cv.sh, which were the same 70 lines four times over with one or two flags
# different -- and four separate copies of the Akama protocol that could drift apart.
# The protocol now comes from sweeps/sweep_common.sh, the single source shared with the
# sweeps and mirrored by src/run.py's PROTOCOL.
#
# Usage (from anywhere):
#   bash scripts/train.sh                                   # baseline, within-subject
#   MODEL=clap bash scripts/train.sh                        # CLAP variant
#   MODEL=clap CV=song HELD=44 bash scripts/train.sh        # one leave-song-out fold
#   MODEL=baseline CV=subject HELD=3 TEST=1 bash scripts/train.sh   # train, then evaluate
#
#   MODEL : baseline | clap | audio_only | eeg_only
#           audio_only / eeg_only are the supervisor's diagnostic controls: a SEPARATE
#           cross-entropy module (src/modules/supervised_classification.py). The
#           contrastive loss, split and metric are untouched.
#   CV    : within | song | subject          HELD : held-out id (ignored for within)
#   TAG   : results subdirectory (default: derived from MODEL/CV/HELD)
#   TEST  : 1 to run checkpoint_test.py on the result
#
# stdout+stderr are teed to runs/logs/. main.py sees a non-TTY stdout and switches from
# the per-batch tqdm bar to one summary line per epoch.
set -uo pipefail
source "$(dirname "$0")/../sweeps/sweep_common.sh"   # conda env + $PY + $PROTO + $RESULTS
unset PYTHONPATH             # keep /opt/meg-tools off the path on the CIMeC servers

MODEL=${MODEL:-baseline}
CV=${CV:-within}
HELD=${HELD:--1}
TEST=${TEST:-0}

case "$MODEL" in
  baseline)   OBJ=""; REPR="raw"  ;;
  clap)       OBJ=""; REPR="clap" ;;
  audio_only) OBJ="--objective classify_audio"; REPR="${AUDIO_REPR:-raw}" ;;
  eeg_only)   OBJ="--objective classify_eeg";   REPR="raw" ;;
  *) echo "MODEL must be baseline|clap|audio_only|eeg_only (got '$MODEL')" >&2; exit 2 ;;
esac
case "$CV" in
  within)  CV_MODE="within"; HELD=-1 ;;
  song)    CV_MODE="leave_song_out"    ;;
  subject) CV_MODE="leave_subject_out" ;;
  *) echo "CV must be within|song|subject (got '$CV')" >&2; exit 2 ;;
esac
if [ "$CV" != "within" ] && [ "$HELD" = "-1" ]; then
  echo "CV=$CV needs HELD=<id>" >&2; exit 2
fi

_suffix=""; [ "$HELD" != "-1" ] && _suffix="_$HELD"
TAG=${TAG:-"${MODEL}_${CV}${_suffix}"}

_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$_ROOT/runs/logs"
LOG_FILE="$_ROOT/runs/logs/train_${TAG}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[train] model=$MODEL cv=$CV_MODE held=$HELD tag=$TAG"
echo "[train] logging to $LOG_FILE"
cd "$_ROOT/src"

# --workers is deliberately not passed: the YAML default (-1) triggers the auto-sizing in
# src/utils/paths.py, which leaves headroom for other users on a shared box.
{
  $PY -u main.py $PROTO --cv_mode "$CV_MODE" --cv_held_out_id "$HELD" \
      --audio_repr "$REPR" $OBJ --training_date "$TAG"
  if [ "$TEST" = "1" ]; then
    echo "[train] evaluating $TAG"
    $PY -u checkpoint_test.py $PROTO --cv_mode "$CV_MODE" --cv_held_out_id "$HELD" \
        --audio_repr "$REPR" $OBJ --training_date "$TAG" --test_breakdown 1
  fi
} 2>&1 | tee "$LOG_FILE"
