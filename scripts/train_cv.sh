#!/bin/bash
# Train + test a single CV fold (leave-song-out or leave-subject-out).
#
# Usage:
#   bash scripts/train_cv.sh <cv_mode> <held_out_id> [<tag>]
#     cv_mode      : leave_song_out | leave_subject_out
#     held_out_id  : int (song id OR subject id depending on cv_mode)
#     tag          : optional training_date prefix; defaults to "cv"
#
# Output:
#   results/<tag>_<cv_mode>_<held_out_id>/
#     nmed-CL-preprocessing_eegmusic/version_*/checkpoints/best-checkpoint.ckpt
#     nmed-CL-preprocessing_eegmusic/version_*/test_*.png / *.csv
#
# Environment overrides (all optional):
#   AUDIO_REPR=raw|clap   audio side; defaults to clap (the headline run)
#   MAX_EPOCHS=N          override default 1000 (EarlyStopping still applies)
#   LR=...                override --learning_rate 0.003
#
# Note: every fold trains from scratch — this is the whole point of CV. On a
# single L40S a fold takes ~30–60 min; loop this script over ids for a sweep.

set -e

CV_MODE="${1:?usage: $0 <cv_mode> <held_out_id> [tag]}"
HELD_OUT="${2:?usage: $0 <cv_mode> <held_out_id> [tag]}"
TAG="${3:-cv}"

case "$CV_MODE" in
    leave_song_out|leave_subject_out) ;;
    *) echo "[train_cv.sh] cv_mode must be 'leave_song_out' or 'leave_subject_out'; got '$CV_MODE'"; exit 2 ;;
esac

# --- Activate conda env (search common locations) ---------------------------
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "eeg_attention" ]; then
    _conda_sh=""
    for _p in \
        /opt/conda/etc/profile.d/conda.sh \
        /etc/profile.d/conda.sh \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh"; do
        if [ -f "$_p" ]; then _conda_sh="$_p"; break; fi
    done
    if [ -n "$_conda_sh" ]; then
        # shellcheck source=/dev/null
        source "$_conda_sh"
        conda activate eeg_attention 2>/dev/null \
            || echo "[train_cv.sh] warning: conda env 'eeg_attention' not found"
    else
        echo "[train_cv.sh] warning: no conda init script found"
    fi
fi

unset PYTHONPATH

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT/src"

TRAINING_DATE="${TAG}_${CV_MODE}_${HELD_OUT}"
AUDIO_REPR="${AUDIO_REPR:-clap}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
LR="${LR:-0.003}"

LOG_FILE="$PROJECT_ROOT/logs/cv_${TRAINING_DATE}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[$(date +%H:%M:%S)] === CV fold: cv_mode=${CV_MODE} held_out=${HELD_OUT} audio_repr=${AUDIO_REPR} ==="
echo "[$(date +%H:%M:%S)] training_date=${TRAINING_DATE}"
echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"

# --- Phase 1: train -------------------------------------------------------
python -u main.py \
  --dataset preprocessing_eegmusic \
  --test_dataset preprocessing_eegmusic_test \
  --devices 1 \
  --max_epochs "$MAX_EPOCHS" \
  --batch_size 8 \
  --eeg_length 768 \
  --loss_function clip_loss \
  --eeg_normalization MetaAI \
  --clamp_value 20 \
  --learning_rate "$LR" \
  --supervised 1 \
  --dim_reduction 1 \
  --shifting_time 0 \
  --split_seed 42 \
  --detach_z_c 0 \
  --window_size 1280 \
  --stride 256 \
  --test_window_size 768 \
  --test_stride 256 \
  --seed 42 \
  --start_position 0 \
  --attention_values 4 5 \
  --key all \
  --audio_repr "$AUDIO_REPR" \
  --cv_mode "$CV_MODE" \
  --cv_held_out_id "$HELD_OUT" \
  --training_date "$TRAINING_DATE" 2>&1 | tee -a "$LOG_FILE"

# --- Phase 2: test on the held-out fold ------------------------------------
echo "[$(date +%H:%M:%S)] --- testing held-out fold ---" | tee -a "$LOG_FILE"
python -u checkpoint_test.py \
  --dataset preprocessing_eegmusic \
  --test_dataset preprocessing_eegmusic_test \
  --devices 1 \
  --max_epochs "$MAX_EPOCHS" \
  --batch_size 8 \
  --eeg_length 768 \
  --loss_function clip_loss \
  --eeg_normalization MetaAI \
  --clamp_value 20 \
  --learning_rate "$LR" \
  --supervised 1 \
  --dim_reduction 1 \
  --shifting_time 0 \
  --split_seed 42 \
  --detach_z_c 0 \
  --window_size 1280 \
  --stride 256 \
  --test_window_size 768 \
  --test_stride 256 \
  --seed 42 \
  --start_position 0 \
  --attention_values 4 5 \
  --key all \
  --audio_repr "$AUDIO_REPR" \
  --cv_mode "$CV_MODE" \
  --cv_held_out_id "$HELD_OUT" \
  --test_breakdown 1 \
  --training_date "$TRAINING_DATE" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date +%H:%M:%S)] === fold done: ${TRAINING_DATE} ===" | tee -a "$LOG_FILE"
