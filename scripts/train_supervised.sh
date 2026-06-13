#!/bin/bash
# Train a supervised unimodal baseline (supervisor experiments #1 / #2).
# Execute from anywhere with:
#
#   OBJECTIVE=classify_eeg                  bash scripts/train_supervised.sh   # #2 EEG-only
#   OBJECTIVE=classify_audio AUDIO_REPR=raw bash scripts/train_supervised.sh   # #1 audio-only (control)
#   OBJECTIVE=classify_audio AUDIO_REPR=clap bash scripts/train_supervised.sh  # #1 with CLAP stems
#
# This is scripts/train.sh with the same fixed Akama protocol flags, plus
# --objective (and --audio_repr). The objective selects a SEPARATE LightningModule
# (src/modules/supervised_classification.py) with a cross-entropy loss; the
# contrastive loss/metric/split are not touched. classify_eeg uses no audio
# encoder, so AUDIO_REPR is pinned to raw to avoid loading CLAP.
#
# NOTE on classify_audio: it is a NEGATIVE CONTROL — the classifier sees the
# song's four stems with no cue about what was attended, so it is expected to sit
# near chance (~25%). Above-chance accuracy quantifies song->label leakage, not
# performance. See docs/METHODOLOGY.md.

OBJECTIVE="${OBJECTIVE:-classify_eeg}"
AUDIO_REPR="${AUDIO_REPR:-raw}"
if [ "$OBJECTIVE" = "classify_eeg" ]; then
    AUDIO_REPR="raw"   # no audio encoder is used; raw avoids loading CLAP
fi
if [ "$OBJECTIVE" != "classify_eeg" ] && [ "$OBJECTIVE" != "classify_audio" ]; then
    echo "[train_supervised.sh] error: OBJECTIVE must be classify_eeg or classify_audio (got '$OBJECTIVE')" >&2
    exit 2
fi

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
        if [ -f "$_p" ]; then
            _conda_sh="$_p"
            break
        fi
    done
    if [ -n "$_conda_sh" ]; then
        # shellcheck source=/dev/null
        source "$_conda_sh"
        conda activate eeg_attention 2>/dev/null \
            || echo "[train_supervised.sh] warning: conda env 'eeg_attention' not found, continuing with current Python"
    else
        echo "[train_supervised.sh] warning: no conda init script found, continuing with current Python"
    fi
fi

# --- Avoid /opt/meg-tools leaking into the env on CIMeC servers -------------
unset PYTHONPATH

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/train_${OBJECTIVE}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[$(date +%H:%M:%S)] objective=$OBJECTIVE audio_repr=$AUDIO_REPR"
echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"
cd "$PROJECT_ROOT/src"

# Run label: clf_eeg_within / clf_audio_<repr>_within (matches run.py's suggest_tag).
if [ "$OBJECTIVE" = "classify_eeg" ]; then
    TRAINING_DATE="${TRAINING_DATE:-clf_eeg_within}"
else
    TRAINING_DATE="${TRAINING_DATE:-clf_audio_${AUDIO_REPR}_within}"
fi

python -u main.py \
  --dataset preprocessing_eegmusic \
  --test_dataset preprocessing_eegmusic_test \
  --devices 1 \
  --max_epochs 1000 \
  --batch_size 8 \
  --eeg_length 768 \
  --loss_function clip_loss \
  --eeg_normalization MetaAI \
  --clamp_value 20 \
  --learning_rate 0.003 \
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
  --objective "$OBJECTIVE" \
  --training_date "$TRAINING_DATE" 2>&1 | tee "$LOG_FILE"
