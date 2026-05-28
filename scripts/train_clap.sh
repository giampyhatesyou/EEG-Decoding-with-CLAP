#!/bin/bash
# Train the CLAP variant — execute from anywhere with: bash scripts/train_clap.sh
#
# Same protocol as scripts/train.sh (same splits, same seed, same hyperparams,
# same loss) except for the audio side, which is replaced by a frozen
# LAION-CLAP encoder + a small trainable projection head (see
# src/models/clap_encoder.py for the design and forward contract).
#
# Output goes under results/<TRAINING_DATE>/ so it doesn't collide with the
# baseline (which uses --training_date test). Override TRAINING_DATE from the
# environment to label different runs, e.g.:
#   TRAINING_DATE=clap_run2_lr3e-3 bash scripts/train_clap.sh

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
            || echo "[train_clap.sh] warning: conda env 'eeg_attention' not found, continuing with current Python"
    else
        echo "[train_clap.sh] warning: no conda init script found, continuing with current Python"
    fi
fi

# --- Avoid /opt/meg-tools leaking into the env on CIMeC servers -------------
unset PYTHONPATH

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"

# Distinct training_date so we don't overwrite baseline checkpoints/logs.
TRAINING_DATE="${TRAINING_DATE:-clap_run1}"
LOG_FILE="$PROJECT_ROOT/logs/train_${TRAINING_DATE}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "[$(date +%H:%M:%S)] training_date=$TRAINING_DATE"
echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"
cd "$PROJECT_ROOT/src"

# Hyperparameters are *identical* to scripts/train.sh on purpose: keeping
# everything fixed except --audio_repr makes the CLAP vs raw comparison
# apples-to-apples on the same splits, seed, optimizer, and loss.
#
# --audio_repr clap   -> swaps the 4 SampleCNN2DEEG audio encoders for a
#                        single shared CLAPEncoder (frozen backbone + MLP head)
# --training_date     -> isolated output dir, lets us keep both runs
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
  --audio_repr clap \
  --training_date "$TRAINING_DATE" 2>&1 | tee "$LOG_FILE"
