#!/bin/bash
# Test script — execute from anywhere with: bash scripts/test.sh
#
# Activates the eeg_attention conda env if available (and not already active).
# Tees stdout/stderr to logs/test_<timestamp>.log.
# --checkpoint_path: remove this flag after training; auto-discovery will find
#                    the latest checkpoint in {log_dir}/{training_date}/... automatically.

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
            || echo "[test.sh] warning: conda env 'eeg_attention' not found, continuing with current Python"
    else
        echo "[test.sh] warning: no conda init script found, continuing with current Python"
    fi
fi

# --- Avoid /opt/meg-tools leaking into the env on CIMeC servers -------------
unset PYTHONPATH

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/test_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"
cd "$PROJECT_ROOT/src"

python -u checkpoint_test.py \
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
  --training_date test \
  --checkpoint_path ../checkpoints/model-all0.ckpt 2>&1 | tee "$LOG_FILE"
