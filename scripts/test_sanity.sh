#!/bin/bash
# Sanity-check sweep for the CLAP (or baseline) checkpoint.
#
# Runs `checkpoint_test.py` three times:
#   1) shuffle_test_mode=none        -> the real evaluation
#   2) shuffle_test_mode=labels      -> task labels permuted per batch;
#                                       accuracy MUST drop to chance (~0.25)
#   3) shuffle_test_mode=audio_pair  -> EEG<->audio coupling permuted per
#                                       batch; accuracy MUST drop to chance
#
# If (2) or (3) come back > 0.30 there is a measurement bug — the evaluation
# is picking up signal that does not depend on the labels / EEG-audio pairing.
#
# Each run writes into its own `training_date` so logs/checkpoints/plots stay
# side-by-side under `results/sanity_<MODE>/`.
#
# Override the checkpoint with CKPT=... and the base name with TAG=...:
#   CKPT=../checkpoints/model-all0.ckpt TAG=baseline bash scripts/test_sanity.sh

set -e

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
            || echo "[test_sanity.sh] warning: conda env 'eeg_attention' not found, continuing with current Python"
    else
        echo "[test_sanity.sh] warning: no conda init script found, continuing with current Python"
    fi
fi

# --- Avoid /opt/meg-tools leaking into the env on CIMeC servers -------------
unset PYTHONPATH

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"

# Defaults that mirror scripts/test.sh; override from the environment.
CKPT="${CKPT:-checkpoints/model-all0.ckpt}"
TAG="${TAG:-sanity}"
AUDIO_REPR="${AUDIO_REPR:-raw}"

# Resolve CKPT relative to PROJECT_ROOT (not to src/, where we cd into below)
# so the user can pass e.g. CKPT=results/clap_run1/.../best-checkpoint.ckpt
# without worrying about the working-directory shuffle inside this script.
case "$CKPT" in
    /*) ;;                                  # already absolute
    *)  CKPT="$PROJECT_ROOT/$CKPT" ;;       # treat as repo-relative
esac
if [ ! -f "$CKPT" ]; then
    echo "[test_sanity.sh] checkpoint not found: $CKPT"
    echo "[test_sanity.sh] available under results/:"
    find "$PROJECT_ROOT/results" -name "best-checkpoint.ckpt" 2>/dev/null | sed "s|^|  |"
    exit 2
fi
echo "[test_sanity.sh] using checkpoint: $CKPT"

cd "$PROJECT_ROOT/src"

for MODE in none labels audio_pair; do
    TRAINING_DATE="${TAG}_${MODE}"
    LOG_FILE="$PROJECT_ROOT/logs/test_${TRAINING_DATE}_$(date +%Y-%m-%d_%H-%M-%S).log"
    echo "[$(date +%H:%M:%S)] === shuffle_test_mode=${MODE} -> ${TRAINING_DATE} ==="
    echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"

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
      --audio_repr "$AUDIO_REPR" \
      --training_date "$TRAINING_DATE" \
      --shuffle_test_mode "$MODE" \
      --test_breakdown 1 \
      --checkpoint_path "$CKPT" 2>&1 | tee "$LOG_FILE"
done

echo ""
echo "Sanity sweep done. Inspect:"
echo "  results/sanity_none/.../test_breakdown_summary.txt        <- should look like the headline run"
echo "  results/sanity_labels/.../test_breakdown_summary.txt      <- accuracy MUST be ~0.25"
echo "  results/sanity_audio_pair/.../test_breakdown_summary.txt  <- accuracy MUST be ~0.25"
