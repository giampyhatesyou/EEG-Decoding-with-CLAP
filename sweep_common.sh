#!/usr/bin/env bash
# Shared setup sourced by the sweep scripts (lso_*_sweep.sh, converged_contrastive.sh).
# Keeps the conda activation and the fixed Akama protocol in exactly one place instead
# of copy-pasted into every sweep. Source it BEFORE the per-script `cd` (it has no
# cwd dependency, and `cd`-then-source breaks when $0 is a relative path).
#
# Provides:
#   $PY     - python interpreter (override with PY=... in the environment)
#   $PROTO  - the fixed protocol flags every sweep passes verbatim. NOTE: this base
#             does NOT include --cv_mode; the leave-song-out scripts append
#             `--cv_mode leave_song_out` themselves, the spectra/converged scripts
#             pass --cv_mode per fold.
# Side effect: activates the eeg_attention conda env (no-op if already active).

if [ "${CONDA_DEFAULT_ENV:-}" != "eeg_attention" ]; then
  for c in /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
           "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && . "$c" && break
  done
  conda activate eeg_attention 2>/dev/null || true
fi
PY=${PY:-python}

PROTO="--dataset preprocessing_eegmusic --test_dataset preprocessing_eegmusic_test \
--max_epochs 1000 --batch_size 8 --eeg_length 768 --loss_function clip_loss \
--eeg_normalization MetaAI --clamp_value 20 --learning_rate 0.003 --supervised 1 \
--dim_reduction 1 --split_seed 42 --detach_z_c 0 --window_size 1280 --stride 256 \
--test_window_size 768 --test_stride 256 --start_position 0 --key all \
--attention_values 4 5 --devices 1 --shifting_time 0 --seed 42"
