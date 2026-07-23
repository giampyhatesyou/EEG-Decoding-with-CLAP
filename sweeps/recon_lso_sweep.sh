#!/usr/bin/env bash
# CHANGED(baseline): NEW FILE — leave-song-out sweep for the stimulus-reconstruction
#   (backward-model) AAD decoder (src/stimulus_reconstruction.py). CPU-only, fast,
#   resumable. Does NOT touch the contrastive baseline. Each fold holds out one song,
#   trains subject-specific ridge decoders on the remaining training songs, and writes
#   a test_records.csv compatible with aggregate_lso.py under tag recon_lso_song<ID>.
#
# Usage (run from repo root, env active):
#   bash sweeps/recon_lso_sweep.sh                 # default representative set (one per class)
#   SONGS="2 7 44 8" bash sweeps/recon_lso_sweep.sh
#   SONGS="$(all songs)" bash sweeps/recon_lso_sweep.sh   # full LSO
# Env knobs: PY, DATASET_DIR, LOG_DIR, COMPRESSION, ENV_CUTOFF, LAGS_MS.
set -uo pipefail
cd "$(dirname "$0")/../src"

PY=${PY:-python}
MODEL=${MODEL:-ridge}                 # ridge (CPU, linear anchor) | cnn (GPU, non-linear)
DATASET_DIR=${DATASET_DIR:-"$(cd .. && pwd)/dataset"}
LOG_DIR=${LOG_DIR:-../results}
COMPRESSION=${COMPRESSION:-0.3}
ENV_CUTOFF=${ENV_CUTOFF:-20.0}
LAGS_MS=${LAGS_MS:-250}
# Default: one representative song per class (vocal=2, drum=7, bass=44, others=8).
# Expand to the full song list for the complete leave-song-out evaluation.
SONGS=${SONGS:-"2 7 44 8"}

# Separate tag namespaces so ridge/cnn folds never collide and aggregate apart.
if [ "$MODEL" = "cnn" ]; then PREFIX="recon_cnn_lso"; GPU_ENV=""; else PREFIX="recon_lso"; GPU_ENV="CUDA_VISIBLE_DEVICES="; fi

done_test(){ compgen -G "$LOG_DIR/$1/nmed-CL-*/version_*/test_breakdown_summary.txt" >/dev/null; }

echo "[recon-sweep] $(date) model=$MODEL songs=[$SONGS] dataset=$DATASET_DIR log=$LOG_DIR"
for sid in $SONGS; do
  tag="${PREFIX}_song${sid}"
  if done_test "$tag"; then echo "  skip(done): $tag"; continue; fi
  echo "===== [$(date +%H:%M)] $tag (model=$MODEL) ====="
  env $GPU_ENV $PY -u stimulus_reconstruction.py \
      --training_date "$tag" --cv_mode leave_song_out --cv_held_out_id "$sid" \
      --dataset_dir "$DATASET_DIR" --log_dir "$LOG_DIR" --recon_model "$MODEL" \
      --recon_compression "$COMPRESSION" --recon_env_cutoff_hz "$ENV_CUTOFF" --recon_lags_ms "$LAGS_MS" \
    || echo "  FOLD FAILED: $tag"
done
echo "[recon-sweep] done. Aggregate with:  python sweeps/aggregate_lso.py   (reads ${PREFIX}_* too)"
