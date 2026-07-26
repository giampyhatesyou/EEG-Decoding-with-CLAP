#!/usr/bin/env bash
# Leave-one-SONG-out sweep across the 4 model types. RESUMABLE.
#
#   models : contrastive(raw), contrastive(clap), audio-only(classify_audio raw),
#            eeg-only(classify_eeg)
#   songs  : 5 per class (vocal/drum/bass/others), all heard by every subject
#            -> 5 x 4 = 20 folds per model, the design the thesis table reports.
#
# Resume : at start it scans the results dir and SKIPS any fold already tested (any
#          tag) -> re-running continues where it stopped. A fold cut mid-training has
#          no checkpoint, so it is simply re-done next time.
# Budget : stops LAUNCHING new folds after BUDGET seconds; each training capped by CAP.
#
# Usage (from the repo root, env eeg_attention):
#   bash sweeps/sweep_song_out.sh
#   CUDA_VISIBLE_DEVICES=0 BUDGET=21600 CAP=80m bash sweeps/sweep_song_out.sh
#   # 2 GPUs: launch twice, one per shard, with separate done-files
#   NSHARD=2 SHARD=0 DONE_FILE=/tmp/song_s0.txt bash sweeps/sweep_song_out.sh
#   NSHARD=2 SHARD=1 DONE_FILE=/tmp/song_s1.txt bash sweeps/sweep_song_out.sh
#
# The table is NOT printed here: `python sweeps/report.py` is the single aggregator.
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"   # conda env + $PY + $PROTO + $RESULTS + run_fold
cd "$(dirname "$0")/../src"                # entrypoints main.py/checkpoint_test.py live in src/
echo_env

BUDGET=${BUDGET:-23400}    # 6.5h launch cutoff
CAP=${CAP:-80m}            # per-training-fold hard cap
NSHARD=${NSHARD:-1}        # how many parallel instances split the folds
SHARD=${SHARD:-0}          # which shard this instance is (0..NSHARD-1)
DONE_FILE=${DONE_FILE:-/tmp/sweep_song_out_done.txt}

CV_FLAG="--cv_mode leave_song_out"         # consumed by run_fold
build_done_set leave_song_out "$DONE_FILE"

# 5 songs per class, all heard by every subject.
declare -A SONGS=( [vocal]="2 3 5 33 45" [drum]="7 18 59 112 145" \
                   [bass]="44 50 55 68 140" [others]="8 16 43 62 120" )
CLASSES="vocal drum bass others"
# Model priority (highest marginal value first). Fields: objective:audio_repr:tagprefix
MODELS="contrastive:raw:ct_raw contrastive:clap:ct_clap classify_audio:raw:clf_audio_raw classify_eeg:raw:clf_eeg"
NROUND=5

declare -A MC=()   # per-model fold counter, for shard balancing
for ((r=0;r<NROUND;r++)); do
  for cls in $CLASSES; do
    arr=(${SONGS[$cls]}); s=${arr[$r]:-}; [ -z "$s" ] && continue
    for m in $MODELS; do
      [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop launching"; break 3; }
      _mi=${MC[$m]:-0}; MC[$m]=$((_mi+1)); (( _mi % NSHARD != SHARD )) && continue
      obj="${m%%:*}"; rest="${m#*:}"; repr="${rest%%:*}"
      case "$obj" in
        classify_eeg)   tag="clf_eeg_lso_song$s" ;;
        classify_audio) tag="clf_audio_raw_lso_song$s" ;;
        *)              tag="${rest#*:}_lso_song$s" ;;
      esac
      run_fold "$obj" "$repr" "raw" "$s" "$tag"
    done
  done
done

echo; echo "######## DONE ########"
echo "Table: python sweeps/report.py   (pin new folds in results_manifest.tsv first)"
