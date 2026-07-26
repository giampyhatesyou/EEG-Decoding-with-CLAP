#!/usr/bin/env bash
# Leave-one-SUBJECT-out sweep across the 4 model types. RESUMABLE.
#
# The cross-SUBJECT counterpart of sweep_song_out.sh, and the reason the pair matters:
# the same model generalises to unseen *people* but not to unseen *songs*, which is the
# cleanest statement of the dataset confound. Until 2026-07-26 no committed script
# produced these folds -- the ones on disk came from uncommitted May runs, which is why
# they carry a different code vintage (see results_manifest.tsv, column `vintage`).
#
#   subjects : 0..7 (8 subjects; verified from the `subject` column of the within-subject
#              test records, not assumed)
#   models   : contrastive(raw), contrastive(clap), audio-only, eeg-only -> 8 folds each
#
# Resume : skips any fold already tested (any tag). Budget/CAP as in the song sweep.
#
# Usage (from the repo root, env eeg_attention):
#   bash sweeps/sweep_subject_out.sh
#   CUDA_VISIBLE_DEVICES=0 BUDGET=21600 CAP=80m bash sweeps/sweep_subject_out.sh
#   MODELS="contrastive:clap:ct_clap" bash sweeps/sweep_subject_out.sh   # one model only
#   NSHARD=2 SHARD=0 DONE_FILE=/tmp/subj_s0.txt bash sweeps/sweep_subject_out.sh
#
# The table is NOT printed here: `python sweeps/report.py` is the single aggregator.
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"   # conda env + $PY + $PROTO + $RESULTS + run_fold
cd "$(dirname "$0")/../src"                # entrypoints main.py/checkpoint_test.py live in src/
echo_env

BUDGET=${BUDGET:-23400}
CAP=${CAP:-80m}
NSHARD=${NSHARD:-1}
SHARD=${SHARD:-0}
DONE_FILE=${DONE_FILE:-/tmp/sweep_subject_out_done.txt}

CV_FLAG="--cv_mode leave_subject_out"      # consumed by run_fold
build_done_set leave_subject_out "$DONE_FILE"

SUBJECTS=${SUBJECTS:-"0 1 2 3 4 5 6 7"}
# Same field layout as the song sweep: objective:audio_repr:tagprefix
MODELS=${MODELS:-"contrastive:raw:ct_raw contrastive:clap:ct_clap classify_audio:raw:clf_audio_raw classify_eeg:raw:clf_eeg"}

declare -A MC=()   # per-model fold counter, for shard balancing
for subj in $SUBJECTS; do
  for m in $MODELS; do
    [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop launching"; break 2; }
    _mi=${MC[$m]:-0}; MC[$m]=$((_mi+1)); (( _mi % NSHARD != SHARD )) && continue
    obj="${m%%:*}"; rest="${m#*:}"; repr="${rest%%:*}"
    tag="${rest#*:}_lsubj_sub$subj"
    run_fold "$obj" "$repr" "raw" "$subj" "$tag"
  done
done

echo; echo "######## DONE ########"
echo "Table: python sweeps/report.py   (pin new folds in results_manifest.tsv first)"
