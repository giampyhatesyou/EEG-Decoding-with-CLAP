#!/usr/bin/env bash
# =============================================================================
# replicate.sh — reproduce, in one command, every CHAPTER 1 number this project
# reports. Chapter 2 (MAD-EEG) is a different arm and lives in scripts/replicate/;
# `python src/run.py exp all` runs both.
# (Was reproduce_akama.sh; it now also runs the negative controls that used to
#  live in test_sanity.sh, and ends by rendering the thesis table.)
#
# Phase   what it produces                                     cost
#   within   Akama Table 1 — within-subject, from model-all0.ckpt      ~10 min CPU
#   loso     Akama Table 2 — cross-subject, from model-sub{3,7,2}.ckpt ~20 min CPU
#   sanity   the negative controls (none / labels / audio_pair)        ~15 min CPU
#   report   render RESULTS.md from results_manifest.tsv               seconds
#
# Everything runs from the authors' RELEASED checkpoints: no training, no GPU.
# Run `bash scripts/setup_checkpoints.sh` first to unpack them.
#
# Reference values.
#   Table 1 (paper): global accuracy 0.865.
#   Table 2 (paper, all-data): sub3 0.6458  sub7 0.8447  sub2 0.7763  mean 0.7556.
#   Verified here 2026-06-02 (CPU, cluster): sub3 0.6642, sub7 0.8750, sub2 0.8664
#   (mean 0.8019). sub3/sub7 land within 2-3 points and reproduce the ranking
#   reversal (sub3 best within-subject -> worst cross-subject; sub7 the opposite);
#   sub2 is ~9 points higher, so this is a CLOSE-BUT-NOT-EXACT reproduction. The
#   residual sub2 gap is most likely a test-window composition difference under
#   leave-subject-out and is not yet pinned down.
#   Negative controls: labels -> ~0.225 and audio_pair -> ~0.383, both far below
#   the 0.865 headline. `labels` at chance says the evaluation is honest; note
#   that `audio_pair` sits above 0.25 because of the majority-class prior plus
#   intact instrument slots -- that is the confound, not a measurement bug.
#
# NOTHING methodological happens here: every invocation passes $PROTO from
# sweeps/sweep_common.sh verbatim, the same flags training used.
#
# Usage
#   bash scripts/replicate.sh                          # all four phases
#   PHASES="within loso" bash scripts/replicate.sh     # a subset
#   PHASES=report bash scripts/replicate.sh            # just re-render the table
#
# Resumable: a phase whose test_breakdown_summary.txt already exists is skipped.
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/../sweeps/sweep_common.sh"   # conda env + $PY + $PROTO + $RESULTS
unset PYTHONPATH

_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHASES=${PHASES:-"within loso sanity report"}
mkdir -p "$_ROOT/runs/logs"
LOG_FILE="$_ROOT/runs/logs/replicate_$(date +%Y-%m-%d_%H-%M-%S).log"

_has()      { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
_summary()  { ls "$RESULTS"/"$1"/nmed-CL-*/version_*/test_breakdown_summary.txt 2>/dev/null | tail -1; }
_acc()      { grep -hi 'global accuracy (records, all)' "$1" 2>/dev/null | grep -oE '[0-9]*\.[0-9]+' | tail -1; }
_eval()     {   # $1 tag, rest: extra flags
  local tag="$1"; shift
  if [ -n "$(_summary "$tag")" ]; then echo "[replicate] $tag already done, skipping."; return 0; fi
  $PY -u checkpoint_test.py $PROTO --audio_repr raw --test_breakdown 1 \
      --training_date "$tag" "$@" || echo "[replicate] $tag FAILED"
}

cd "$_ROOT/src"
{
echo "========================================================================"
echo "[replicate] phases='$PHASES'  results=$RESULTS  start $(date)"
echo "========================================================================"

if _has within; then
  echo; echo "--- within-subject (Akama Table 1) from model-all0.ckpt ---"
  if [ ! -f ../checkpoints/model-all0.ckpt ]; then
    echo "[replicate] ERROR: ../checkpoints/model-all0.ckpt missing — run scripts/setup_checkpoints.sh"
  else
    _eval akama_within --cv_mode within --cv_held_out_id -1 \
          --checkpoint_path ../checkpoints/model-all0.ckpt
  fi
fi

if _has loso; then
  echo; echo "--- cross-subject LOSO (Akama Table 2) from model-sub{3,7,2}.ckpt ---"
  for _s in 3 7 2; do
    if [ ! -f "../checkpoints/model-sub${_s}.ckpt" ]; then
      echo "[replicate] ERROR: ../checkpoints/model-sub${_s}.ckpt missing — run scripts/setup_checkpoints.sh"
      continue
    fi
    echo "[replicate] === held-out subject $_s ==="
    _eval "akama_loso_sub${_s}" --cv_mode leave_subject_out --cv_held_out_id "$_s" \
          --checkpoint_path "../checkpoints/model-sub${_s}.ckpt"
  done
fi

if _has sanity; then
  echo; echo "--- negative controls on model-all0.ckpt (none / labels / audio_pair) ---"
  # Same checkpoint, same protocol, only --shuffle_test_mode changes: `labels` permutes
  # the task labels, `audio_pair` permutes the EEG<->audio coupling. If `labels` does not
  # collapse to chance the evaluation is leaking; see the header for how to read audio_pair.
  CKPT=${CKPT:-../checkpoints/model-all0.ckpt}
  if [ ! -f "$CKPT" ]; then
    echo "[replicate] ERROR: $CKPT missing — run scripts/setup_checkpoints.sh"
  else
    for MODE in none labels audio_pair; do
      _eval "sanity_${MODE}" --cv_mode within --cv_held_out_id -1 \
            --shuffle_test_mode "$MODE" --checkpoint_path "$CKPT"
    done
  fi
fi

echo; echo "========================================================================"
echo "[replicate] reproduction vs Akama et al. (2025)"
echo "========================================================================"
printf "  %-40s %-9s %-9s\n" "experiment" "paper" "this run"
_w=$(_summary akama_within); printf "  %-40s %-9s %-9s\n" "within-subject (Table 1)" "0.865" "$( [ -n "$_w" ] && _acc "$_w" || echo n/a)"
declare -A PAPER_LOSO=( [3]=0.6458 [7]=0.8447 [2]=0.7763 )
_sum=0; _n=0
for _s in 3 7 2; do
  _f=$(_summary "akama_loso_sub${_s}"); _a=$( [ -n "$_f" ] && _acc "$_f" )
  printf "  %-40s %-9s %-9s\n" "LOSO sub${_s} (Table 2, all-data)" "${PAPER_LOSO[$_s]}" "${_a:-n/a}"
  [ -n "$_a" ] && { _sum=$(awk -v s="$_sum" -v a="$_a" 'BEGIN{print s+a}'); _n=$((_n+1)); }
done
[ "$_n" -gt 0 ] && printf "  %-40s %-9s %-9s\n" "LOSO mean (Table 2, all-data)" "0.7556" \
  "$(awk -v s="$_sum" -v n="$_n" 'BEGIN{printf "%.4f", s/n}')"
for MODE in labels audio_pair; do
  _f=$(_summary "sanity_${MODE}"); _a=$( [ -n "$_f" ] && _acc "$_f" )
  printf "  %-40s %-9s %-9s\n" "negative control: ${MODE}" "$([ "$MODE" = labels ] && echo 0.225 || echo 0.383)" "${_a:-n/a}"
done

if _has report; then
  echo; echo "--- rendering RESULTS.md from results_manifest.tsv ---"
  $PY "$_ROOT/sweeps/report.py" || echo "[replicate] report.py reported bad pins (see above)"
fi

echo; echo "[replicate] per-subject / per-task CSVs sit next to each test_breakdown_summary.txt"
echo "[replicate] done $(date)"
} 2>&1 | tee "$LOG_FILE"
