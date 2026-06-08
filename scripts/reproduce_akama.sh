#!/bin/bash
# =============================================================================
# reproduce_akama.sh — one script that reproduces the Akama et al. (2025)
# experiments end-to-end, using the paper's OWN architecture (audio_repr=raw)
# and the authors' OWN released checkpoints, so the numbers line up with the
# paper before any CLAP / audit work is layered on top.
#
# It reproduces, from the released checkpoints (fast, CPU-friendly):
#   * Table 1 — within-subject decoding.   checkpoint: model-all0.ckpt
#       paper global accuracy (all-data) = 0.865.
#   * Table 2 — cross-subject leave-one-subject-out (LOSO) on the paper's three
#       representative subjects.            checkpoints: model-sub{3,7,2}.ckpt
#       paper per-subject global accuracy (all-data):
#           sub3 0.6458   sub7 0.8447   sub2 0.7763   mean 0.7556
#
# VERIFIED (2026-06-02, CPU on baldo) reproduction of Table 2 from these
# checkpoints: sub3 0.6642, sub7 0.8750, sub2 0.8664 (mean 0.8019). sub3/sub7
# land within 2-3 points of the paper and reproduce the ranking reversal
# (sub3 best within-subject -> worst cross-subject; sub7 -> best cross-subject);
# sub2 is ~9 points higher than the paper, so this is a CLOSE-BUT-NOT-EXACT
# reproduction. The residual sub2 gap is most likely a test-set/window
# composition difference under leave-subject-out and is not yet pinned down.
#
# NOTHING methodological is changed here. Every python invocation mirrors the
# verified command in scripts/test.sh (within) / scripts/train_cv.sh (LOSO),
# with audio_repr=raw and the per-window breakdown turned on.
#
# -----------------------------------------------------------------------------
# Usage
#   bash scripts/reproduce_akama.sh                    # Table 1 + Table 2 (both from checkpoints, ~30 min CPU)
#   REPRO_PHASES="within" bash scripts/reproduce_akama.sh        # only Table 1
#   REPRO_PHASES="loso"   bash scripts/reproduce_akama.sh        # only Table 2 (from checkpoints)
#   REPRO_PHASES="loso_scratch" bash scripts/reproduce_akama.sh  # RETRAIN LOSO from scratch (~15 h)
#   REPRO_PHASES="within_scratch" bash scripts/reproduce_akama.sh# RETRAIN within from scratch (~5 h)
#
# Phases (REPRO_PHASES, space-separated; default "within loso"):
#   within          eval model-all0.ckpt          -> Table 1       (~10 min CPU)
#   loso            eval model-sub{3,7,2}.ckpt     -> Table 2       (~20 min CPU)
#   within_scratch  RETRAIN within-subject raw, then test          (~5 h, GPU)
#   loso_scratch    RETRAIN raw LOSO folds from scratch, then test (~5 h each, GPU)
#
# Resumable: a phase whose test_breakdown_summary.txt already exists is skipped.
# Requires the checkpoints: run `bash scripts/extract_checkpoints.sh` first
# (unpacks model-all0 + model-sub{2,3,7} from archive/proposed_model_checkpoints.7z).
# =============================================================================

# --- Activate conda env (same search used by the other scripts) --------------
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "eeg_attention" ]; then
    _conda_sh=""
    for _p in \
        /opt/conda/etc/profile.d/conda.sh \
        /etc/profile.d/conda.sh \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/.conda/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh"; do
        if [ -f "$_p" ]; then _conda_sh="$_p"; break; fi
    done
    if [ -n "$_conda_sh" ]; then
        # shellcheck source=/dev/null
        source "$_conda_sh"
        conda activate eeg_attention 2>/dev/null \
            || echo "[reproduce_akama] warning: conda env 'eeg_attention' not found, continuing"
    else
        echo "[reproduce_akama] warning: no conda init script found, continuing"
    fi
fi

unset PYTHONPATH
export EEG_FORCE_PROGRESS="${EEG_FORCE_PROGRESS:-0}"   # quiet per-batch bar in logs

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/reproduce_akama_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[$(date +%H:%M:%S)] Logging stdout+stderr to $LOG_FILE"

REPRO_PHASES="${REPRO_PHASES:-within loso}"

# Shared hyperparameters — identical to scripts/test.sh / scripts/train_cv.sh.
COMMON_ARGS=(
  --dataset preprocessing_eegmusic
  --test_dataset preprocessing_eegmusic_test
  --devices 1
  --max_epochs 1000
  --batch_size 8
  --eeg_length 768
  --loss_function clip_loss
  --eeg_normalization MetaAI
  --clamp_value 20
  --learning_rate 0.003
  --supervised 1
  --dim_reduction 1
  --shifting_time 0
  --split_seed 42
  --detach_z_c 0
  --window_size 1280
  --stride 256
  --test_window_size 768
  --test_stride 256
  --seed 42
  --start_position 0
  --attention_values 4 5
  --key all
  --audio_repr raw
)

cd "$PROJECT_ROOT/src"

_has_phase () { case " $REPRO_PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
_summary_for () { ls "$PROJECT_ROOT"/results/"$1"/nmed-CL-preprocessing_eegmusic/version_*/test_breakdown_summary.txt 2>/dev/null | tail -1; }
_grep_acc () { grep -hi 'global accuracy (records, all)' "$1" 2>/dev/null | grep -oE '[0-9]*\.[0-9]+' | tail -1; }

{
echo "========================================================================"
echo "[reproduce_akama] phases='$REPRO_PHASES'   start $(date)"
echo "========================================================================"

# -----------------------------------------------------------------------------
# Phase A — Table 1, within-subject, from the authors' released checkpoint.
# -----------------------------------------------------------------------------
if _has_phase within; then
    echo ""; echo "--- Phase A: within-subject (Table 1) from model-all0.ckpt ---"
    if [ ! -f ../checkpoints/model-all0.ckpt ]; then
        echo "[reproduce_akama] ERROR: ../checkpoints/model-all0.ckpt missing — run scripts/extract_checkpoints.sh"
    elif [ -n "$(_summary_for akama_within)" ]; then
        echo "[reproduce_akama] already done, skipping."
    else
        python -u checkpoint_test.py "${COMMON_ARGS[@]}" \
          --cv_mode within --cv_held_out_id -1 --test_breakdown 1 \
          --training_date akama_within \
          --checkpoint_path ../checkpoints/model-all0.ckpt \
          || echo "[reproduce_akama] Phase A FAILED"
    fi
fi

# -----------------------------------------------------------------------------
# Phase B — Table 2, leave-one-subject-out, from the authors' released
# per-subject checkpoints model-sub{3,7,2}.ckpt (the paper's three subjects).
# -----------------------------------------------------------------------------
if _has_phase loso; then
    echo ""; echo "--- Phase B: cross-subject LOSO (Table 2) from model-sub{3,7,2}.ckpt ---"
    for _s in 3 7 2; do
        _td="akama_loso_sub${_s}"
        if [ ! -f "../checkpoints/model-sub${_s}.ckpt" ]; then
            echo "[reproduce_akama] ERROR: ../checkpoints/model-sub${_s}.ckpt missing — run scripts/extract_checkpoints.sh"
            continue
        fi
        if [ -n "$(_summary_for "$_td")" ]; then
            echo "[reproduce_akama] subject $_s already done, skipping."; continue
        fi
        echo "[reproduce_akama] === LOSO held-out subject $_s ==="
        python -u checkpoint_test.py "${COMMON_ARGS[@]}" \
          --cv_mode leave_subject_out --cv_held_out_id "$_s" --test_breakdown 1 \
          --training_date "$_td" \
          --checkpoint_path "../checkpoints/model-sub${_s}.ckpt" \
          || echo "[reproduce_akama] LOSO subject $_s FAILED"
    done
fi

# -----------------------------------------------------------------------------
# Phase C (optional) — RETRAIN within-subject raw from scratch.
# -----------------------------------------------------------------------------
if _has_phase within_scratch; then
    echo ""; echo "--- Phase C: within-subject RETRAINED from scratch ---"
    if [ -z "$(_summary_for akama_within_scratch)" ]; then
        python -u main.py "${COMMON_ARGS[@]}" --cv_mode within --cv_held_out_id -1 \
          --training_date akama_within_scratch || echo "[reproduce_akama] Phase C train FAILED"
        python -u checkpoint_test.py "${COMMON_ARGS[@]}" --cv_mode within --cv_held_out_id -1 \
          --test_breakdown 1 --training_date akama_within_scratch || echo "[reproduce_akama] Phase C test FAILED"
    else echo "[reproduce_akama] already done, skipping."; fi
fi

# -----------------------------------------------------------------------------
# Phase D (optional) — RETRAIN raw LOSO folds from scratch (independent check).
# -----------------------------------------------------------------------------
if _has_phase loso_scratch; then
    echo ""; echo "--- Phase D: LOSO RETRAINED from scratch (subjects 3 7 2) ---"
    for _s in 3 7 2; do
        _td="akama_loso_scratch_leave_subject_out_${_s}"
        if [ -n "$(_summary_for "$_td")" ]; then echo "[reproduce_akama] subject $_s already done, skipping."; continue; fi
        AUDIO_REPR=raw bash "$PROJECT_ROOT/scripts/train_cv.sh" leave_subject_out "$_s" akama_loso_scratch \
            || echo "[reproduce_akama] LOSO-scratch subject $_s FAILED"
    done
fi

# -----------------------------------------------------------------------------
# Aggregation — comparison table against the paper's exact numbers.
# -----------------------------------------------------------------------------
echo ""; echo "========================================================================"
echo "[reproduce_akama] RESULTS vs Akama et al. (2025)"
echo "========================================================================"
printf "  %-40s %-9s %-9s\n" "experiment" "paper" "this run"

_w=$(_summary_for akama_within); _wacc=$( [ -n "$_w" ] && _grep_acc "$_w" )
printf "  %-40s %-9s %-9s\n" "within-subject (Table 1)" "0.865" "${_wacc:-n/a}"

# paper Table 2 per-subject global accuracy (all-data)
declare -A PAPER_LOSO=( [3]=0.6458 [7]=0.8447 [2]=0.7763 )
_sum=0; _n=0
for _s in 3 7 2; do
    _f=$(_summary_for "akama_loso_sub${_s}")
    _a=$( [ -n "$_f" ] && _grep_acc "$_f" )
    printf "  %-40s %-9s %-9s\n" "LOSO sub${_s} (Table 2, all-data)" "${PAPER_LOSO[$_s]}" "${_a:-n/a}"
    if [ -n "$_a" ]; then _sum=$(awk -v s="$_sum" -v a="$_a" 'BEGIN{print s+a}'); _n=$((_n+1)); fi
done
if [ "$_n" -gt 0 ]; then
    _mean=$(awk -v s="$_sum" -v n="$_n" 'BEGIN{printf "%.4f", s/n}')
    printf "  %-40s %-9s %-9s\n" "LOSO mean (Table 2, all-data)" "0.7556" "$_mean"
fi
echo ""
echo "[reproduce_akama] per-subject / per-task CSVs sit next to each test_breakdown_summary.txt"
echo "[reproduce_akama] done $(date)"
} 2>&1 | tee "$LOG_FILE"
