#!/usr/bin/env bash
# =============================================================================
# Chapter 1 - LEAVE-SUBJECT-OUT: is the collapse per SUBJECT or per SONG?
#
# WHAT IT MEASURES: the same architecture with a SUBJECT held out instead of a
#   song. This is the contrast that identifies the cause of the Chapter 1
#   collapse: leave-subject-out 0.943 against leave-song-out 0.258/0.268. If the
#   model generalises across people but not across songs, what it learned is the
#   SONG.
# DATE       : folds from 2026-05, verified 2026-07-23/26; raw branch pinned 2026-07.
# QUESTION   : "is the collapse per subject or per song?"
# NULL       : 0.25 (4 fixed instrument slots, assumed from the design).
# PRE-REGISTERED THRESHOLD: none (reproduction, not test).
# VERDICT    : closed - 0.943 over 6 folds (clap), 0.781 over 3 folds (raw).
# COST       : reproducing Akama's Table 2 from the released checkpoints is
#              ~20 min CPU and is what this script does. Redoing the folds is GPU.
# BUDGET     : n/a (held-out-look accounting belongs to Chapter 2, not this dataset).
# GPU        : NO for the checkpoint reproduction; YES to redo the folds.
#
# EXPECTED REFERENCE NUMBERS
#
#   (A) RESULTS.md subject_out section - the two rows used in the write-up:
#       contrastive_clap  6 folds  MACRO 0.943  GLOBAL 0.946  vintage 2026-05
#       contrastive_raw   3 folds  MACRO 0.781  GLOBAL 0.801  vintage 2026-07
#                                  (pins repro_loso_sub{2,3,7})
#
#   (B) Reproduction of Akama's Table 2 (all-data), from their checkpoints -
#       this is what `scripts/replicate.sh` prints at the end:
#         paper  sub3 0.6458 / sub7 0.8447 / sub2 0.7763 / mean 0.7556
#         ours   sub3 0.6642 / sub7 0.8750 / sub2 0.8664 / mean 0.8019
#       sub3 and sub7 land within 2-3 points and reproduce the RANK INVERSION
#       (sub3 best within -> worst cross-subject; sub7 the opposite). sub2 is
#       ~9 points above: CLOSE BUT NOT EXACT reproduction, and the residual is
#       NOT yet explained (hypothesis in the replicate.sh header: composition of
#       the test windows under leave-subject-out).
#
# TWO MANDATORY CAVEATS:
#   (a) 5 of the 6 folds behind the 0.943 row are PRE-REFACTOR (vintage 2026-05).
#       The 0.943 vs 0.258 contrast compares TWO CODE ERAS. Quote it stating the
#       vintage, not as if it were the same pipeline. Folds 6 and 7 exist but
#       produced no `test_records.csv`: it is 6 folds, not 8, and that is why.
#   (b) OPEN STALENESS: 0.607 still circulates as the "unreliable" number of the
#       raw branch. The current pin is 0.781 (repro_loso_sub{2,3,7}). The entry
#       has to be closed in writing.
#
# TAG DISCREPANCY, verified 2026-08-15, still to be resolved:
#    `scripts/replicate.sh` phase `loso` writes the tags akama_loso_sub{3,7,2},
#    while `results_manifest.tsv` pins repro_loso_sub{2,3,7}. So relaunching
#    replicate.sh does NOT update the pinned row: it produces a PARALLEL series.
#    The manifest header says the opposite ("the latter is what the committed
#    scripts/replicate.sh produces") and is STALE. Neither the tags nor the pins
#    were changed here: changing them would move a number, and that is not this
#    script's job. The two options, both still open:
#      1. rename the tags in replicate.sh to `repro_loso_sub{N}` (the number does
#         not change, but the run has to be redone to fill them);
#      2. re-pin the manifest onto `akama_loso_sub{N}` AFTER verifying they give
#         the same figures (report.py verifies the pin, not the equality).
#    `raw_loso_sub7` (0.408) exists on disk and CONFLICTS with repro_loso_sub7
#       (0.875) on the same fold: it is declared NOT pinned in the manifest.
#
# CANARY     : (1) `src/modules/clip_loss.py`; (2) `src/run.py --selftest`;
#              (3) `report.py` checks every pin against the hparams.yaml, and the
#              fold key includes `cv_mode` - because a leave_subject_out has
#              already shadowed a leave_song_out with the same numeric id
#              (fixed in 10cdc49).
# PRE-REGISTRATION: none (reproduction).
# PROVENANCE : RESULTS.md subject_out section <- results_manifest.tsv;
#              scripts/replicate.sh header for table (B)
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}
PHASES=${PHASES:-"loso report"}

cd "$REPO"

echo "=== CANARY 1 - src/modules/clip_loss.py (synthetic, no data) ==="
if [ -z "$PY_TORCH" ]; then
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/modules/clip_loss.py
else
  echo "  NOT CROSSED (no torch). PY_TORCH=... bash \$0"
fi

echo
echo "=== CANARY 2 - src/run.py --selftest ==="
"$PY" src/run.py --selftest

echo
echo "=== (A) THE PINNED NUMBERS (no run: they are re-read) ==="
# report.py --check exits 1 if even ONE pin is unverifiable: with `set -e` + pipefail
# a pipe would kill the script instead of saying so. Capture the exit code and declare it.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## subject_out/,/^## within/p'
if [ "$RC" != 0 ]; then
  echo "report.py --check exited $RC - pins not verifiable on this machine:"
  echo "$CHECK_OUT" | grep "^\[BAD PIN\]" || true
fi

echo
echo "=== (B) REPRODUCTION OF AKAMA'S TABLE 2, from the released checkpoints ==="
if [ ! -f "$REPO/checkpoints/model-sub3.ckpt" ]; then
  echo "checkpoints model-sub{3,7,2}.ckpt are missing."
  echo "  bash scripts/setup_checkpoints.sh"
  exit 1
fi
# replicate.sh sources sweep_common.sh, which activates `eeg_attention` (the env
#    name ON THE CLUSTER). On macOS the Chapter 1 env is `attention`: pass PY by path.
# `env`, not a bare prefix: `${PY_TORCH:+PY=...}` is EXPANDED after bash has already
#    decided what is an assignment, so it would become the NAME of the command
#    ("PY=/path/python: command not found") and with set -e the script would die here.
env PHASES="$PHASES" ${PY_TORCH:+PY="$PY_TORCH"} bash scripts/replicate.sh

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
echo "the paper-vs-ours table is printed by replicate.sh above."
echo "the tags written are akama_loso_sub{3,7,2}, NOT the pins repro_loso_sub{2,3,7}:"
echo "   see TAG DISCREPANCY in this script's header."

cat <<'EOF'

###############################################################################
# REDOING THE FOLDS REQUIRES A GPU
###############################################################################
cd $REPO
python src/run.py sweep --cv subject                  # 4 models x 8 subjects
CUDA_VISIBLE_DEVICES=0 CAP=80m bash sweeps/sweep_subject_out.sh    # explicit equivalent

# a single fold, to close a gap (e.g. folds 6 and 7 with no test_records.csv):
python src/run.py train --model clap --cv subject --held 6 --test --tag cv_sweep_leave_subject_out_6

# nohup, not tmux (the cluster has no tmux):
cd $REPO && nohup python src/run.py sweep --cv subject > runs/logs/lso_subject.log 2>&1 &
python src/run.py report
EOF
