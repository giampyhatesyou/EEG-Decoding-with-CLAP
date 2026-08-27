#!/usr/bin/env bash
# =============================================================================
# Step C - the CLAP<->EEG contrastive model with BATCH negatives, on MAD-EEG.
#          TRAINING NEEDS A GPU: this script prepares, verifies and PRINTS.
#
# WHAT IT MEASURES: whether a CLIP-style contrastive model - the same InfoNCE
#   arithmetic as Chapter 1, with a different sampler - decides which of a duo's
#   two instruments was attended. It is the direct transfer of the Chapter 1
#   architecture onto a dataset where the Chapter 1 shortcut cannot win.
# DATE       : 2026-07-27 (the valid run). An earlier run of 2026-07-26 is VOID,
#              see below.
# QUESTION   : "does the CLAP<->EEG contrastive model with batch negatives decide
#              attention?"
# NULL       : 0.500 by construction - every duo mixture appears with BOTH its
#              instruments as target (18/18), so "which stem is attended" is not a
#              property of the stimulus.
# PRE-REGISTERED THRESHOLD: 88/154 = 0.5714 (smallest integer with one-sided exact
#              binomial p < 0.05 at n = 154). Criterion of 2026-07-26, written
#              before the code.
# VERDICT    : BELOW CHANCE, AND EXPLAINED - 58/154 = 0.3766, 95% CI [0.300, 0.458],
#              one-sided p 0.9992.
#              THE CI EXCLUDES 0.50: it is systematically wrong, not random.
#              THE SIGN IS NOT FLIPPED. Inverting the decision would give 0.62, and
#              it would be a number with no hypothesis behind it. The explanation is
#              prior-following: `madeeg_diagnose.py --records`, the command at the
#              end of this script (pinned provenance
#              docs/provenance/2026-08-11_diagnose_stepC.txt).
# COST       : GPU for training. Preparation and gates on CPU: ~2 min.
# BUDGET     : YES - it decides on the duos. On 2026-07-27 it was the contrastive
#              arm's FIRST look, hence CONFIRMATORY. Today all 309 duos are spent:
#              rerunning it opens no new material. The TRIOS are never touched.
# GPU        : YES for training. This script prepares the command; it does not run it.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY             58/154 = 0.3766  95% CI [0.300, 0.458]  p 0.9992
#   POSITIVE CONTROL    148/154 = 0.9610 on SYNTHETIC EEG
#                       threshold 0.90 DECLARED IN THE CODE BEFORE THE RUN
#                       95% CI [0.917, 0.986]
#   run hyperparameters: ensemble=duo folds=5 epochs=10 lr=0.003 batch=8 seed=42
#
# RUN 1 OF 2026-07-26 IS VOID, and stays on record. It gave 61/154 = 0.3961. The
#    bug: the `stim` key was not unique across subjects, so the decisions were NOT
#    independent and margins merged across different subjects. That number goes
#    nowhere. Fixed in `69b07a6` with an assert: the decision unit is the trial, keyed
#    by `(subject, stim)`, and an assert defends it.
#
# THE POSITIVE CONTROL SAYS NOTHING ABOUT THE REAL DATA. The EEG is SYNTHETIC (a
#    fixed linear operator on the expected source's envelope, snr 4.0). The 148/154
#    says the wiring works, not that the model decodes. The caveat lives INSIDE the
#    summary file, above the numbers - it was added after a real incident
#    (`7021cec`: a summary said `ABOVE CHANCE` without saying the EEG was synthetic).
#
# STEP C DOES NOT FILTER to 1-8 Hz. The filter belongs to the reconstruction arm
#    (`madeeg_reconstruction.py`: 13 occurrences; the contrastive driver: 0). So the
#    0.3766 does NOT depend on the band choice, and the open entry about the 1-8 Hz
#    band does not touch it.
#
# CANARY     : (1) `src/modules/clip_loss.py` - the InfoNCE arithmetic is THE SAME
#              as Chapter 1's: if it moves, every Chapter 1 number moves with it.
#              (2) `--loss` is ASSERTED: a mistyped value BREAKS, it does not
#              silently fall back to the default (`batch`).
#              (3) `_self_check` and `_check_soli_row_order` inside the dataset adapter.
#              (4) `--check_rule` on the archived records -> 58/154 EXACT, agreeing
#              TRIAL BY TRIAL: the decision rule has a single implementation
#              (`trial_decision`) and can be rerun offline.
# PRE-REGISTRATION: "Chapter 2 - pre-registered criterion (2026-07-26)"
# PROVENANCE : RESULTS.md madeeg_duo section <- results_manifest.tsv, pin
#              `madeeg_clap_kfold`; records in the run archive,
#              results/madeeg_clap_{kfold,selftest}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # for the CPU gates that need no torch
PY_TORCH=${PY_TORCH:-}                        # for clip_loss.py (needs torch)
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}

cd "$REPO"

# ---- GATE 1: the loss arithmetic (the same as Chapter 1's) -----------------
echo "=== GATE 1 - src/modules/clip_loss.py (synthetic, no data, no GPU) ==="
if [ -z "$PY_TORCH" ]; then
  # The canary's REFERENCE env is conda `attention` (torch 2.2.2, numpy<2): it must be
  # searched BEFORE /opt/anaconda3 base (torch 2.11), which stays the last resort.
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "/opt/anaconda3/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/modules/clip_loss.py
  echo "  expected: 0.628491 / 1.2994 / 4.8198  and step D  0.951610 vs 2.160834"
else
  echo "  NOT CROSSED (no torch found). PY_TORCH=... bash \$0"
  echo "      This is not a pass. Before training on the cluster it is MANDATORY."
fi

# ---- GATE 2: the decision rule, on the archived records --------------------
echo
echo "=== GATE 2 - --check_rule on the step C records: must give 58/154 EXACT ==="
REC="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC" ]; then
  "$PY" src/madeeg_diagnose.py --check_rule "$REC"
else
  echo "  MISSING $REC"
  echo "      The run archive is OUTSIDE the repo. Use ARCHIVE=... to point elsewhere."
fi

# ---- GATE 3: the adapter's self-check --------------------------------------
echo
echo "=== GATE 3 - self-check of the contrastive dataset adapter ==="
# `--ensemble duo` EXPLICIT. The adapter's default is `both`, which also builds the
#    TRIO trials: those are the only remaining holdout (92 stereo + 93 mono, one shot)
#    and a smoke test does not need them. A DECLARED deviation from the documented
#    command; it cannot move any number because the smoke test produces none.
if [ -d "$MADEEG_DIR" ] && [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/datasets/madeeg_contrastive_dataset.py \
      --madeeg_dir "$MADEEG_DIR" --ensemble duo \
    || echo "  the adapter exited with an error: do NOT launch the training."
else
  echo "  needs both the dataset in $MADEEG_DIR and an interpreter with torch."
fi

cat <<'EOF'

###############################################################################
# THE TRAINING - RUN IT ON THE GPU NODE. This script does not launch it.
###############################################################################
# The cluster has no tmux: use nohup, so the web session can drop without taking the
# run with it. `conda activate` does NOT work in non-interactive shells: pass the
# interpreter BY PATH.

REPO=$HOME/EEG-Attention-decoding-with-CLAP
PY=$HOME/.conda/envs/eeg_attention/bin/python
MAD=$HOME/madeeg

cd $REPO && git pull && mkdir -p runs/logs

# (1) MANDATORY PRE-FLIGHT - the canary, with real torch:
$PY src/modules/clip_loss.py

# (2) THE POSITIVE CONTROL, BEFORE the real number. Threshold 0.90 IN THE CODE.
#     SYNTHETIC EEG: it says the wiring works, NEVER that the model decodes.
#     Writes to madeeg_clap_selftest/, a SEPARATE directory.
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --self_test --kfold 5 \
    > runs/logs/stepC_selftest.log 2>&1 &
#     expected: 148/154 = 0.9610, 95% CI [0.917, 0.986]  -> ABOVE CHANCE
#     if it does NOT pass the 0.90 threshold, the real number is not looked at. Stop.

# (3) THE REAL NUMBER - only after (2) has passed.
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --kfold 5 \
    > runs/logs/stepC_kfold.log 2>&1 &
#     expected: 58/154 = 0.3766, 95% CI [0.300, 0.458], p 0.9992
#     pre-registered threshold 88/154 = 0.5714 -> NOT DISTINGUISHABLE FROM CHANCE
#     --loss defaults to `batch`: step C IS the default, not a flag.

# WHERE EVERYTHING LANDS:
#   runs/results/madeeg_clap_kfold/madeeg_contrastive_{records.csv,summary.txt}
#   runs/results/madeeg_clap_selftest/...
#   runs/ is gitignored: bringing the records back needs an rsync, which is why the
#      step D numbers were not pinnable until 2026-08-11.

# (4) THEN, LOCALLY, the diagnostic that EXPLAINS the 0.3766 (CPU, no GPU).
#     There is no `diagnose_prior_stepCD.sh` script: the diagnostic is a single
#        command, and it is exactly what produced the pinned file
#        docs/provenance/2026-08-11_diagnose_stepC.txt (`folds=global` is the default).
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv
#     expected: 58/154 = 0.3766, prior-following 89/124 = 0.7177 with null 0.4274 (NOT 0.5),
#               cells 32/53 and 14/71, Fisher p 4.87e-06, odds ratio 6.20
EOF
