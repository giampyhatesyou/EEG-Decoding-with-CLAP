#!/usr/bin/env bash
# =============================================================================
# Step D - the WITHIN-MIXTURE loss: aligning the objective with the task.
#          TRAINING NEEDS A GPU: this script prepares, verifies and PRINTS.
#
# WHAT IT MEASURES: step C uses BATCH NEGATIVES - a window's loss depends on
#   whatever happened to sit next to it in the batch, which is information the task
#   does not have. Step D restricts the negatives TO THE SAME MIXTURE: for every
#   window, the only negative is the other stem of the same duo. The question is
#   whether aligning the loss with the task removes the shortcut.
# DATE       : 2026-07-28.
# QUESTION   : "does aligning the loss with the task remove the shortcut?"
# NULL       : 0.500 by construction on the duo (every mixture appears with both its
#              instruments as target).
# PRE-REGISTERED THRESHOLD: 88/154 = 0.5714 - the SAME as step C, not recomputed.
# VERDICT    : PRIMARY FAILED + SECONDARY FALSIFIED IN THE OPPOSITE DIRECTION.
#              Primary 70/154 = 0.4545, 95% CI [0.374, 0.537], p 0.8867.
#              And the pre-declared secondary ("prior-following must COLLAPSE towards
#              0.4274") went the OTHER WAY: 0.7177 -> 0.8145.
#              "We removed the shortcut" was the line to write if it had collapsed:
#              it is false, and it has to be written as false.
# COST       : GPU for training. Gates on CPU: ~2 min.
# BUDGET     : YES, and it is a SECOND LOOK at the SAME 154 trials, decided AFTER
#              the step C number was known, hence EXPLORATORY BY CONSTRUCTION, and
#              the caveat is written INSIDE the run's summary (the file's first
#              4 lines). The TRIOS are never touched.
# GPU        : YES for training. This script prepares the command; it does not run it.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY             70/154 = 0.4545  95% CI [0.374, 0.537]  p 0.8867
#   POSITIVE CONTROL    142/154 = 0.9221  95% CI [0.868, 0.959]
#                       threshold 0.90 DECLARED IN THE CODE BEFORE THE RUN
#   SECONDARY 3 (verified): FIRST-epoch train_loss 0.665-0.692 against
#                       log 2 = 0.6931, the value declared IN ADVANCE - i.e. the
#                       proof that with a single negative the loss starts from the
#                       right chance level. Confirmed.
#   hyperparameters: ensemble=duo folds=5 epochs=10 lr=0.003 batch=8 seed=42
#                    loss=within_mixture
#
# THE C -> D ARITHMETIC, the most useful fact of this arm:
#      "prior RIGHT" cell   32/53 -> 44/53  (+12)
#      "prior WRONG" cell   14/71 -> 14/71  ( 0)
#    The null is 0.500 in BOTH cells. All +12 gained trials come from the cell where
#    the stimulus prior was already on the right side, and ZERO from the other. A
#    pure prior follower would score about 68/154 = 0.4416; observed 70/154. Step D
#    did not learn to decide: it learned to follow the prior better.
#
# THE TRIO GATE DID NOT OPEN. It required >= 0.5714 AND a collapsed diagnosis: both
#    failed. The 92 stereo trios + 93 mono stay intact. That is why they still exist.
#
# THE SECONDARY IS A RESULT, not a detail: it was step D's falsifiable part. That it
#    went the other way is the reason the chapter's mechanism is called
#    "prior-following" and not "noise".
#
# CANARY     : (1) `src/modules/clip_loss.py` - and the SECOND line of its self-check
#              is exactly step D's objective: within-mixture 0.951610 against batch
#              2.160834, checked against the arithmetic worked out by hand inside the
#              file. Proof that the flag REALLY CHANGES the objective (a flag that
#              changed nothing would produce a run that looks like the new arm and is
#              the old one).
#              (2) `--loss` is ASSERTED: `within-mixture` with a hyphen BREAKS.
#              (3) The self-check also verifies the PROPERTY the whole step rests on:
#              under within-mixture a window's loss is INDEPENDENT of its batch mates
#              (and under batch it is NOT - that is how we know the control
#              discriminates).
#              (4) SEPARATE directories: `madeeg_clap_kfold_within` and
#              `madeeg_clap_selftest_within`. A control never writes over a result,
#              nor does a different objective.
# PRE-REGISTRATION: "Chapter 2 - step D: within-mixture loss, pre-registered
#              criterion (2026-07-28)"
# PROVENANCE : run archive, results/madeeg_clap_{kfold,selftest}_within/
#              madeeg_contrastive_summary.txt (verified 2026-08-11);
#              C->D arithmetic in docs/provenance/2026-08-11_diagnose_step{C,D}.txt
#              NOT PINNED in results_manifest.tsv, and the reason is written in the
#              manifest: the artefacts were not on the analysis machine. It should be
#              pinned the day they land.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}

cd "$REPO"

# ---- GATE 1: the loss self-check, which INCLUDES step D's objective --------
echo "=== GATE 1 - src/modules/clip_loss.py (its 2nd line IS step D's objective) ==="
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
  OUT=$("$PY_TORCH" src/modules/clip_loss.py 2>&1); echo "$OUT"
  if echo "$OUT" | grep -q "0.951610" && echo "$OUT" | grep -q "2.160834"; then
    echo "  -> [PASSED] within-mixture 0.951610 vs batch 2.160834: the flag CHANGES the objective."
  else
    echo "  -> [FAILED] step D's two values are not the pinned ones. STOP and report."
    exit 1
  fi
else
  echo "  NOT CROSSED (no torch found). PY_TORCH=... bash \$0"
  echo "      Before training on the cluster it is MANDATORY."
fi

# ---- GATE 2: a mistyped --loss MUST break ----------------------------------
echo
echo "=== GATE 2 - a mistyped --loss must BREAK, not fall back to the default ==="
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  if "$PY_TORCH" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" \
       --loss within-mixture --smoke >/dev/null 2>&1; then
    echo "  -> [FAILED] '--loss within-mixture' (with a hyphen) was ACCEPTED."
    echo "     Such a run would look like step D and would be step C. STOP."
    exit 1
  fi
  echo "  -> [PASSED] rejected, as it must be."
else
  echo "  needs an interpreter with torch."
fi

# ---- GATE 3: there is only ONE decision rule -------------------------------
echo
echo "=== GATE 3 - --check_rule: step D's rule and step C's rule ==="
echo "    agree on 2000 random duo trials, and reproduce step C at 58/154."
REC="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC" ]; then
  "$PY" src/madeeg_diagnose.py --check_rule "$REC"
else
  echo "  MISSING $REC (archive outside the repo). Use ARCHIVE=... to point elsewhere."
fi

cat <<'EOF'

###############################################################################
# THE TRAINING - RUN IT ON THE GPU NODE. This script does not launch it.
###############################################################################
REPO=$HOME/EEG-Attention-decoding-with-CLAP
PY=$HOME/.conda/envs/eeg_attention/bin/python
MAD=$HOME/madeeg

cd $REPO && git pull && mkdir -p runs/logs

# (1) MANDATORY PRE-FLIGHT:
$PY src/modules/clip_loss.py     # 0.628491 / 1.2994 / 4.8198 / 0.951610 vs 2.160834

# (2) STEP D'S POSITIVE CONTROL. Threshold 0.90 IN THE CODE, IN ADVANCE.
#     SYNTHETIC EEG. SEPARATE directory: madeeg_clap_selftest_within/
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --self_test --loss within_mixture --kfold 5 \
    > runs/logs/stepD_selftest_within.log 2>&1 &
#     expected: 142/154 = 0.9221, 95% CI [0.868, 0.959] -> ABOVE CHANCE

# (3) THE REAL NUMBER - only after (2) has passed.
#     SEPARATE directory: madeeg_clap_kfold_within/  (never over step C)
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --loss within_mixture --kfold 5 \
    > runs/logs/stepD_kfold_within.log 2>&1 &
#     expected: 70/154 = 0.4545, 95% CI [0.374, 0.537], p 0.8867
#     threshold 88/154 = 0.5714 -> NOT DISTINGUISHABLE FROM CHANCE
#     check IN THE LOG: the FIRST epoch's train_loss must be 0.665-0.692, against
#        log 2 = 0.6931 declared in advance. That is secondary 3.

# (4) THEN, LOCALLY: the diagnostic that says what actually happened (CPU).
#     THIS is where step D falsifies itself: prior-following does NOT collapse, it
#        rises (0.7177 -> 0.8145). The primary alone would not say so.
#     There is no `diagnose_prior_stepCD.sh` script: these are two commands, and this
#        is how the two pinned files under docs/provenance/ were produced.
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv
#     -> step C: 58/154, prior-following 89/124 = 0.7177, cells 32/53 and 14/71, OR 6.20
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold_within/madeeg_contrastive_records.csv
#     -> step D: 70/154, prior-following 101/124 = 0.8145, cells 44/53 and 14/71, OR 19.90
#     pinned provenance: docs/provenance/2026-08-11_diagnose_step{C,D}.txt
EOF
