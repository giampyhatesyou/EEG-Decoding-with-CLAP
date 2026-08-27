#!/usr/bin/env bash
# =============================================================================
# Chapter 1 - WITHIN-SPLIT (same songs in train and test) + the negative
#             controls + the audio-only leakage control.
#
# WHAT IT MEASURES: whether the Akama EEG<->audio contrastive model reproduces
#   the published baseline when the SAME songs appear in training and test; and,
#   next to it, whether the evaluation is honest (permuted labels -> chance) and
#   whether the audio ALONE already identifies the target (0.996 / 0.967, which
#   is the confound this chapter is about).
# DATE       : within pinned 2026-07-18; negative controls 2026-07-18.
# QUESTION   : "does the model reproduce the published baseline on the same
#              songs?" and, immediately after, "what part of the accuracy does
#              NOT come from the EEG?"
# NULL       : 0.25 - assumed from the 4-fixed-instrument-slot design, not measured.
# PRE-REGISTERED THRESHOLD: none. This is not a test, it is a REPRODUCTION, and
#              the criterion was set by the paper (0.865). The only written
#              threshold is the diagnostic 0.30 inside the negative-control script.
# VERDICT    : reproduced - GLOBAL 0.8650 EXACT from the checkpoint.
# COST       : ~25 min CPU (within ~10' + sanity ~15'). No training.
# BUDGET     : n/a - the held-out-look budget is Chapter 2 accounting (the MAD-EEG
#              duos). Chapter 1 runs on the Akama dataset.
# GPU        : NO for the pinned part (everything comes from the checkpoints
#              RELEASED by the authors). YES for the training block at the end
#              (audio_only / eeg_only have no released checkpoints).
#
# THE MANDATORY SENTENCE NEXT TO THIS NUMBER:
#    within = the confounded split. The same songs are in train and in test, so
#    HIGHER = MEMORISES BETTER, not "decodes better". The within number must
#    always be quoted next to leave-song-out (0.268) and the audio-only control
#    (0.996): the ORDER of those three numbers is the chapter's result.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   contrastive_raw   within   MACRO 0.875   GLOBAL 0.865   (pin `sanity_none`,
#                              ckpt model-all0, 1 fold, vintage 2026-07)
#                              Deterministic: 0.8650 is EXACT, bit for bit.
#   control `labels`           MACRO 0.227   GLOBAL 0.225
#   control `audio_pair`       MACRO 0.339   GLOBAL 0.383
#   THE TWO CONTROLS ARE NOT BIT-REPRODUCIBLE (manifest, row "NOT REPRODUCIBLE
#      BIT-FOR-BIT"): `checkpoint_test` forces `shuffle=True` when
#      shuffle_test_mode != none, and `--workers` auto-sizes to the machine, so
#      the permutation changes between runs. Rerun on 2026-07-26 they gave 0.2500
#      and 0.3525. Cite them as "at chance" and "far below 0.865", NEVER as
#      exact digits.
#   [GPU] audio_only_clap within  MACRO 0.996  GLOBAL 0.993   <- the confound
#   [GPU] audio_only_raw  within  MACRO 0.967  GLOBAL 0.963
#   [GPU] eeg_only        within  MACRO 0.250  GLOBAL 0.471
#   [GPU] contrastive_clap within MACRO 0.946  GLOBAL 0.935   (8 seeds)
#
# OPEN NUMBER CONFLICT: the pair 0.865/0.852 also circulates for this same row.
#    The canonical source is `results_manifest.tsv` -> `RESULTS.md`, i.e. 0.875
#    MACRO / 0.865 GLOBAL. The entry has to be closed in writing, not resolved
#    silently by whoever writes it up.
#
# CANARY     : (1) `python src/modules/clip_loss.py` - the InfoNCE arithmetic
#              that holds up EVERY Chapter 1 number (0.628491 / 1.2994 / 4.8198).
#              It needs torch: if the interpreter does not have it, the script
#              says so and does NOT pretend the canary was crossed.
#              (2) `python src/run.py --selftest` - the 23 protocol flags live in
#              two places (PROTOCOL in run.py and $PROTO in
#              sweeps/sweep_common.sh) and must agree flag by flag.
#              (3) downstream, `report.py` checks EVERY pin against its run's
#              hparams.yaml: if one disagrees, RESULTS.md is not written at all.
# PRE-REGISTRATION: none - Chapter 1 is a reproduction, not a test.
# PROVENANCE : RESULTS.md within and control sections <- results_manifest.tsv
#              (pins `sanity_none`, `sanity_labels`, `sanity_audio_pair`)
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}          # stdlib-only canaries (selftest/report)
PY_TORCH=${PY_TORCH:-}                        # torch env: conda `attention` on macOS
PHASES=${PHASES:-"within sanity report"}      # subset of replicate.sh phases

cd "$REPO"

# ---- CANARY 1: the loss arithmetic, before anything else -------------------
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
  echo "  expected: 0.628491 / 1.2994 / 4.8198  and step D  0.951610 vs 2.160834"
else
  echo "  NOT CROSSED: no interpreter with torch found."
  echo "      PY_TORCH=~/miniconda3/envs/attention/bin/python bash \$0"
  echo "      This is NOT a pass: it is a canary that was not crossed."
fi

# ---- CANARY 2: the protocol lives in two places and they must agree --------
echo
echo "=== CANARY 2 - src/run.py --selftest (23 protocol flags) ==="
"$PY" src/run.py --selftest

# ---- THE REPRODUCTION ------------------------------------------------------
# replicate.sh is resumable: a phase whose test_breakdown_summary.txt already
# exists is SKIPPED. To really redo it, remove its directory under runs/results/
# (which is gitignored: nothing tracked is being deleted).
echo
echo "=== THE REPRODUCTION - from the RELEASED checkpoints, no training ==="
if [ ! -f "$REPO/checkpoints/model-all0.ckpt" ]; then
  echo "checkpoints/model-all0.ckpt is missing."
  echo "  bash scripts/setup_checkpoints.sh     # unpacks them from archive/*.7z"
  exit 1
fi
# replicate.sh sources sweeps/sweep_common.sh, which activates the conda env
#    `eeg_attention` - the name the env has ON THE CLUSTER. On macOS the Chapter 1
#    env is called `attention`: there `conda activate eeg_attention` is a no-op and
#    $PY falls back to bare `python`. Always pass the interpreter by path:
#      PY=~/miniconda3/envs/attention/bin/python PHASES="within sanity" bash scripts/replicate.sh
# `env`, not a bare prefix: `${PY_TORCH:+PY=...}` is EXPANDED after bash has
#    already decided what is an assignment, so it would end up being the NAME of the
#    command ("PY=/path/python: command not found") and with set -e the script would
#    die right here.
env PHASES="$PHASES" ${PY_TORCH:+PY="$PY_TORCH"} bash scripts/replicate.sh

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
# report.py --check exits 1 if even ONE pin is unverifiable (in a clean clone,
# without runs/, they all are). With `set -e` + pipefail a pipe would kill the script
# BEFORE printing the table and the reason: capture the exit code and declare it.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## within/,$p'
if [ "$RC" != 0 ]; then
  echo "report.py --check exited $RC: one or more pins are not verifiable on this"
  echo "   machine (runs/ is gitignored and folds live where they were run). The"
  echo "   [BAD PIN] lines are above. This is NOT a green light: an unverifiable pin"
  echo "   is not a pin."
fi
echo
echo "provenance: RESULTS.md within and control sections <- results_manifest.tsv"
echo "within = the confounded split: higher = MEMORISES better, not decodes better."

cat <<'EOF'

###############################################################################
# THE REST OF THE `within` TABLE REQUIRES TRAINING (GPU)
###############################################################################
# There are no released checkpoints for audio_only / eeg_only / contrastive_clap:
# those RESULTS.md rows come from our own runs, and redoing them means training.
#
#   audio_only within = THE LEAKAGE CONTROL, and it is the proof of the Chapter 1
#      confound: audio alone reaches 0.996 (clap) / 0.967 (raw). The song
#      identifies the answer. Quote it NEXT TO 0.865, never after it.

cd $REPO
python src/run.py train --model audio_only --cv within --test --tag clf_audio_raw_within
python src/run.py train --model eeg_only   --cv within --test --tag clf_eeg_within
python src/run.py train --model clap       --cv within --test --tag clapseed_within_42

# audio_only with CLAP audio (the 0.996 row) goes through main.py, because
# run.py's `--model audio_only` pins audio_repr=raw:
python src/main.py --objective classify_audio --audio_repr clap --cv_mode within \
    --cv_held_out_id -1 --training_date clf_audio_clap_within   # + $PROTO
#   ^ $PROTO = the 23 flags in sweeps/sweep_common.sh. run.py emits them itself;
#     by hand, copy them FROM THERE, do not retype them from memory.

# Then, to get the fold into the table: add a row to results_manifest.tsv and
# re-render. report.py verifies the pin against the run's hparams.yaml and
# REFUSES to write if it disagrees.
python src/run.py report
EOF
