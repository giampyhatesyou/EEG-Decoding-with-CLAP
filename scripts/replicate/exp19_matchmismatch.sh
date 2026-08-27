#!/usr/bin/env bash
# =============================================================================
# Exp. 19 - match-mismatch at convergence (S1) and the instrument stage
#           (S2, EXPLORATORY).
#
# WHAT IT MEASURES: nothing by itself. It crosses the criterion's FOUR CPU GATES
#   (Chapter 1 canary, S1 sampling reproduction, S2 null RE-MEASURED after
#   balancing, optimisation sanity) and then PRINTS the two training commands.
#   Training does NOT start: it is a GPU step.
# COST       : ~4 min CPU measured.
# BUDGET     : NO. Only the 105 `solo` trials. No duo, no trio, no split by
#              genre/melody/instrument/subject. The TRIOS are never touched.
# GPU        : yes, for training only - the two commands are printed, not launched.
#
# WHAT CHANGES RELATIVE TO EXP. 18, AND ONLY THIS:
#   (1) the COMPUTE BUDGET: cap 120 epochs, early stopping on the TRAIN loss
#       (patience 15, min delta 0.002). The held-out bar stays 0.70. The budget
#       changes, NOT the criterion.
#   (2) the NULL REPAIR for S2 (--balance_pairs), which is not a hyperparameter.
#   NOTHING ELSE: learning rate 0.003, batch 8, temperature 0.5, seed 42,
#   architecture and encoders are those of steps C/D and Exp. 18. One variable at
#   a time.
#
# ENVIRONMENT TRAP, the same as Exp. 18: this needs `torch`, which is NOT in
#    `/opt/miniconda3`. It uses `/opt/anaconda3` (torch 2.11.0).
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   GATE 1 - Chapter 1 canary (`src/modules/clip_loss.py`, synthetic, seed 0):
#     0.628491 / 1.2994 / 4.8198   and step D  0.951610  vs  2.160834
#   GATE 2 - S1 sampling UNCHANGED with respect to Exp. 18:
#     3346 pairs / negative from the SAME wav 3346/3346 / minimum offset 2.000 s /
#     same instant 0/3346 / NULL 0.5000 / and balancing is a NO-OP
#     (3346 -> 3346, 2810 -> 2810, 536 -> 536, all nulls 0.5000)
#   GATE 3 - S2 null RE-MEASURED after balancing:
#     held-out  0.6061 -> 0.5000 (rule by candidate identity)
#               0.8898 -> 0.5000 (same rule, also conditioning on the SUBJECT)
#     cost: held-out 853 -> 188 pairs (13 of the 21 recordings), train 3561 -> 2840
#   GATE 4 - optimisation sanity, 20 examples, CPU, `--clap_stub`:
#     S1 20/20 = 1.0000 (stop at epoch 51/120) / S2 20/20 = 1.0000 (epoch 62/120)
#
# THE RUN ITSELF (executed on the cluster on 2026-08-13, results in the repo):
#   S1  244/536 = 0.4552 at the stopping epoch (119 of a 120 cap; early stopping
#       did NOT fire), measured null 0.5000, one-sided p 0.9829, per-recording mean
#       over 21 held-out recordings 0.4215, train loss 0.6962 -> 0.3160.
#       Gate 0.70 -> NOT PASSED.
#       provenance: docs/provenance/2026-08-13_exp19_S1_RESULT_summary.txt
#   S2  113/188 = 0.6011 at the stopping epoch (119 of 120), repaired null 0.5000,
#       one-sided p 0.003404, per-recording mean over 13 held-out recordings 0.5999,
#       train loss 0.6908 -> 0.1564. Gate 0.70 -> NOT PASSED.
#       provenance: docs/provenance/2026-08-13_exp19_S2_RESULT_summary.txt
#
#   WHAT THIS RESULT DOES, AND IT IS A MOTIVATION, NOT A VERDICT: it FALSIFIES the
#   explanation Exp. 18 suggested ("the train loss does not move: no learnable
#   signal"). At 120 epochs the train loss FALLS, from 0.6962 to 0.3160, and the
#   held-out accuracy stays FLAT. The head claim holds and is strengthened, but it
#   has to be motivated by the DISSOCIATION BETWEEN THE TWO CURVES, i.e. a DATA
#   limit. Do NOT write that training "stopped": early stopping never fired, and the
#   120-epoch cap was imposed by us.
#
#   AND S2 IS THE ONLY SIGNAL IN THE CHAPTER: 0.6011 against a repaired null of
#   0.5000. WHICH INSTRUMENT decodes a little; WHICH INSTANT does not. Exploratory
#   by construction, below the gate, 13 recordings out of 21. It opens nothing on
#   its own.
#
# `--clap_stub` replaces the FROZEN CLAP tower with a fixed random projection (the
#    1.74 GiB checkpoint is not on the analysis machine, and there is no torchaudio
#    there). Any number produced with that flag is a check on the WIRING, never a
#    result - and the sentence goes INSIDE the summary file.
#
# S2 IS EXPLORATORY BY CONSTRUCTION: the decision to run it was taken AFTER seeing
#    S1's number. It cannot support any confirmatory claim. The caveat is also
#    inside the run's summary.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 19: match-mismatch at convergence (S1) and the
#              instrument stage (S2, EXPLORATORY), pre-registered criterion
#              (2026-08-13)"
# PROVENANCE : docs/provenance/2026-08-13_exp19_S1.txt and _S2.txt (preparation),
#              docs/provenance/2026-08-13_exp19_S{1,2}_RESULT_* (the runs)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/anaconda3/bin/python}           # see the trap above: this one needs torch
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}

# The cluster commands - printed, not executed. Override these if the repo lives
# under a different name on the GPU node.
GPU_REPO=${GPU_REPO:-\$HOME/EEG-Attention-decoding-with-CLAP}
GPU_ENV=${GPU_ENV:-\$HOME/.conda/envs/eeg_attention/bin/python}
GPU_CKPT=${GPU_CKPT:-\$HOME/clap_ckpt/630k-audioset-best.pt}
GPU_MADEEG=${GPU_MADEEG:-\$HOME/madeeg}

print_run_command () {
  cat <<EOF

===============================================================================
THE TRAINING - runs on the GPU node (L40S). This script does not launch it.
    The node has no tmux: use nohup, so the terminal session can drop without
    taking the run with it.
===============================================================================
# (0) ONCE, before anything else:
REPO=$GPU_REPO
PY=$GPU_ENV
CKPT=$GPU_CKPT
MAD=$GPU_MADEEG

cd \$REPO && git pull && mkdir -p runs/logs

# (1) PRE-FLIGHT, ~3 min, MANDATORY: the same four gates run locally, but with the
#     REAL CLAP tower. If step 4 does not reach 20/20, the wiring is broken and
#     NOTHING is launched.
\$PY src/modules/clip_loss.py
\$PY src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir \$MAD \\
    --negative temporal_offset --check_sampling --check_null_by_split
\$PY src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir \$MAD \\
    --negative cross_instrument --check_sampling --check_null_by_split
\$PY src/madeeg_contrastive.py --madeeg_dir \$MAD --loss temporal_offset \\
    --clap_pretrained \$CKPT --overfit 20 --epochs 60 --batch_size 4

# =============================================================================
# COMMAND 1 - S1 at convergence.
# =============================================================================
cd \$REPO && nohup \$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss temporal_offset --clap_pretrained \$CKPT \\
    --epochs 120 --early_stop_patience 15 --early_stop_min_delta 0.002 \\
    --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    > runs/logs/exp19_S1_temporal_offset.log 2>&1 &

# =============================================================================
# COMMAND 2 - S2, the instrument stage, EXPLORATORY, with the repaired null.
#    It can start right after the first one or in parallel: independent runs.
# =============================================================================
cd \$REPO && nohup \$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss cross_instrument --clap_pretrained \$CKPT \\
    --balance_pairs \\
    --epochs 120 --early_stop_patience 15 --early_stop_min_delta 0.002 \\
    --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    > runs/logs/exp19_S2_cross_instrument.log 2>&1 &

# to follow them:  tail -f \$REPO/runs/logs/exp19_S1_temporal_offset.log
#                  tail -f \$REPO/runs/logs/exp19_S2_cross_instrument.log

# WHERE EVERYTHING LANDS (no invented path: it is resolve_log_dir()):
#   runs/results/madeeg_exp19_temporal_offset/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/results/madeeg_exp19_cross_instrument/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/logs/exp19_S{1,2}_*.log
#   exp19, NOT exp18: the new run is written BESIDE the old one, never over it.
#      No checkpoint is saved.
#
# HOW LONG IT TAKES - INFERRED, anchored to the MEASURED figure (40 s/epoch on 3346
#   pairs, L40S):
#   S1  3346 pairs/epoch -> 40 s  x 120 epochs = 80 min (CEILING: early stopping can
#       fire much earlier - in the actual run it did not)
#   S2  3028 pairs/epoch after balancing (2840 train + 188 held-out)
#       -> 40 x 3028/3346 = 36 s/epoch x 120 = 72 min (same ceiling)
#   plus ~15 s of dataset construction per run (measured locally).
#   If it is slower than expected the lever is free and deliberately NOT implemented:
#      CLAP is FROZEN and the distinct audio windows are ~400 in total, so its
#      embeddings could be computed ONCE instead of every epoch. It is code the
#      criterion does not ask for: add it only if the measured time justifies it.
===============================================================================
EOF
}

usage () {
  cat <<EOF
usage: bash scripts/replicate/exp19_matchmismatch.sh [--checks]

  (no flag)  cross the four CPU gates and PRINT the two GPU commands.
             Trains nothing: training is a GPU step.
EOF
}

cd "$REPO"
case "${1:---checks}" in
  --checks)
    echo "=== GATE 1 - Chapter 1 canary (synthetic, no data) ==="
    "$PY" src/modules/clip_loss.py

    echo
    echo "=== GATE 2 - S1 sampling UNCHANGED (the proof that the diff did not touch it) ==="
    "$PY" src/datasets/madeeg_solo_matchmismatch.py \
        --madeeg_dir "$MADEEG_DIR" --negative temporal_offset \
        --check_sampling --check_null_by_split

    echo
    echo "=== GATE 3 - S2 null RE-MEASURED after balancing, BEFORE any training ==="
    "$PY" src/datasets/madeeg_solo_matchmismatch.py \
        --madeeg_dir "$MADEEG_DIR" --negative cross_instrument \
        --check_sampling --check_null_by_split

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== GATE 4 - optimisation sanity, $NEG ==="
      echo "    20 examples, CPU, CLAP tower REPLACED: a control, never a result."
      BAL=""; [ "$NEG" = "cross_instrument" ] && BAL="--balance_pairs"
      "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
          --overfit 20 --clap_stub --epochs 120 --batch_size 4 \
          --early_stop_patience 15 --early_stop_min_delta 0.002 $BAL \
          --training_date "${TAG}exp19_overfit_${NEG}" \
        | grep -E "^\[balance|^\[overfit|^\[null|^\[$NEG|^\[early|reported at|one-sided|per held|written to"
    done

    print_run_command
    ;;
  *)
    usage; exit 2 ;;
esac
