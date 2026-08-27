#!/usr/bin/env bash
# =============================================================================
# Exp. 18 - match-mismatch on the MAD-EEG SOLOS: preparation and gates.
#
# WHAT IT MEASURES: nothing by itself. This script crosses the criterion's CPU
#   GATES (Chapter 1 canary, sampling test, optimisation sanity) and then PRINTS
#   the training command. Training does NOT start without an explicit flag: it is
#   a GPU step.
# COST       : ~3 min CPU measured (2 x check_sampling ~16 s + 2 x overfit ~60 s).
# BUDGET     : NO. Only the `solo` trials are opened (105 recordings, 420
#              repetitions), which the ledger of looks declares free training
#              material. No duo decided, no trio, no split by
#              genre/melody/instrument/subject. The TRIOS are never touched.
# GPU        : yes, for training only - the command is printed, not launched.
#              The gates here run on CPU.
#
# ENVIRONMENT TRAP, DIFFERENT FROM EVERY OTHER SCRIPT IN THIS DIRECTORY:
#    this one needs `torch`, which is NOT in `/opt/miniconda3`. It uses
#    `/opt/anaconda3` (torch 2.11.0 + h5py + soundfile + sklearn). This is not
#    environment trap #1: that one concerns the md5 gates of
#    `madeeg_reconstruction.py`, and there are none here - no number produced by
#    this script is compared against a pinned md5.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   CHAPTER 1 CANARY (`src/modules/clip_loss.py`, synthetic, torch.manual_seed(0)):
#     0.628491 / 1.2994 / 4.8198   and step D  0.951610  vs  2.160834
#   STEP C REGRESSION GATE (no torch, reads the archived records):
#     58/154 = 0.3766, and all 154 decisions memorised
#   SOLO CENSUS: 105 recordings / 420 repetitions / 39.4 min of EEG /
#     96 distinct wav files / 7 song+theme groups
#   S1 SAMPLING `temporal_offset`: 3346 pairs / negative from the SAME wav
#     3346/3346 / minimum offset 2.000 s (declared threshold 2.0 s) / same instant
#     0/3346 / NULL 0.5000 (best EEG-free rule) - exact by construction
#   S2 SAMPLING `cross_instrument`: 4414 pairs / same song+theme and DIFFERENT
#     instrument 4414/4414 / same instant on both sides 4414/4414 / NULL 0.5063
#   OPTIMISATION SANITY, 20 examples, CPU, `--clap_stub`:
#     S1 20/20 = 1.0000 / S2 20/20 = 1.0000 (both within ~10 epochs)
#
# THE RUN ITSELF (executed on the cluster on 2026-08-13, results in the repo):
#   S1  278/536 = 0.5187 against a measured null of 0.5000, one-sided p 0.2059,
#       per-recording mean over 21 held-out recordings 0.4938
#       pre-registered gate 0.70 -> NOT PASSED, the criterion stops there
#       provenance: docs/provenance/2026-08-13_exp18_S1_RESULT_summary.txt
#   S2  not run: the criterion makes it conditional on S1 crossing the gate.
#   The follow-up at a larger epoch budget is Exp. 19, and it falsified the
#   explanation this run's flat train loss suggested. See exp19_matchmismatch.sh.
#
# `--clap_stub` replaces the FROZEN CLAP tower with a fixed random projection: the
#    1.74 GiB checkpoint is not on the analysis machine and that machine has no
#    torchaudio either. Any number produced with that flag is a check on the
#    WIRING, never a result - and the sentence goes INSIDE the summary file, not
#    only on the terminal.
#
# AND THE NUMBER TO LOOK AT BEFORE LAUNCHING S2: the criterion declares "chance
#    0.500". For S1 that is true and measured (0.5000, exact by construction: every
#    pair of instants is in the index in BOTH directions). For S2 it is NOT:
#    subjects did not all hear the same solos, so the best EEG-free rule scores
#    0.5063 over the whole set and 0.6061 over the 21 recordings held out with
#    seed 42. Against 0.6061 the 0.70 gate is worth +0.094, not +0.20. It is a
#    metadata fact (zero held-out looks spent) and it has to be settled BEFORE the
#    run. Exp. 19 repairs it with --balance_pairs.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 18: match-mismatch on the MAD-EEG solos,
#              pre-registered criterion (2026-08-12)"
# PROVENANCE : docs/provenance/2026-08-12_exp18_matchmismatch.txt (pinned) and
#              docs/provenance/2026-08-13_exp18_S1_RESULT_{summary.txt,records.csv}
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/anaconda3/bin/python}           # see the trap above: this one needs torch
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}
STEP_C_RECORDS=${STEP_C_RECORDS:-$REPO/../_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv}

# The cluster command - printed, not executed.
# Every hyperparameter is the one from steps C/D (epochs 10, batch 8, lr 0.003,
# T 0.5, seed 42): NOTHING was tuned for this experiment, so the only factor that
# changes between step D and Exp. 18 is where the negative comes from.
GPU_REPO=${GPU_REPO:-\$HOME/EEG-Attention-decoding-with-CLAP}
GPU_ENV=${GPU_ENV:-\$HOME/.conda/envs/eeg_attention/bin/python}
GPU_CKPT=${GPU_CKPT:-\$HOME/clap_ckpt/630k-audioset-best.pt}
GPU_MADEEG=${GPU_MADEEG:-\$HOME/madeeg}

print_run_command () {
  cat <<EOF

===============================================================================
THE TRAINING - runs on the GPU node. This script does not launch it.
===============================================================================
REPO=$GPU_REPO
PY=$GPU_ENV
CKPT=$GPU_CKPT
MAD=$GPU_MADEEG

# (0) prerequisites on the node, to be verified ONCE before launching:
#     - the RAW release: madeeg_raw.hdf5, madeeg_raw.yaml, madeeg_sequences_raw.yaml
#       and the (unpacked) stimuli/ directory inside \$MAD.
#       Steps C/D used only madeeg_preprocessed.*: the preprocessed release contains
#       NO solos at all, so these four files are genuinely required.
#     - the CLAP checkpoint already on disk (Exp. 17):
#       \$CKPT
#       sha256 8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037
#     - torchaudio in the env (CLAPEncoder resamples 44.1 -> 48 kHz internally).

cd \$REPO && git pull && mkdir -p runs/logs

# (1) PRE-FLIGHT, 2 minutes, mandatory: the same gates run above, but with the REAL
#     CLAP tower. If this does not reach 20/20, the wiring on the node is broken and
#     the real training must not be launched.
\$PY src/datasets/madeeg_solo_matchmismatch.py \\
    --madeeg_dir \$MAD --check_sampling
\$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss temporal_offset \\
    --clap_pretrained \$CKPT \\
    --overfit 20 --epochs 60 --batch_size 4

# (2) S1 - the negative is the SAME song at a declared temporal offset.
\$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss temporal_offset \\
    --clap_pretrained \$CKPT \\
    --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    2>&1 | tee \$REPO/runs/logs/exp18_S1_temporal_offset.log

# (3) S2 - ONLY IF S1 crosses the gate (>= 0.70). Otherwise the criterion stops:
#     no further measurement, no claim. It did not: 278/536 = 0.5187.
\$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss cross_instrument \\
    --clap_pretrained \$CKPT \\
    --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    2>&1 | tee \$REPO/runs/logs/exp18_S2_cross_instrument.log

# WHERE EVERYTHING LANDS (no path to invent: it is resolve_log_dir()):
#   runs/results/madeeg_exp18_temporal_offset/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/results/madeeg_exp18_cross_instrument/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/logs/exp18_S{1,2}_*.log
#   No checkpoint is saved: the criterion asks for a held-out accuracy, not a model
#   to reuse. If the weights are wanted, that has to be asked for and added.
#
# HOW LONG IT TAKES - estimate, with its MEASURED anchor and its "to be confirmed":
#   measured anchor: the Exp. 17 extraction on the cluster did 8876 CLAP forwards in
#   496.6 s = 17.9 clips/s ON CPU (HTSAT-tiny, 10 s input, batch 32).
#   This run asks for 2 candidates per item:
#     S1  (2810 train + 536 held-out) x 2 x 10 epochs = 66,920 CLAP forwards
#     S2  (3561 train + 853 held-out) x 2 x 10 epochs = 88,280 CLAP forwards
#   -> ON CPU: S1 ~62 min, S2 ~82 min (arithmetic on the measured anchor).
#   -> ON GPU: TO BE CONFIRMED ON THE FIRST RUN. No CLAP-on-GPU measurement exists in
#      this project. At 100-300 clips/s that is 4-11 min (S1) and 5-15 min (S2);
#      dataset construction (reading the raw + band-pass + 96 wav) costs ~15 s
#      measured locally and has to be added once per run.
#   IF THE RUN IS SLOWER THAN EXPECTED, the obvious lever is free and deliberately
#      not implemented: CLAP is FROZEN and the distinct audio windows are ~400 in
#      total, so its embeddings could be computed ONCE instead of every epoch. It was
#      not done because it is extra code the criterion does not ask for; add it only
#      if the measured time justifies it.
===============================================================================
EOF
}

usage () {
  cat <<EOF
usage: bash scripts/replicate/exp18_matchmismatch.sh [--checks] [--train-s1] [--train-s2]

  (no flag)   cross the CPU gates and PRINT the GPU command. Trains nothing.
  --checks    the same, explicitly.
  --train-s1  run S1 locally. GPU STEP: normally run on the cluster node.
  --train-s2  run S2 locally. Same, and ONLY if S1 crossed the gate.
EOF
}

mkdir -p "$OUT_DIR"
cd "$REPO"
MODE=${1:---checks}

case "$MODE" in
  --checks)
    echo "=== GATE 1 - Chapter 1 canary (synthetic, no data) ==="
    "$PY" src/modules/clip_loss.py

    echo
    echo "=== GATE 2 - step C regression on the archived records ==="
    if [ -f "$STEP_C_RECORDS" ]; then
      /opt/miniconda3/bin/python src/madeeg_diagnose.py --check_rule "$STEP_C_RECORDS"
    else
      echo "  SKIPPED: step C records not found at $STEP_C_RECORDS"
    fi

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== GATE 3 - sampling test, $NEG ==="
      "$PY" src/datasets/madeeg_solo_matchmismatch.py \
          --madeeg_dir "$MADEEG_DIR" --negative "$NEG" --check_sampling
    done

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== GATE 4 - optimisation sanity, $NEG ==="
      echo "    20 examples, CPU, CLAP tower REPLACED: a control, never a result."
      "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
          --overfit 20 --clap_stub --epochs 60 --batch_size 4 \
          --training_date "${TAG}exp18_overfit_${NEG}" \
        | grep -E "^\[solo|^\[overfit|^\[Exp|one-sided|per held|pre-registered|written to"
    done

    print_run_command
    ;;
  --train-s1|--train-s2)
    NEG=temporal_offset; [ "$MODE" = "--train-s2" ] && NEG=cross_instrument
    echo "Running the TRAINING locally ($NEG). This is a GPU step; it only runs here"
    echo "    if CLAP and torchaudio are available in $PY."
    "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
        --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \
        --training_date "${TAG}exp18_${NEG}"
    ;;
  *)
    usage; exit 2 ;;
esac
