#!/usr/bin/env bash
# =============================================================================
# Exp. 9 (EXPLORATORY) - the flux front end on the ATTENTION task.
#
# WHAT IT MEASURES: whether the flux gain on own-vs-other (Exp. 8, +35/376)
#   transfers to the attention decision on the duo, with paired McNemar against the
#   mel anchors on the same trials and alpha = 0.05/2 = 0.025.
# COST       : ~5 min CPU (estimate: 4 decision runs at 64 Hz; TO BE CONFIRMED -
#              the 2026-08-11 timestamps are all identical and give no duration).
# BUDGET     : YES - it decides on the DUOS. All 309 duos are ALREADY SPENT:
#              rerunning them opens no new material, but every duo number stays
#              EXPLORATORY BY CONSTRUCTION and must be labelled so IN THE TITLE.
#              The TRIOS are never touched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   mel anchor P1 (raw_solos) 74/154 = 0.4805   mel anchor P2 (k-fold) 86/154 = 0.5584
#   P1 flux  76/154 = 0.4935   McNemar vs mel 33/64  p = 0.4503  -> does not pass
#   P2 flux  77/154 = 0.5000   McNemar vs mel 26/61  p = 0.9000  -> does not pass
#   OUTCOME: the tracking -> attention transfer FAILS. Null 0.500 by construction
#   on the duo; absolute bar at alpha 0.025 for n=154 is 90/154.
#
# CANARY     : the two mel anchors are rerun FIRST and must reproduce the PER-TRIAL
#              DECISIONS of the pinned canaries (74/154 and 86/154 with ZERO
#              discordants, not just the same total). The script stops otherwise.
# PRE-REGISTRATION: "Chapter 2 - Exp. 9 (EXPLORATORY): the flux front end on the
#              attention task, pre-registered criterion (2026-08-11)"
# PROVENANCE : docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt (pinned),
#              runs/results/exp9_*/, anchors runs/results/canary_{rawsolos,duo}/
#
# The AUDIO-ONLY block of the same provenance file (stem correlations: mel
# 0.1772/0.1442, flux 0.2874/0.2417, flux more similar in 30/36 duos) is NOT
# reproduced from here: it is the Exp. 13 canary, which reprints it to 4 decimals.
#   -> bash scripts/replicate/exp13_stem_separability.sh
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
RES="$REPO/runs/results"

mkdir -p "$OUT_DIR"
cd "$REPO"

# The two Exp. 9 protocols. `eeg_clean none` everywhere, like the anchors.
# P1 = paper protocol (train on the solos, test on the raw EEG).
# P2 = k-fold protocol on the duos.
P1=(--train_on raw_solos --test_eeg raw --spatial stereo)
P2=(--train_on duos_kfold --spatial stereo)
DEC=(--madeeg_dir "$MADEEG_DIR" --estimator ridge --filters pooled --eeg_clean none
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 --cv_folds 5
     --ensemble duo)

rec() { echo "$RES/$1/madeeg_records.csv"; }
acc() { grep -h "OVERALL AAD accuracy" "$RES/$1/madeeg_summary.txt"; }

# ---- CANARY: the mel anchors, decision-identical, before anything else -----
echo "--- canary: mel anchor P1 (paper protocol) ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P1[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}exp9_gate_rawsolos_mel"
echo "--- canary: mel anchor P2 (duo k-fold) ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P2[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}exp9_gate_duo_mel"

GATE="$OUT_DIR/exp09_gate_anchors.txt"
: > "$GATE"
for P in "canary_rawsolos:${TAG}exp9_gate_rawsolos_mel:74" "canary_duo:${TAG}exp9_gate_duo_mel:86"; do
  OLD=${P%%:*}; REST=${P#*:}; NEW=${REST%%:*}; K=${REST##*:}
  OUT=$("$PY" src/madeeg_diagnose.py --mcnemar "$(rec "$OLD")" "$(rec "$NEW")")
  echo "$OUT" | tee -a "$GATE"
  # zero discordants: madeeg_diagnose.py prints "no discordant comparisons" and not
  # the discordants line (path verified in src/madeeg_diagnose.py:404).
  if ! echo "$OUT" | grep -q "no discordant comparisons"; then
    echo "  -> [FAILED] anchor $OLD is not decision-identical. STOP and report."
    exit 1
  fi
  if ! echo "$OUT" | grep -q "A correct $K/154"; then
    echo "  -> [FAILED] anchor $OLD does not give $K/154. STOP and report."
    exit 1
  fi
  echo "  -> [PASSED] $OLD: $K/154, zero discordant decisions."
done

# ---- THE TWO PRIMARIES - K = 2, closed by the criterion --------------------
echo; echo "--- P1 : flux, paper protocol, stereo ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P1[@]}" --target flux \
      --training_date "${TAG}exp9_P1_rawsolos_flux"
echo; echo "--- P2 : flux, duo k-fold, stereo ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P2[@]}" --target flux \
      --training_date "${TAG}exp9_P2_duo_flux"

# The criterion's secondaries S1/S2 (mel and flux k-fold on the MONO duos) are
# UNEXECUTABLE and are declared as such on record: --spatial mono asserts
# --train_on raw_solos --test_eeg raw, so a mono k-fold does not exist in the code.
# They are NOT substituted with another configuration.
#   "${DEC[@]}" "${P2[@]/--spatial stereo/--spatial mono}"   <-- breaks on the assert

# ---- THE PAIRED TEST --------------------------------------------------------
MC="$OUT_DIR/exp09_flux_attention_mcnemar.txt"
: > "$MC"
"$PY" src/madeeg_diagnose.py --mcnemar "$(rec canary_rawsolos)" "$(rec "${TAG}exp9_P1_rawsolos_flux")" | tee -a "$MC"
"$PY" src/madeeg_diagnose.py --mcnemar "$(rec canary_duo)"      "$(rec "${TAG}exp9_P2_duo_flux")"      | tee -a "$MC"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in "${TAG}exp9_gate_rawsolos_mel" "${TAG}exp9_gate_duo_mel" \
         "${TAG}exp9_P1_rawsolos_flux" "${TAG}exp9_P2_duo_flux"; do
  printf "%-34s " "$R"; acc "$R"
done
echo
echo "replication gate    : $GATE"
echo "replication McNemar : $MC"
echo "pinned provenance   : docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt"
