#!/usr/bin/env bash
# =============================================================================
# THE LINEAR ANCHOR (ridge) - the three variants in RESULTS.md, madeeg_duo section.
#
# WHAT IT MEASURES: whether the safe linear family (stimulus reconstruction +
#   argmax of the correlation across the present sources) decides WHICH of a duo's
#   two instruments was attended. Three variants, each differing in ONE thing:
#   how it is trained (k-fold on the duos vs the paper protocol's solos) and which
#   representation is reconstructed (log-mel vs Hilbert envelope).
# DATE       : archive 2026-05/07; the two pinned runs were rerun 2026-07-30/08
#              with the full header; `raw_solos` 2026-07-03.
# QUESTION   : "does the linear family decide attention on the duo?"
# NULL       : 0.500 - on a duo chance is 1/n_present, and every mixture appears
#              with BOTH its instruments as target (18/18), so the stimulus prior
#              is not legitimate information.
# PRE-REGISTERED THRESHOLD: 88/154 = 0.5714 (smallest integer with one-sided exact
#              binomial p < 0.05 at n=154). It is the threshold of the 2026-07-26
#              pre-registration.
# VERDICT    : none of the three crosses it.
#              A1 duos_kfold mel  86/154 = 0.5584  p = 0.0853   <- the project's
#                 HIGHEST attention number ever, and it is NOT significant.
#              A2 duos_kfold env  78/154 = 0.5065
#              A3 raw_solos  mel  74/154 = 0.4805  <- below chance
# COST       : ~5-10 min CPU for all three (estimate; at 64 Hz these are light runs).
#              TO BE CONFIRMED on first execution: no measured timing on record.
# BUDGET     : YES - these decide on the DUOS. All 309 duos have been spent since
#              2026-07-30: rerunning them opens no new material, but every duo
#              number after that date is EXPLORATORY BY CONSTRUCTION. The TRIOS are
#              never touched. A1 and A3 are FIRST LOOKS PRE-RULE: they were looked
#              at before the project's statistical convention existed, and cannot
#              be relabelled confirmatory after the fact.
# GPU        : no - madeeg_reconstruction.py is numpy/scipy/sklearn/h5py, no torch.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   A1  train_on=duos_kfold  target=mel n_bands=8   -> 0.5584, F1 micro/macro/weighted
#       0.5584/0.5448/0.5548, r(att) 0.0202 vs r(best unatt) 0.0118, inner_val_r 0.0245
#   A2  train_on=duos_kfold  target=envelope n_bands=1 -> 0.5065, r 0.0139/0.0133,
#       inner_val_r 0.0164
#   A3  train_on=raw_solos --test_eeg raw  target=mel  -> 0.4805, F1 0.4805/0.4849/0.4771,
#       r(att) 0.0162 vs r(best unatt) 0.0221, inner_val_r 0.0582
#
# THE FLAG THAT CHANGES THE NUMBER, AND THE TRAP THAT BIT:
#    A3 REQUIRES `--test_eeg raw`. The documented command WITHOUT that flag (i.e.
#    with the preprocessed release at test time) gives 0.4870, not 0.4805 - which
#    is also the accuracy of AXIS 0. Two numbers one digit apart for an unwritten
#    flag: that is why it is explicit here.
#
# A2 (envelope) IS NOT PINNED in results_manifest.tsv, and the manifest states the
#    reason: the only archived run (`madeeg_ridge_duo_env`) has a summary that
#    PREDATES the `--estimator` and `--spatial` flags, so the pin would not be
#    VERIFIABLE against it. 0.5065 lives in the run archive, outside RESULTS.md.
#    Rerunning it with today's header makes it pinnable: a 5-minute job nobody has
#    done yet.
#
# CANARY     : the three runs ARE each other's canaries over time - the two pinned
#              records (`madeeg_ridge_{duo,solos}_repro2026-08`) exist precisely to
#              be reproduced byte for byte. This script compares the md5 of
#              `madeeg_records.csv` against the pinned records and stops if it
#              differs. On top of that the ridge positive control (`--self_test`,
#              threshold 0.90 DECLARED IN THE CODE) runs FIRST.
# PRE-REGISTRATION: "Chapter 2 - pre-registered criterion (2026-07-26)"
#              (threshold 88/154)
# PROVENANCE : RESULTS.md madeeg_duo section <- results_manifest.tsv, pins
#              `madeeg_ridge_duo_repro2026-08` and `madeeg_ridge_solos_repro2026-08`;
#              A2 in the run archive, results/madeeg_ridge_duo_env/
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

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
acc()  { grep -h "OVERALL AAD accuracy" "$RES/$1/madeeg_summary.txt"; }
rec()  { echo "$RES/$1/madeeg_records.csv"; }

# The invariants of the three variants: identical everywhere, ONLY what the variant
# declares to change changes. The seed must be the same or the folds change.
BASE=(--madeeg_dir "$MADEEG_DIR" --ensemble duo --estimator ridge --filters pooled
      --eeg_clean none --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250
      --seed 42 --cv_folds 5 --spatial stereo)

# ---- CANARY 1: the ridge positive control, BEFORE the real numbers ----------
# SYNTHETIC EEG (mixture of the expected source at the model's lags + noise, snr 4.0),
# then the REAL ridge and the REAL decision. Threshold 0.90 declared in the code.
echo "--- CANARY 1: ridge positive control (--self_test, synthetic EEG) ---"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
      --training_date "${TAG}anchors_ctrl"
S="$RES/${TAG}anchors_ctrl_selftest/madeeg_selftest_summary.txt"
grep -h "AAD accuracy" "$S"
if ! grep -q "\[PASS\]" "$S"; then
  echo "  -> [FAILED] positive control not crossed: the real numbers are NOT looked at."
  exit 1
fi
echo "  -> [PASSED] this licenses the wiring and nothing else."

# ---- THE THREE VARIANTS ------------------------------------------------------
echo; echo "--- A1 : duos_kfold, target mel (the historical maximum, 0.5584) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on duos_kfold \
      --target mel --n_mels 8 --training_date "${TAG}ridge_A1_duo_mel"

echo; echo "--- A2 : duos_kfold, target envelope (NOT pinned, see the header) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on duos_kfold \
      --target envelope --training_date "${TAG}ridge_A2_duo_env"
#   ^ --n_mels is NOT passed: `envelope` is single-band by construction.

echo; echo "--- A3 : raw_solos, paper protocol. --test_eeg raw IS MANDATORY ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on raw_solos --test_eeg raw \
      --target mel --n_mels 8 --training_date "${TAG}ridge_A3_solos_mel"

# ---- CANARY 2: the two pinned records must reproduce byte for byte ----------
echo
echo "=== CANARY 2 - the pinned records, md5 by md5 ==="
FAILED=0
for P in "${TAG}ridge_A1_duo_mel:madeeg_ridge_duo_repro2026-08" \
         "${TAG}ridge_A3_solos_mel:madeeg_ridge_solos_repro2026-08"; do
  NEW=${P%%:*}; OLD=${P##*:}
  if [ -f "$(rec "$OLD")" ]; then
    A=$(_md5 "$(rec "$OLD")"); B=$(_md5 "$(rec "$NEW")")
    printf "  %-34s pinned    %s\n  %-34s replicated %s\n" "$OLD" "$A" "$NEW" "$B"
    if [ "$A" = "$B" ]; then
      echo "  -> [PASSED]"
    else
      echo "  -> [FAILED] the records differ. Check the INTERPRETER FIRST"
      echo "     (trap #1: /opt/anaconda3 moves the floats at 2e-5), then STOP and report."
      FAILED=1
    fi
  else
    echo "  SKIPPED: $OLD is not on this machine (runs/ is gitignored): nothing to compare against."
  fi
done

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in "${TAG}ridge_A1_duo_mel" "${TAG}ridge_A2_duo_env" "${TAG}ridge_A3_solos_mel"; do
  printf "%-30s " "$R"; acc "$R"
done
echo
echo "pre-registered threshold (2026-07-26): 88/154 = 0.5714. None of the three crosses it."
echo "pinned records : runs/results/madeeg_ridge_{duo,solos}_repro2026-08/"
echo "A2 (envelope)  : run archive, results/madeeg_ridge_duo_env/ - outside RESULTS.md"
[ "$FAILED" = 0 ] || exit 1
