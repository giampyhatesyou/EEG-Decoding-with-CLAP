#!/usr/bin/env bash
# =============================================================================
# Exp. 8 - the spectral-flux target on own-vs-other.
#
# WHAT IT MEASURES: whether changing the stimulus REPRESENTATION (log-mel ->
#   onset strength / per-band spectral flux) breaks the own-vs-other ceiling, with
#   paired McNemar against the mel-8 reference and alpha = 0.05/2 = 0.025.
# COST       : ~5 min CPU (estimate: 2 self-tests + 3 own-vs-other runs at 64 Hz,
#              ~1 min each from the Exp. 14 run timestamps of 2026-08-12).
# BUDGET     : NO - own-vs-other on held-out SOLO segments (structural assert
#              --train_on raw_solos). No duo, no trio.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   REF  (mel-8)      208/376 = 0.5532   md5 2eaa926244de340d31907c6deeb04b0c
#   F1   --target flux      243/376 = 0.6463   McNemar vs REF 84/133  p = 0.0015
#   F2   --target flux_mel  238/376 = 0.6330   McNemar vs REF 77/124  p = 0.0045
#   OUTCOME: BOTH pass Bonferroni (alpha 0.025) - the project's first
#   pre-registered positive. Null 0.500 EXACT by symmetry.
#   Positive controls (synthetic EEG, threshold 0.90 in the code): 1.0000 for both.
#
# CANARIES   : (1) the mel-8 reference is rerun FIRST and must give 208/376 with an
#              md5 identical to runs/results/ovo_ridge_ica/;
#              (2) the two --self_test positive controls are crossed BEFORE the real
#              numbers and the script stops if one does not print [PASS].
# PRE-REGISTRATION: "Chapter 2 - Exp. 8: the spectral-flux target on own-vs-other,
#              pre-registered criterion (2026-08-11)"
# PROVENANCE : docs/provenance/2026-08-11_exp8_flux_mcnemar.txt (pinned),
#              runs/results/{flux_ctrl,fluxmel_ctrl}_selftest, ovo_flux_{regcheck,F1,F2}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
REF_CSV=${REF_CSV:-$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv}
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

# The pre-registered reference: every other flag stays as here, ONLY --target changes.
OVO=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
     --eeg_clean notch_ica --estimator ridge --filters pooled
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42)

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
csv()  { echo "$REPO/runs/results/${TAG}$1/madeeg_ownvsother.csv"; }
acc()  { grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}$1/madeeg_ownvsother_summary.txt"; }

# ---- CANARY 1: the default path, before anything else ----------------------
echo "--- regression canary: the mel-8 reference ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}ovo_flux_regcheck"
GOT=$(_md5 "$(csv ovo_flux_regcheck)")
acc ovo_flux_regcheck
echo "  md5 obtained = $GOT"
echo "  md5 expected = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FAILED] the default changed: no candidate is run. STOP."
  exit 1
fi
echo "  -> [PASSED]"

# ---- CANARY 2: the positive controls, before the real numbers --------------
# Synthetic EEG built FROM the candidate representation; threshold 0.90 declared in
# the code. They write to <tag>_selftest, separate directories.
for T in flux flux_mel; do
  NAME=$([ "$T" = flux ] && echo flux_ctrl || echo fluxmel_ctrl)
  echo; echo "--- positive control --self_test --target $T ---"
  "$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
        --target "$T" --training_date "${TAG}${NAME}"
  S="$REPO/runs/results/${TAG}${NAME}_selftest/madeeg_selftest_summary.txt"
  grep -h "AAD accuracy" "$S"
  if ! grep -q "\[PASS\]" "$S"; then
    echo "  -> [FAILED] positive control not crossed: the real number is NOT looked at."
    exit 1
  fi
  echo "  -> [PASSED] this licenses the wiring and nothing else."
done

# ---- THE TWO CANDIDATES - K = 2, closed by the criterion -------------------
# F1: flux is single-band, --n_mels is not passed (the criterion says so explicitly).
echo; echo "--- F1 : --target flux ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target flux \
      --training_date "${TAG}ovo_flux_F1"
echo; echo "--- F2 : --target flux_mel (8 bands, capacity-matched to the reference) ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target flux_mel --n_mels 8 \
      --training_date "${TAG}ovo_flux_F2"

# ---- THE PAIRED TEST --------------------------------------------------------
MC="$OUT_DIR/exp08_flux_mcnemar.txt"
: > "$MC"
for F in F1 F2; do
  "$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_flux_regcheck)" "$(csv "ovo_flux_$F")" | tee -a "$MC"
done

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in ovo_flux_regcheck ovo_flux_F1 ovo_flux_F2; do printf "%-18s " "$R"; acc "$R"; done
echo
echo "replication McNemar : $MC"
echo "pinned provenance   : docs/provenance/2026-08-11_exp8_flux_mcnemar.txt"
echo "NOTE: alpha = 0.05/2 = 0.025. The threshold is the smallest integer with"
echo "      one-sided exact binomial p < alpha on the OBSERVED discordants."
