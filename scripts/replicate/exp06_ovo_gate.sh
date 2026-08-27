#!/usr/bin/env bash
# =============================================================================
# Exp. 6 [G] - the REPLACEMENT gate: own-vs-other, ridge against multi-view CCA.
#              All FOUR configurations, not just the reference.
#
# WHAT IT MEASURES: on held-out SOLO segments, does the estimator give the same EEG
#   a higher score against its OWN audio than against another solo's? This is
#   DISCRIMINATION - the quantity de Cheveigne's objection is about - on TRAINING
#   material: no duo is loaded and no attention decision is taken.
# DATE       : 2026-08-10. It replaces the step-1 gate, declared ill-posed in the
#              pre-registration itself (and the failed gate stays on record).
# QUESTION   : "does multi-view CCA beat ridge on a DISCRIMINATIVE metric?"
# NULL       : 0.500 EXACT BY SYMMETRY, not estimated: the pairs are mutual and
#              disjoint, with different instruments, and each is evaluated in BOTH
#              directions. An estimator with nothing but a fixed instrument
#              preference gets exactly one of the two. n = 376.
# PRE-REGISTERED THRESHOLD: paired McNemar, smallest integer with one-sided exact
#              binomial p < 0.05 on the OBSERVED discordants:
#                no cleaning   80/139     /   notch+ICA   76/131
# VERDICT    : NO-GO for CCA. With notch+ICA: 66/131 discordants favouring CCA
#              against 65 favouring ridge, p = 0.5000. It is not "CCA loses": it is
#              "CCA adds nothing".
# COST       : ~10-15 min CPU for the four runs (estimate: ~1 min each at 64 Hz for
#              ridge, CCA is slower). TO BE CONFIRMED on first execution.
# BUDGET     : NO - own-vs-other runs on held-out SOLO segments (the structural
#              assert forces `--train_on raw_solos`): no duo loaded, no attention
#              decision. The TRIOS are never touched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#
#   configuration        ridge              CCA                paired McNemar
#   base (no cleaning)   202/376 = 0.5372   207/376 = 0.5505   72/139  p 0.3673  threshold 80/139
#   notch + ICA          208/376 = 0.5532   209/376 = 0.5559   66/131  p 0.5000  threshold 76/131
#
#   one-sided binomial p: 0.0819 / 0.0281 / 0.0221 / 0.0172
#   EXACT paired p (permutation T = n1 + 2*Binom(n0+n2, 1/2), the right form because
#   the 376 decisions are 188 pairs, not 376 independent units):
#                        0.0851 / 0.0265 / 0.0235 / 0.0230
#   THE EXACT p IS THE ONE TO QUOTE. The dependence concern was well founded in
#      principle and negligible in fact: no verdict changes.
#   intersection of the comparisons between the two estimators: 376/376 (the
#   `--mcnemar` check), so the two are compared on the SAME comparisons.
#
# THE CONDITIONAL BRANCH FIRES WHERE IT MATTERS, AND THAT IS THE INFORMATIVE PART:
#    - without cleaning the ridge ITSELF is at chance (0.5372, p = 0.0819 > 0.05):
#      there the test HAS NO SENSITIVITY and says nothing about either;
#    - with notch+ICA the ridge is above chance (0.5532, p = 0.0221): there
#      sensitivity is DEMONSTRATED, and exactly there CCA does not beat it.
#    The NO-GO is therefore pronounced where it means something.
#    The preprocessing configuration was NOT pre-registered. Both were run and the
#    NO-GO was pronounced on the one WITH sensitivity - a choice UNFAVOURABLE to the
#    hypothesis being tested, but taken AFTERWARDS. State it that way, do not hide it.
#
# THE FACT ABOUT THE DATA, which is worth more than the verdict:
#    on training material, in-distribution, BOTH methods discriminate barely above
#    chance: ~0.55 against 0.500, with n = 376. It is not an estimator problem: it is
#    how little decodable tracking there is in these 20 channels. It is what makes
#    the rest of Chapter 2 readable.
#
# AN UNREPRODUCED PRECEDENT, and it has to be said: the 49/70 = 0.70 (p 6.0e-04) of
#    2026-07-29 ran with `--filters per_instrument` and a definition of *other* that
#    is NOT preserved in the repo. With per-instrument decoders the OWN audio is
#    reconstructed by the RIGHT decoder and the *other* audio by a WRONG one: an
#    undeclared methodological choice, the same one that produced the 0.1100 artefact
#    of AXIS 2. Today `--own_vs_other` FORBIDS `--filters per_instrument` with an
#    assert. Those 70 comparisons and these 376 are not the same number and do not
#    belong in the same table.
#
# CANARY     : the ridge + notch_ica configuration is the whole project's default
#              reference: 208/376 + md5 `2eaa926244de340d31907c6deeb04b0c`. It is
#              the base of Exp. 7, 8, 14, 16B and 18. It is rerun FIRST and the
#              script stops if the md5 disagrees. On top of that, `--mcnemar` checks
#              on its own that the two CSVs share 376/376 comparisons.
# PRE-REGISTRATION: "Chapter 2 - Exp. 6: own-vs-other gate and paired alpha,
#              pre-registered criterion (2026-08-10)"
# PROVENANCE : runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
RES="$REPO/runs/results"
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
csv()  { echo "$RES/${TAG}$1/madeeg_ownvsother.csv"; }
acc()  { grep -h "own-vs-other accuracy" "$RES/${TAG}$1/madeeg_ownvsother_summary.txt"; }

# The invariants: identical across all four. `--filters pooled` is IMPOSED by the
# --own_vs_other assert (see the 49/70 precedent above); the seed must be the same
# or the paired test would compare different comparisons.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
      --filters pooled --target mel --n_mels 8 --target_fs 64
      --band_low 1 --band_high 8 --lags_ms 250 --seed 42)

run() {  # run <name> <configuration flags...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "$@" --training_date "${TAG}${name}"
}

# ---- CANARY: the default reference, before anything else -------------------
run ovo_ridge_ica --estimator ridge --eeg_clean notch_ica
GOT=$(_md5 "$(csv ovo_ridge_ica)")
echo; echo "=== CANARY - ridge+ICA is the reference of the WHOLE project ==="
acc ovo_ridge_ica
echo "  md5 obtained = $GOT"
echo "  md5 expected = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FAILED] the default path has changed."
  echo "     Check the INTERPRETER FIRST: /opt/anaconda3 gives the same 376 decisions"
  echo "     with floats differing at 2e-5 and this md5 changes (it happened, Exp. 7)."
  echo "     Then STOP and report."
  exit 1
fi
echo "  -> [PASSED] this licenses the wiring and nothing else."

# ---- THE OTHER THREE CONFIGURATIONS -----------------------------------------
# Every CCA default reproduces EXACTLY the ridge data path (one EEG view = its own
# design matrix, one stimulus view = its own target), so `--estimator cca` with no
# other flag is the like-for-like comparison.
run ovo_cca_ica --estimator cca --eeg_clean notch_ica
run ovo_ridge   --estimator ridge --eeg_clean none
run ovo_cca     --estimator cca   --eeg_clean none

# ---- THE PAIRED TEST - McNemar, within each configuration -------------------
# Ridge is paired against CCA WITHIN the same preprocessing configuration. Pairing
# ridge-base against CCA-ICA would confound two axes in a single test.
MC="$OUT_DIR/exp06_ovo_gate_mcnemar.txt"
: > "$MC"
echo; echo "--- McNemar, notch+ICA configuration (the one with demonstrated sensitivity) ---"
"$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_ridge_ica)" "$(csv ovo_cca_ica)" | tee -a "$MC"
echo; echo "--- McNemar, base configuration (here the ridge itself is at chance) ---"
"$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_ridge)" "$(csv ovo_cca)" | tee -a "$MC"

# ---- THE EXACT PAIRED p VALUES - the ones to quote --------------------------
echo
echo "=== EXACT paired p (188 pairs, not 376 independent units) ==="
"$PY" docs/provenance/2026-08-11_ovo_sweep_paired_stats.py \
      "$(csv ovo_ridge_ica)" "$(csv ovo_cca_ica)" "$(csv ovo_ridge)" "$(csv ovo_cca)" \
  | tee "$OUT_DIR/exp06_ovo_gate_paired_stats.txt"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in ovo_ridge ovo_cca ovo_ridge_ica ovo_cca_ica; do printf "%-14s " "$R"; acc "$R"; done
echo
echo "replication McNemar    : $MC"
echo "replication exact p    : $OUT_DIR/exp06_ovo_gate_paired_stats.txt"
echo "pinned records (2026-08-10): runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/"
echo
echo "The verdict is 'CCA ADDS nothing', not 'CCA loses'."
echo "And the fact about the data is that BOTH sit at ~0.55 against a null of 0.500."
