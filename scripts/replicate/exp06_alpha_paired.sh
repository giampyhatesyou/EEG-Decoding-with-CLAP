#!/usr/bin/env bash
# =============================================================================
# Exp. 6 [H] - PAIRED LATERALISED ALPHA. The real test of the spatial-attention
#              hypothesis, plus [H-prime], the mono control.
#
# WHAT IT MEASURES: whether alpha-power lateralisation predicts the SIDE of the
#   attended instrument. No decoder, no audio, no training: it is a SIGN TEST on
#   the alpha laterality index (F3/F4, C3/C4, P3/P4, O1/O2), evaluated on TWIN
#   PAIRS - same subject, same mixture, both targets - where every subject,
#   channel, impedance and session constant CANCELS ALGEBRAICALLY in the difference.
# DATE       : 2026-08-10.
# QUESTION   : "does alpha lateralisation predict the SIDE of the attended instrument?"
# NULL       : 0.500 EXACT BY EXCHANGEABILITY, not estimated: under H0 the two
#              twins are exchangeable (everything identical except which source is
#              attended), so the sign of the difference of their LI is equally
#              likely either way.
# PRE-REGISTERED THRESHOLD: 28/44 = 0.6364 on the primary (p = 0.0481; 27 would give
#              0.0871, so no freedom of choice), and 27/42 = 0.6429 on the mono
#              control (p = 0.0442; 26 would give 0.0821). Written 2026-08-09, i.e.
#              BEFORE any number existed.
# VERDICT    : BELOW THRESHOLD - primary 23/44 = 0.5227, exact p 0.4402.
#              Mono control 24/42 = 0.5714, p 0.2204: below threshold too, but
#              NUMERICALLY HIGHER THAN THE PRIMARY.
# COST       : ~10-20 min CPU for the six runs (2 real + 4 controls). TO BE CONFIRMED.
# BUDGET     : YES - it reads the duos. All already spent: EXPLORATORY BY
#              CONSTRUCTION, and the caveat is written INSIDE the run's summary.
#              The TRIOS are untouched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#
#   PRIMARY    stereo, twin pairs   23/44 = 0.5227  p 0.4402  threshold 28/44  below
#              per subject 0001=0.50 0002=1.00 0003=0.50 0004=0.25 0005=0.50
#                          0007=0.20 0008=0.60 0009=0.50
#   CONTROL    mono, twin pairs     24/42 = 0.5714  p 0.2204  threshold 27/42  below
#   POSITIVE CONTROLS (lateralised injection into the REAL EEG, side taken from the
#   TRUE panning, identical code; threshold 0.90 declared in the code before the
#   run; reference amplitude 1.0 declared in advance, 0.5 and 2.0 as dose-response
#   and NOT as criterion):
#              a = 0.5   43/44 = 0.9773
#              a = 1.0   44/44 = 1.0000  <- THE CRITERION
#              a = 2.0   44/44 = 1.0000
#              a = 1.0 in the run's EXACT configuration (notch+ICA)  44/44
#
#   n = 44 and not 47: the 3 pairs with a CENTRED target (panning 0.5,
#      `pop_mixtape_duo_GtVx_theme1_stereo_Vx`) are excluded A PRIORI by the
#      2026-08-09 criterion. Today's code excludes them, so a rerun reproduces the
#      numbers at n = 44 (`alpha_ctrl2_*`, `alpha_ctrl_ica_inject`). The directories
#      `alpha_ctrl_a*_inject` predate the exclusion and report 46/47, 47/47, 47/47
#      at n = 47. They do not conflict with the others: they are two VINTAGES, and
#      must be cited as such. If a rerun today gives 47 pairs, it is not the current
#      code.
#
# THE POWER, which is what makes the negative readable:
#    0.371 against an effect of 0.60. This negative is NOT informative against small
#    effects. Saying "alpha is irrelevant" would claim more than the data supports;
#    the defensible sentence is "at n = 44 it cannot be seen, with power 0.371".
#    The same power is elsewhere quoted "against 0.65": the value recomputed on
#    2026-08-12 in the threshold audit is 0.371 against 0.60.
#
# THE CONTROL IS HIGHER THAN THE PRIMARY, AND THE TWO PRE-REGISTRATIONS DISAGREE
#    ON WHAT THAT MEANS. This has to be resolved in writing, not chosen silently.
#    The two texts, verbatim (translated from the originals):
#      - Exp. 6 criterion, section 2.4: "Mono must stay AT CHANCE. If mono beats
#        stereo, the spatial reading is FALSIFIED even if stereo passes" -
#        condition: mono > stereo. SATISFIED (0.5714 > 0.5227).
#      - Exp. 5 criterion, section 4(a): "If mono CROSSES THE THRESHOLD as much as
#        or more than stereo, the spatial reading is FALSIFIED even if the primary
#        passes" - condition: mono >= threshold. NOT SATISFIED (24/42 < 27/42).
#    One report concludes that "the formal falsification clause does not fire - the
#    primary does not pass", while the evidence dossier records "pre-written
#    falsifier FIRED". STILL OPEN: it must be settled by stating which of the two
#    formulations governs. The sentence that holds in either branch:
#    "there is nothing in these numbers that looks like a spatial effect".
#
# AND MONO AND STEREO ARE NOT THE SAME SET OF PAIRS: stereo excludes the 3
#    centred-target pairs (n = 44), mono does not (n = 42, and by construction in
#    mono the panning is 0.5 everywhere). It is NOT a paired comparison and must not
#    be presented as one.
#
# THE WIDER POINT, which holds for all of Chapter 2: lateralised alpha is a TONIC
#    correlate, and the safe family decides by WITHIN-TRIAL CORRELATION - which
#    centres away any trial constant. The two forms do not meet without changing one
#    of them. It is also why the CCA `position` view is inert (|delta| < 1e-9,
#    asserted in the code).
#
# CANARY     : the injection positive control runs BEFORE the real number and in the
#              run's EXACT configuration (notch+ICA), not only in a convenient one.
#              Threshold 0.90 in the code. The controls write to separate directories
#              (`<tag>_inject`) with the caveat INSIDE the file, above the numbers.
# PRE-REGISTRATION: "Chapter 2 - Exp. 6: own-vs-other gate and paired alpha,
#              pre-registered criterion (2026-08-10)" (the thresholds come from the
#              Exp. 5 criterion of 2026-08-09)
# PROVENANCE : runs/results/alpha_real_{stereo,mono}/madeeg_alpha_summary.txt,
#              runs/results/alpha_ctrl2_a{0.5,1.0,2.0}_inject/,
#              runs/results/alpha_ctrl_ica_inject/
#              All under runs/, which is GITIGNORED: from a clean clone these
#              numbers do not exist until this script is rerun.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
RES="$REPO/runs/results"

cd "$REPO"

# `--alpha_li` ASSERTS `--train_on raw_solos`: it reads the continuous RAW record.
#    It trains nothing - that flag is only what loads the raw release.
#    And it ASSERTS `--spatial != both`: mono is the CONTROL arm and merging it with
#    the primary would destroy it.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw
      --alpha_li --ensemble duo --seed 42)

pair() { grep -hE "PAIRED ACCURACY|threshold|twin pairs" \
         "$RES/${TAG}$1/madeeg_alpha_summary.txt" 2>/dev/null; }

# ---- THE POSITIVE CONTROLS, BEFORE THE REAL NUMBER -------------------------
# Lateralised injection of known amplitude into the REAL EEG, side from the TRUE
# panning, SAME code. If the wiring cannot recover an effect that is there by
# construction, the real number means nothing.
echo "=== POSITIVE CONTROLS - lateralised injection (threshold 0.90 in the code) ==="
for A in 1.0 0.5 2.0; do
  echo; echo "--- injection a = $A (dose-response; 1.0 is THE criterion) ---"
  # tag `ctrl2` like the pinned directories of the CURRENT vintage (n = 44).
  # The `alpha_ctrl_a*_inject` ones are the n = 47 vintage, BEFORE the exclusion of
  # the 3 centred targets: they are neither overwritten nor compared.
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
        --eeg_clean none --alpha_inject "$A" \
        --training_date "${TAG}alpha_ctrl2_a${A}"
  pair "alpha_ctrl2_a${A}_inject"
done

echo; echo "--- injection a = 1.0 in the RUN'S EXACT CONFIGURATION (notch+ICA) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
      --eeg_clean notch_ica --alpha_inject 1.0 \
      --training_date "${TAG}alpha_ctrl_ica"
S="$RES/${TAG}alpha_ctrl_ica_inject/madeeg_alpha_summary.txt"
pair "alpha_ctrl_ica_inject"
if ! grep -q "44/44 = 1.0000" "$S"; then
  echo "  -> [FAILED] the positive control in the run's configuration does not give 44/44."
  echo "     The real number is NOT looked at. STOP and report."
  exit 1
fi
echo "  -> [PASSED] this licenses the wiring and nothing else. It says nothing about the real data."

# ---- THE PRIMARY AND ITS CONTROL -------------------------------------------
echo
echo "=== [H] PRIMARY - STEREO twin pairs (this IS a held-out look on the duos) ==="
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
      --eeg_clean notch_ica --training_date "${TAG}alpha_real_stereo"

echo
echo "=== [H-prime] CONTROL - MONO twin pairs (must stay AT CHANCE) ==="
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial mono \
      --eeg_clean notch_ica --training_date "${TAG}alpha_real_mono"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
echo "--- [H] PRIMARY stereo ---";      pair "alpha_real_stereo"
echo "--- [H-prime] CONTROL mono ---";  pair "alpha_real_mono"
echo
echo "pinned records (2026-08-10): runs/results/alpha_real_{stereo,mono}/ (gitignored)"
echo
echo "power 0.371 against an effect of 0.60: this negative is NOT informative"
echo "   against small effects. And the mono control is HIGHER than the primary:"
echo "   see in this script's header the TWO pre-registered formulations that give"
echo "   different answers on what that means. The entry is OPEN."
echo "The sign is NOT flipped: the physiological direction (CONTRALATERAL alpha"
echo "   desynchronisation) is fixed in the code and in the summary BEFORE the run,"
echo "   and was not touched afterwards."
