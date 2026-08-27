#!/usr/bin/env bash
# =============================================================================
# THE FOUR AXES of the published protocol (+ AXIS 0) - which axis moves the signal?
#
# WHAT IT MEASURES: the protocol of Cantisani et al. (WASPAA 2019) differs from
#   ours along five independent axes. This script moves them ONE AT A TIME,
#   starting from the `raw_solos` base, and judges them on `inner_val_r` - the
#   reconstruction quality on a split INTERNAL to the TRAINING material.
# DATE       : 2026-07-29.
# QUESTION   : "which axis of the published protocol moves the signal?"
# NULL       : none - `inner_val_r` is a RECONSTRUCTION metric, not a decision.
#              It has no chance level and is not tested against 0.5.
# PRE-REGISTERED THRESHOLD: none. This is a CONFIGURATION step, not a test: that
#              is why it is measured on `inner_val_r` and not on accuracy.
# VERDICT    : only one axis genuinely improves - ICA (AXIS 1b, +16%).
# COST       : ~30-60 min CPU. TO BE CONFIRMED: the --target_fs 256 runs (AXIS 0
#              and MAG@256) are ~6 GB of design matrix each. Needs 16 GB of RAM.
# BUDGET     : YES, AS WRITTEN HERE. `inner_val_r` is itself FREE (computed on a
#              split internal to the TRAINING material, touching no decision), but
#              the runs that produced it are FULL runs: they build the test trials
#              and also print duo accuracy. Reading that is a held-out look, and
#              all 309 duos are already spent.
#              For CONFIGURATION work there is `--inner_val_only`, which does not
#              build a single test trial - but ITS NUMBER IS NOT THE SAME: an
#              UNWEIGHTED mean over (subject x fold) instead of over TRIALS. On the
#              raw_solos ridge base: 0.0562 with `--inner_val_only` against 0.0582
#              weighted by trials, and the difference is declared inside the
#              script's own summary. Compare inner_val_only against inner_val_only,
#              NEVER against the pinned numbers below.
#              The TRIOS are untouched in both modes.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS - `inner_val_r`, base = 0.0582:
#   AXIS 1a  --eeg_clean notch                        0.0582   delta  0.0000  (exactly zero)
#   AXIS 1b  --eeg_clean notch_ica                    0.0673   delta +0.0091  (+16%)  <- THE ONLY ONE
#   AXIS 2   --filters per_instrument                 0.1100   ARTEFACT, see below
#   AXIS 3   --estimator shrinkage (lambda 0.1)       0.0349
#   AXIS 4   --target mag  (@64 Hz)                   0.0331   /  @256 Hz  0.0344
#   AXIS 0   --n_mels 24 --target_fs 256              0.0468   (the two halves, separately:
#            24 bands @64 -> 0.0528 / 8 bands @256 -> 0.0492: they worsen INDEPENDENTLY)
#   combinations with ICA: hparams 0.0485 / per_instrument 0.1162 / shrinkage 0.0471
#
#   AXIS 2 IS A METRIC ARTEFACT, NOT A DECODER GAIN.
#      With per-instrument decoders every source is reconstructed by the RIGHT
#      decoder, and `inner_val_r` measures exactly that. On the SAME 70 held-out
#      segments: pooled 0.0428 vs own 0.0442, and the fair test gives 36/70,
#      p = 0.29. It is the same undeclared choice that had produced the 49/70 =
#      0.70 of 2026-07-29 (explained and retracted on 2026-08-11): today
#      `--own_vs_other` FORBIDS `--filters per_instrument` with an assert, citing
#      this artefact.
#
#   AXIS 0 is the only ACCURACY number declared as a held-out look: 0.4870
#      (F1 micro/macro/weighted 0.4870/0.4899/0.4843), against the base 0.4805.
#      The PUBLISHED hyperparameters (24 mel @ 256 Hz) make it WORSE: they do not
#      recover the paper's 78-79%. Pinned run: `madeeg_repro_hparams`.
#
# THE ICA POSITIVE CONTROL, which is why AXIS 1b is believed:
#    frontal-EOG coupling in 1-8 Hz goes 0.552 -> 0.006, and the removed component
#    correlates r = 0.94 with the EOG. It is not "r went up": it is "the thing that
#    had to disappear disappeared".
#    TRAP PAID FOR: the channel *called* `ECG` is typed MISC; the real references
#    are AUX1 (kind 402) and AUX3 (kind 202 = EOG). Typing by NAME lets the ICA run,
#    declare success and remove nothing. The ICA removes components in only
#    5 subjects out of 8 (0004 and 0005 stay at 0).
#
# CANARY     : every axis DEFAULTS to the pre-existing behaviour and is ASSERTED
#              (a wrong combination breaks instead of silently falling back to the
#              old path while passing for the new one). The arm's three reference
#              numbers - 0.5584 / --self_test PASS / 0.4805 - must stay bit-identical
#              under the defaults AFTER any change: `ridge_anchors.sh` verifies that,
#              and this script points at it.
# PRE-REGISTRATION: none - it is a configuration step (hence `inner_val_r`).
#              Method note: "Chapter 2 - the 1-8 Hz band is ours: the candidate
#              right after testing (2026-07-29)".
# PROVENANCE : runs/results/ax_*/ and runs/results/madeeg_repro_hparams/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
RES="$REPO/runs/results"
INNER_ONLY=${INNER_ONLY:-0}                   # 0 = FULL runs: this is how the pinned
                                              #     numbers were produced, and the only way
                                              #     to reproduce them. They also print accuracy.
                                              # 1 = --inner_val_only: builds no test trial, so
                                              #     it CANNOT spend a held-out look - but it
                                              #     gives the UNWEIGHTED mean (0.0562 on the
                                              #     base, not 0.0582): NOT the pinned number.

cd "$REPO"

# The paper protocol's base: train on the raw solos, test on the duos reconstructed
# from the raw release. From here one axis moves at a time.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --spatial stereo
      --ensemble duo --target mel --n_mels 8 --target_fs 64 --band_low 1 --band_high 8
      --lags_ms 250 --estimator ridge --filters pooled --eeg_clean none
      --seed 42 --cv_folds 5)
EXTRA=()
[ "$INNER_ONLY" = 1 ] && EXTRA=(--inner_val_only)

run() {  # run <name> <axis flags...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "${EXTRA[@]}" "$@" \
        --training_date "${TAG}${name}"
}
# full runs -> madeeg_summary.txt ; --inner_val_only -> madeeg_innerval_summary.txt
r() { grep -h "inner_val_r" "$RES/${TAG}$1"/*summary*.txt 2>/dev/null; }

if [ "$INNER_ONLY" = 1 ]; then
  cat <<'WARN'
INNER_ONLY=1: no test trial is built, so no held-out look can be spent. IN EXCHANGE
   the number is not the pinned one: the mean is over (subject x fold) UNWEIGHTED,
   not over trials. raw_solos base: 0.0562, not 0.0582.
   Use it to CONFIGURE, and compare inner_val_only against inner_val_only.
WARN
else
  cat <<'WARN'
INNER_ONLY=0 (default): these are FULL runs - the only way to reproduce the pinned
   numbers. They build the test trials and also print duo ACCURACY. Reading it is a
   HELD-OUT LOOK, and all 309 duos are already spent: any number that comes out is
   EXPLORATORY BY CONSTRUCTION and must be labelled so in the section TITLE, not in
   a footnote. The TRIOS are untouched.
WARN
fi

# ---- BASE + the five axes, one at a time ------------------------------------
run ax_base                                                   # 0.0582
run ax_notch          --eeg_clean notch                       # AXIS 1a  0.0582
run ax_notch_ica      --eeg_clean notch_ica                   # AXIS 1b  0.0673  <- the only one
run ax_per_instrument --filters per_instrument                # AXIS 2   0.1100  artefact
run ax_shrinkage      --estimator shrinkage --shrinkage_lambda 0.1   # AXIS 3   0.0349
run ax_mag_fs64       --target mag                            # AXIS 4   0.0331
#   ^ --target mag ignores --n_mels: the bands are fixed by the STFT geometry.
run ax_mag_fs256      --target mag --target_fs 256            #          0.0344
run ax_mel24_fs64     --n_mels 24                             # AXIS 0, half 1  0.0528
run ax_mel8_fs256     --target_fs 256                         # AXIS 0, half 2  0.0492
run ax_hparams        --n_mels 24 --target_fs 256             # AXIS 0 complete 0.0468

# ---- the three combinations with ICA (the only axis that gained) -----------
run ax_ica_hparams    --eeg_clean notch_ica --n_mels 24 --target_fs 256   # 0.0485
run ax_ica_perinstr   --eeg_clean notch_ica --filters per_instrument      # 0.1162 artefact
run ax_ica_shrink     --eeg_clean notch_ica --estimator shrinkage --shrinkage_lambda 0.1  # 0.0471

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in ax_base ax_notch ax_notch_ica ax_per_instrument ax_shrinkage ax_mag_fs64 \
         ax_mag_fs256 ax_mel24_fs64 ax_mel8_fs256 ax_hparams \
         ax_ica_hparams ax_ica_perinstr ax_ica_shrink; do
  printf "%-20s " "$R"; r "$R" || echo "(no summary)"
done
echo
echo "pinned records (2026-07-29): runs/results/ax_*/ and runs/results/madeeg_repro_hparams/"
echo "AXIS 2 (0.1100 / 0.1162) is a METRIC artefact, not a decoder gain:"
echo "   on the same 70 held-out segments the fair test gives 36/70, p = 0.29."
echo
echo "The declared held-out look of AXIS 0 (accuracy 0.4870) lives in"
echo "runs/results/madeeg_repro_hparams/ - reproduced here by ${TAG}ax_hparams."
echo "To configure without building any test trial:"
echo "  INNER_ONLY=1 bash \$0     # <- UNWEIGHTED mean: 0.0562 on the base, not 0.0582"
