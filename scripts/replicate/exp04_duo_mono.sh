#!/usr/bin/env bash
# =============================================================================
# Exp. 4 - the MONO half of the duos, never looked at before. The CONFIRMATORY run.
#
# WHAT IT MEASURES: whether the maximum over the 16 axis configurations (86/150 =
#   0.5733, read on 2026-07-29 outside the pre-registration) replicates on unseen
#   material - the 150 MONO duos, which exist only in the raw release and whose
#   stems have to be BORROWED from the stereo twin of the same mixture.
# DATE       : 2026-07-30. Criterion written on the evening of 2026-07-29, BEFORE the run.
# QUESTION   : "does the maximum over the 16 configurations replicate on the mono half?"
# NULL       : 0.500 - by construction on the duo (every mixture appears with both
#              its instruments as target).
# PRE-REGISTERED THRESHOLD: 86/150 = 0.5733, RECOMPUTED before the run because the
#              two pre-registered ones (89/155 and 87/151) did not apply: the
#              decidable trials are 150, not 155. Smallest integer with one-sided
#              exact binomial p < 0.05 (86 -> p = 0.0430).
# VERDICT    : DOES NOT REPLICATE - 79/150 = 0.5267, p = 0.2839, 95% CI [0.444, 0.609].
# COST       : ~5-10 min CPU (estimate: one 64 Hz run with ICA over 8 subjects plus
#              the alignment gate). TO BE CONFIRMED on first execution.
# BUDGET     : YES, and it is THE ONLY CONFIRMATORY LOOK the project spent on the
#              duos. With this run all 309 duos became SPENT (2026-07-30): every duo
#              number AFTER that date is exploratory by construction.
#              The TRIOS (92 stereo + 93 mono) are NOT touched: they are the only
#              remaining holdout, one shot, stereo threshold already written 38/90.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY    79/150 = 0.5267   p = 0.2839   95% CI [0.444, 0.609]
#              F1 micro/macro/weighted 0.5267/0.5225/0.5160
#              per subject 0001=0.60 0002=0.70 0003=0.60 0004=0.55 0005=0.70
#                          0007=0.30 0008=0.45 0009=0.20   (range 0.20-0.70)
#              r(attended) 0.0348 vs r(best unattended) 0.0255, inner_val_r 0.1162
#              n = 150 and not 155: the recovery gate declares 155 trials built,
#              150 decidable, and in the records `fold` = -1 on all 150.
#              TO BE CONFIRMED: the exact cause of the 5 missing ones. On the STEREO
#              arm the 4 missing (154 -> 150) are documented as an effect of
#              `--filters per_instrument` (subject 0007 has no Bo/Fh solos); for
#              mono the mechanism is plausibly the same but is written nowhere.
#              The 155/150 count itself IS written down.
#   GATE       stem borrowing r = 1.0000 over 272 comparisons, wav<->mixture 155/155,
#              span within 4 ms, 0 mono onsets coinciding with the stereo ones
#
# THREE THINGS TO WRITE EVERY TIME 79/150 IS QUOTED:
#   (a) power 0.534 against the motivating effect. 79/150 does NOT mean "the effect
#       is not there": it means "at this n it cannot be seen". Recomputed powers:
#       0.534, 0.774 (against 0.60), 0.312 (against 0.55).
#   (b) the threshold 86/150 is THE SAME INTEGER as the exploratory result that
#       motivated it (86/150 = 0.5733). The criterion asked to replicate a quantity
#       biased upward by construction (maximum over 16 comparisons; Bonferroni at 16
#       would want 93/150 = 0.62). Such a criterion is severe for the wrong reason,
#       and that has to be said.
#   (c) the confound did not fire: mono is WORSE. If the decision were spatial,
#       removing the separation should lower the number - and indeed it does. But it
#       drops in a regime where there is no signal anywhere, so it is evidence of
#       nothing.
#
# SPENT CONTRAST, on record: in the same session of 2026-07-30 the BY-GENRE split
#    was also read (classical 47/90 = 0.522, pop 32/60 = 0.533). It is not a result:
#    it is spent material. The by-genre contrast on the mono half is BURNED. This
#    script does not reprint it.
#
# CANARY     : the alignment gate `--check_alignment --spatial mono` is the PREMISE
#              of the whole experiment - if the borrowed stems are not the right
#              ones, the rest means nothing. Crossed FIRST here, and REDIRECTED TO A
#              FILE: on 2026-07-30 its output was saved nowhere, so what had been
#              verified was the gate's CODE, not its numbers. Rerunning it costs one
#              command and spends no held-out looks.
#              Second canary: `--check_alignment --spatial stereo` FAILS, and that is
#              correct - 8 trials of `pop_mixtape_BsDr_theme2` are truncated in the
#              release. That is a DATASET defect, not a code defect.
# PRE-REGISTRATION: "Chapter 2 - Exp. 4: pre-registered criterion on the MONO duos
#              (2026-07-29, evening)"
# PROVENANCE : RESULTS.md madeeg_duo section <- results_manifest.tsv, pin
#              `madeeg_exp4_mono_confirm`
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # the pinned record is never overwritten
RES="$REPO/runs/results"

mkdir -p "$OUT_DIR"
cd "$REPO"

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

# `--spatial mono` ASSERTS `--train_on raw_solos --test_eeg raw`: the mono duos have
#    no EEG in the preprocessed release, they do not exist there at all. The assert is
#    there because a run that LOOKS like the mono arm and is in fact the stereo one
#    would be the worst possible failure here: the two arms differ by ONE trial.
RAW=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --ensemble duo)

# ---- CANARY 1: the alignment gate, before anything else ---------------------
echo "=== CANARY 1 - alignment gate on the MONO arm ==="
echo "    (EEG/audio durations, declared onsets, and the stem BORROWING validated"
echo "     against the stereo half, where the \`soli\` field really exists)"
GATE="$OUT_DIR/exp04_check_alignment_mono.txt"
if "$PY" src/madeeg_reconstruction.py "${RAW[@]}" --check_alignment --spatial mono \
     > "$GATE" 2>&1; then
  grep -iE "PASS|FAIL|1\.0000|155|ms|borrow|onset" "$GATE" | head -12
else
  tail -15 "$GATE"
  echo "  -> [FAILED] the gate does not pass: the rest of the experiment means nothing."
  exit 1
fi
if ! grep -q "PASS" "$GATE"; then
  echo "  -> [FAILED] no PASS in the gate. STOP and report."
  exit 1
fi
echo "  -> [PASSED] full output in $GATE"
echo "     (on 2026-07-30 this file did not exist: that is the only difference)"

# ---- CANARY 2: the SAME gate on the stereo arm MUST fail --------------------
echo
echo "=== CANARY 2 - the same gate on STEREO must FAIL (8 truncated trials) ==="
echo "    Not a code bug: a defect of the release on pop_mixtape_BsDr_theme2."
echo "    If it passed here, the gate would not be checking anything."
GATE_S="$OUT_DIR/exp04_check_alignment_stereo.txt"
if "$PY" src/madeeg_reconstruction.py "${RAW[@]}" --check_alignment --spatial stereo \
     > "$GATE_S" 2>&1; then
  if grep -q "FAIL" "$GATE_S"; then
    grep -iE "FAIL|BsDr|truncat" "$GATE_S" | head -6
    echo "  -> [PASSED] the gate discriminates: it fails where it must."
  else
    echo "  -> WARNING: the stereo gate does NOT report FAIL. Check $GATE_S by hand"
    echo "     before believing the mono arm."
  fi
else
  grep -iE "FAIL|BsDr|truncat" "$GATE_S" | head -6
  echo "  -> [PASSED] exit != 0 on the stereo arm: that is the expected outcome."
fi

# ---- THE CONFIRMATORY RUN ---------------------------------------------------
# Every flag that changes the number, explicit. This is the WINNING configuration
# out of the 16 (ICA + per-instrument), applied to fresh material: that is the point
# of the replication, and it is not renegotiable after the fact.
echo
echo "=== THE CONFIRMATORY RUN - 150 MONO duos, never looked at before 2026-07-30 ==="
"$PY" src/madeeg_reconstruction.py "${RAW[@]}" --spatial mono \
      --eeg_clean notch_ica --filters per_instrument \
      --target mel --n_mels 8 --target_fs 64 \
      --band_low 1 --band_high 8 --lags_ms 250 \
      --estimator ridge --seed 42 --cv_folds 5 \
      --training_date "${TAG}exp4_mono_confirm"

# ---- CANARY 3: the pinned record must reproduce byte for byte ---------------
echo
echo "=== CANARY 3 - comparison with the pinned record madeeg_exp4_mono_confirm ==="
NEW="$RES/${TAG}exp4_mono_confirm/madeeg_records.csv"
OLD="$RES/madeeg_exp4_mono_confirm/madeeg_records.csv"
if [ -f "$OLD" ]; then
  A=$(_md5 "$OLD"); B=$(_md5 "$NEW")
  echo "  pinned     $A"
  echo "  replicated $B"
  if [ "$A" = "$B" ]; then
    echo "  -> [PASSED]"
  else
    echo "  -> [FAILED] the records differ. Check the INTERPRETER FIRST (trap #1),"
    echo "     then STOP and report."
    exit 1
  fi
else
  echo "  SKIPPED: the pinned record is not on this machine (runs/ is gitignored)."
fi

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
sed -n '1,10p' "$RES/${TAG}exp4_mono_confirm/madeeg_summary.txt"
echo
echo "pre-registered threshold: 86/150 = 0.5733 (p = 0.0430)  ->  79/150 does not cross it"
echo "pinned record (2026-07-30): runs/results/madeeg_exp4_mono_confirm/"
echo "power 0.534: this negative is NOT informative against small effects."
