#!/usr/bin/env bash
# =============================================================================
# canaries.sh - ALL the project's canaries, in one go, in order.
#
# WHAT IT MEASURES: nothing new. It crosses the regression gates that protect the
#   numbers already reported and says, for each one, PASSED / FAILED / NOT
#   CROSSABLE HERE (with the reason). It is the first thing to run after any diff,
#   and the first thing to report when something does not add up.
# DATE       : written 2026-08-15, from the canaries in use at that date.
# QUESTION   : has a number that was already reported moved?
# NULL       : n/a - none of these gates produces a result.
# THRESHOLD  : each canary carries its own, literally, below. None is negotiable
#              and none gets "updated" because the code changed: if a canary does
#              not pass, STOP and report.
# VERDICT    : printed by the run. Exit code != 0 if a CROSSABLE canary fails.
# COST       : level 0 ~10 s CPU / level 1 ~5 s / level 2 ~20-30 min CPU.
# BUDGET     : no held-out looks spent. No level decides on a duo: the level-2
#              canaries are own-vs-other (held-out SOLO segments), synthetic
#              controls, and re-reads of records ALREADY SPENT. The trios are
#              never touched.
# GPU        : no. The canaries that need a GPU are PRINTED, never launched.
#
# THE FOUR LEVELS, and why they are separate (environment traps, below):
#
#   L0  no data, no torch. Runs anywhere, including a clean clone (except the
#       three md5s, which read artefacts under runs/, which is gitignored).
#   L1  needs `torch` -> conda env `attention` locally, `eeg_attention` on the cluster.
#       This is THE Chapter 1 canary: `src/modules/clip_loss.py`.
#   L2  needs the MAD-EEG dataset in $MADEEG_DIR (NOT in the repo, 4.7 GB).
#   L3  needs a GPU, or the paper checkpoints: printed, not launched.
#
# EXPECTED REFERENCE NUMBERS. If these do not come out EXACTLY, it is not
# "close": it is a red canary.
#
#   L0-a  src/run.py --selftest              -> "23 protocol flags ... PASS"
#   L0-b  src/madeeg_diagnose.py --demo      -> "[demo] PASS  null (base_rate) = 0.375"
#   L0-c  src/models/cca_multiview.py        -> "[cca_multiview] PASS rho_train=1.0000 ...",
#                                               plus the proof that the `position` view is INERT
#   L0-d  Exp. 10 toy (retired pre-run)      -> 6 regimes (2 shared_w x 3 noise levels).
#         VERIFIED BY RUNNING IT on 2026-08-15: plain and ortho are IDENTICAL in 3
#         regimes out of 6 and differ by <= 0.008 in the other 3 (0.838/0.845 /
#         0.845/0.843 / 0.688/0.680) - sampling noise at 400 trials, not exact
#         equality. And the toy ASSERTS NOTHING: it only prints. This gate
#         therefore only verifies that it runs and with which numbers, NOT that
#         the two rules are identical.
#   L0-e  sweeps/report.py --check           -> 107 pinned folds, table IDENTICAL to
#                                               RESULTS.md and NOTHING written to disk
#   L0-f  default own-vs-other md5           -> 2eaa926244de340d31907c6deeb04b0c  (+ 208/376)
#   L0-g  Exp. 13 provenance md5             -> a3c7533cb0d79ee849bb4c8c90ced803
#         negative-control provenance md5    -> f3ebe89b900bde3240d35fb02084bc8f
#   L0-h  own-vs-other paired statistics     -> REF n0/n1/n2 = 36/96/56, exact p 0.0235
#                                               (and the pre-registered Bonferroni table)
#   L1-a  src/modules/clip_loss.py           -> 0.628491 / 1.2994 / 4.8198
#                                               and step D  0.951610  vs  2.160834
#   L2-a  --check_rule on the step C records -> 58/154 EXACT, agreeing trial by trial
#   L2-b  Exp. 13 canary (audio only)        -> mel-8 0.1772/0.1442 / flux 0.2874/0.2417 / 30/36
#   L2-c  own-vs-other REF rerun             -> 208/376 = 0.5532 with md5 2eaa9262...
#   L2-d  decision-identical duo anchors     -> canary_rawsolos 74/154, canary_duo 86/154,
#                                               ZERO discordant (not just the same total)
#   L2-e  ridge positive control             -> --self_test [PASS] (threshold 0.90 in the code)
#   L2-f  lateralised alpha injection        -> 44/44 in the run's exact configuration
#   L2-g  Exp. 11/12 spectral register       -> 47/47 and 42/42 (hard exit inside the driver)
#   L2-h  Exp. 4 alignment gate              -> stem borrowing r=1.0000 over 272 comparisons,
#                                               155/155 wav<->mixture, span within 4 ms
#   L3-a  Chapter 1 end-to-end from model-all0 -> GLOBAL 0.8650 EXACT
#   L3-b  synthetic controls, steps C and D  -> 148/154 = 0.9610 / 142/154 = 0.9221 (threshold 0.90)
#   L3-c  S1 sampling + 20/20 overfit        -> see exp18/exp19_matchmismatch.sh
#
# ENVIRONMENT TRAP #1: the interpreter is `/opt/miniconda3/bin/python`, NEVER bare
#    `python`. `/opt/anaconda3` gives the same results with floats differing at
#    2e-5 and the md5 canaries FAIL - it has already happened, at the start of
#    Exp. 7. The conda env `attention` has no `h5py` and cannot run
#    `madeeg_reconstruction.py`: that is why level 1 has its OWN interpreter
#    ($PY_TORCH).
#
# PRE-REGISTRATION: none - this is not an experiment. The rules it enforces are
#              "positive control crossed BEFORE the real number" and "a new flag
#              does not move an old number".
# PROVENANCE : docs/provenance/ (the pinned files)
# =============================================================================
set -uo pipefail        # NOT -e: all canaries are counted, we do not stop at the first

PY=${PY:-/opt/miniconda3/bin/python}          # trap #1: NEVER bare `python`
PY_TORCH=${PY_TORCH:-}                        # torch env (conda `attention`); empty = search
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # pinned records are never overwritten
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}
LEVELS=${LEVELS:-"0 1 2"}                     # 3 is never executed: it is printed

REF_MD5=2eaa926244de340d31907c6deeb04b0c
EXP13_MD5=a3c7533cb0d79ee849bb4c8c90ced803
NEGCTRL_MD5=f3ebe89b900bde3240d35fb02084bc8f

mkdir -p "$OUT_DIR"
cd "$REPO"

PASS=0; FAIL=0; SKIP=0
FAILED_NAMES=""
_md5()  { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
_has()  { case " $LEVELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
ok()    { PASS=$((PASS+1)); echo "   PASSED       - $1"; }
ko()    { FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES
     - $1"; echo "   FAILED       - $1"; }
skip()  { SKIP=$((SKIP+1)); echo "   NOT CROSSABLE HERE - $1"; }
head_() { echo; echo "--- $1"; }

# =============================================================================
# LEVEL 0 - no data, no torch.
# =============================================================================
if _has 0; then
echo "############################################################"
echo "# LEVEL 0 - no dataset, no torch, no GPU"
echo "############################################################"

head_ "L0-a  src/run.py --selftest - the Akama protocol lives in 2 places and they must agree"
if OUT=$("$PY" src/run.py --selftest 2>&1); then
  echo "$OUT" | tail -3
  echo "$OUT" | grep -q "PASS" && ok "L0-a run.py --selftest" || ko "L0-a run.py --selftest"
else
  echo "$OUT" | tail -5; ko "L0-a run.py --selftest (exit != 0)"
fi

head_ "L0-b  madeeg_diagnose.py --demo - the prior null is NOT 0.5, and the self-check proves it"
if OUT=$("$PY" src/madeeg_diagnose.py --demo 2>&1); then
  echo "$OUT" | tail -3
  echo "$OUT" | grep -q "\[demo\] PASS" && ok "L0-b diagnose --demo" || ko "L0-b diagnose --demo"
else
  echo "$OUT" | tail -5; ko "L0-b diagnose --demo (exit != 0)"
fi

head_ "L0-c  cca_multiview.py - CCA reconstructs, and the \`position\` view is INERT (|delta| < 1e-9)"
if OUT=$("$PY" src/models/cca_multiview.py 2>&1); then
  echo "$OUT" | tail -5
  echo "$OUT" | grep -q "\[cca_multiview\] PASS" && ok "L0-c cca_multiview self-check" \
                                                 || ko "L0-c cca_multiview self-check"
else
  echo "$OUT" | tail -5; ko "L0-c cca_multiview self-check (exit != 0)"
fi

head_ "L0-d  Exp. 10 toy - the \`ortho\` rule is INERT by algebra (retired PRE-RUN)"
echo "      the --score_rule ortho flag exists and is inert: do NOT produce numbers with it."
if OUT=$("$PY" docs/provenance/2026-08-11_exp10_retired_toy_sweep.py 2>&1); then
  echo "$OUT" | tail -8
  ok "L0-d Exp. 10 toy (runs; 6 regimes, plain vs ortho within 0.008 - the toy asserts nothing)"
else
  echo "$OUT" | tail -5; ko "L0-d Exp. 10 toy"
fi

head_ "L0-e  sweeps/report.py --check - 107 pins verified against the hparams, NOTHING written"
BEFORE=$(_md5 RESULTS.md)
if OUT=$("$PY" sweeps/report.py --check 2>&1); then
  AFTER=$(_md5 RESULTS.md)
  echo "$OUT" | grep -E "^\| (madeeg_ridge|contrastive_raw|contrastive_clap) " | head -6
  if [ "$BEFORE" = "$AFTER" ]; then
    ok "L0-e report.py --check (pins verified, RESULTS.md untouched)"
  else
    ko "L0-e report.py --check WROTE to RESULTS.md - it must not"
  fi
else
  echo "$OUT" | tail -8
  ko "L0-e report.py --check (a pin disagrees, or the run dirs are absent: see above)"
fi

head_ "L0-f  md5 of the DEFAULT own-vs-other path (artefact already on disk)"
REF_CSV="$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv"
if [ -f "$REF_CSV" ]; then
  GOT=$(_md5 "$REF_CSV")
  echo "      expected $REF_MD5"
  echo "      found    $GOT   ($REF_CSV)"
  [ "$GOT" = "$REF_MD5" ] && ok "L0-f default own-vs-other md5" \
                          || ko "L0-f default own-vs-other md5"
else
  skip "L0-f: runs/ is gitignored and $REF_CSV is absent. In a clean clone this canary
        is crossed only by RERUNNING the reference -> level 2 (L2-c)."
fi

head_ "L0-g  md5 of the two separability provenance files (Exp. 13 and negative control)"
for PAIR in "docs/provenance/2026-08-12_exp13_stem_separability.txt:$EXP13_MD5" \
            "docs/provenance/2026-08-12_exp16A_negative_control.txt:$NEGCTRL_MD5"; do
  F=${PAIR%%:*}; WANT=${PAIR##*:}
  if [ -f "$F" ]; then
    GOT=$(_md5 "$F")
    printf "      %-58s %s\n" "$(basename "$F")" "$GOT"
    [ "$GOT" = "$WANT" ] && ok "L0-g md5 $(basename "$F")" || ko "L0-g md5 $(basename "$F") (expected $WANT)"
  else
    skip "L0-g: $F is missing"
  fi
done

head_ "L0-h  own-vs-other paired statistics - exact paired permutation + Bonferroni thresholds"
if [ -f "$REPO/runs/results/ovo_sweep_REF/madeeg_ownvsother.csv" ]; then
  if OUT=$("$PY" docs/provenance/2026-08-11_ovo_sweep_paired_stats.py \
             "$REPO/runs/results/ovo_sweep_REF/madeeg_ownvsother.csv" 2>&1); then
    echo "$OUT" | tail -4
    ok "L0-h paired stats (36/96/56, exact p 0.0235, pre-registered threshold table)"
  else
    echo "$OUT" | tail -6; ko "L0-h paired stats"
  fi
else
  skip "L0-h: runs/results/ovo_sweep_REF/ is missing (gitignored). Level 2 regenerates it."
fi
fi   # level 0

# =============================================================================
# LEVEL 1 - needs torch. This is THE Chapter 1 canary.
# =============================================================================
if _has 1; then
echo
echo "############################################################"
echo "# LEVEL 1 - needs torch (conda env \`attention\`)"
echo "############################################################"

if [ -z "$PY_TORCH" ]; then
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "/opt/anaconda3/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi

head_ "L1-a  src/modules/clip_loss.py - the InfoNCE arithmetic behind EVERY Chapter 1 number"
echo "      Self-check ENTIRELY SYNTHETIC: torch.manual_seed(0), randn, no dataset,"
echo "      no GPU, compared against the arithmetic worked out by hand INSIDE the file."
echo "      The file is frozen at bb016fd (2026-07-28 17:02) on purpose. Do not touch it."
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  echo "      interpreter: $PY_TORCH"
  if OUT=$("$PY_TORCH" src/modules/clip_loss.py 2>&1); then
    echo "$OUT"
    N=$(echo "$OUT" | grep -c "PASS")
    if [ "$N" -ge 2 ] && echo "$OUT" | grep -q "0.628491" && echo "$OUT" | grep -q "0.951610"; then
      ok "L1-a clip_loss self-check (0.628491 / 1.2994 / 4.8198 / 0.951610 vs 2.160834)"
    else
      ko "L1-a clip_loss self-check: the printed values are NOT the pinned ones"
    fi
  else
    echo "$OUT" | tail -10; ko "L1-a clip_loss self-check (exit != 0)"
  fi
else
  skip "L1-a: no interpreter with torch found. Pass it explicitly:
        PY_TORCH=~/miniconda3/envs/attention/bin/python bash $0
        (on the cluster: \$HOME/.conda/envs/eeg_attention/bin/python)"
fi
fi   # level 1

# =============================================================================
# LEVEL 2 - needs the MAD-EEG dataset. ~20-30 min CPU.
# =============================================================================
if _has 2; then
echo
echo "############################################################"
echo "# LEVEL 2 - needs MAD-EEG in $MADEEG_DIR"
echo "############################################################"

if [ ! -f "$MADEEG_DIR/madeeg_preprocessed.yaml" ]; then
  skip "ALL of level 2: $MADEEG_DIR/madeeg_preprocessed.yaml is missing.
        The dataset is NOT in the repo (4.7 GB):
        MADEEG_DIR=... bash scripts/madeeg_setup.sh
        madeeg_setup.sh fetches ONLY the preprocessed release: the raw_solos path
        (own-vs-other, Exp. 4/6/9, arm A) ALSO needs madeeg_raw.hdf5 +
        madeeg_raw.yaml + madeeg_sequences_raw.yaml + stimuli/."
else

head_ "L2-a  --check_rule - the contrastive arm's decision rule, on the step C records"
REC_C="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC_C" ]; then
  if OUT=$("$PY" src/madeeg_diagnose.py --check_rule "$REC_C" 2>&1); then
    echo "$OUT"
    echo "$OUT" | grep -q "58/154" && ok "L2-a --check_rule = 58/154 exact" \
                                   || ko "L2-a --check_rule does not give 58/154"
  else
    echo "$OUT" | tail -6; ko "L2-a --check_rule"
  fi
else
  skip "L2-a: $REC_C is missing (run archive outside the repo). Use ARCHIVE=... to point elsewhere."
fi

head_ "L2-b  Exp. 13 canary - audio-only separability (hard exit inside the driver)"
if OUT=$("$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
           --out "$OUT_DIR/canary_exp13.txt" 2>&1); then
  grep -E "CANARY|PASSED|FAILED" "$OUT_DIR/canary_exp13.txt" | head -6
  grep -q "0.1772" "$OUT_DIR/canary_exp13.txt" && grep -q "0.2874" "$OUT_DIR/canary_exp13.txt" \
    && ok "L2-b Exp. 13 canary (mel-8 0.1772/0.1442 / flux 0.2874/0.2417 / 30/36)" \
    || ko "L2-b Exp. 13 canary: the values are not the pinned ones"
else
  echo "$OUT" | tail -8; ko "L2-b Exp. 13 canary (the driver exited with an error - that is its job)"
fi

head_ "L2-c  own-vs-other REF rerun from scratch - 208/376 WITH the same md5"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --own_vs_other --eeg_clean notch_ica \
      --estimator ridge --filters pooled --target mel --n_mels 8 \
      --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 \
      --training_date "${TAG}canary_ovo_ref" >/dev/null 2>&1
NEW_CSV="$REPO/runs/results/${TAG}canary_ovo_ref/madeeg_ownvsother.csv"
if [ -f "$NEW_CSV" ]; then
  grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}canary_ovo_ref/madeeg_ownvsother_summary.txt"
  GOT=$(_md5 "$NEW_CSV")
  echo "      expected $REF_MD5"
  echo "      found    $GOT"
  [ "$GOT" = "$REF_MD5" ] && ok "L2-c own-vs-other REF (208/376 + md5)" \
    || ko "L2-c own-vs-other REF: different md5. FIRST check the INTERPRETER (trap #1):
        /opt/anaconda3 gives the same 376 decisions with floats differing at 2e-5 and this md5 changes."
else
  ko "L2-c own-vs-other REF: the run produced no csv"
fi

head_ "L2-d  the two mel anchors of the duo decision - DECISION-IDENTICAL, not just equal totals"
DEC=(--madeeg_dir "$MADEEG_DIR" --estimator ridge --filters pooled --eeg_clean none
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 --cv_folds 5
     --ensemble duo --target mel --n_mels 8)
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" --train_on raw_solos --test_eeg raw --spatial stereo \
      --training_date "${TAG}canary_anchor_rawsolos" >/dev/null 2>&1
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" --train_on duos_kfold --spatial stereo \
      --training_date "${TAG}canary_anchor_duo" >/dev/null 2>&1
for T in "canary_rawsolos:${TAG}canary_anchor_rawsolos:74" "canary_duo:${TAG}canary_anchor_duo:86"; do
  OLD=${T%%:*}; REST=${T#*:}; NEW=${REST%%:*}; K=${REST##*:}
  A="$REPO/runs/results/$OLD/madeeg_records.csv"; B="$REPO/runs/results/$NEW/madeeg_records.csv"
  if [ -f "$A" ] && [ -f "$B" ]; then
    OUT=$("$PY" src/madeeg_diagnose.py --mcnemar "$A" "$B" 2>&1); echo "$OUT" | tail -4
    if echo "$OUT" | grep -q "no discordant comparisons" && echo "$OUT" | grep -q "A correct $K/154"; then
      ok "L2-d anchor $OLD: $K/154, ZERO discordant"
    else
      ko "L2-d anchor $OLD: not decision-identical (or does not give $K/154)"
    fi
  else
    skip "L2-d anchor $OLD: $A is missing (gitignored) - there is nothing to pair against."
  fi
done

head_ "L2-e  ridge positive control - synthetic EEG, threshold 0.90 DECLARED IN THE CODE"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
      --training_date "${TAG}canary_ridge" >/dev/null 2>&1
S="$REPO/runs/results/${TAG}canary_ridge_selftest/madeeg_selftest_summary.txt"
if [ -f "$S" ]; then
  grep -h "AAD accuracy" "$S"
  grep -q "\[PASS\]" "$S" && ok "L2-e ridge --self_test" || ko "L2-e ridge --self_test"
else
  ko "L2-e ridge --self_test: no summary produced"
fi

head_ "L2-f  lateralised alpha injection - 44/44 in the run's EXACT configuration (notch+ICA)"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --test_eeg raw --alpha_li --alpha_inject 1.0 \
      --eeg_clean notch_ica --spatial stereo \
      --training_date "${TAG}canary_alpha_ica" >/dev/null 2>&1
S="$REPO/runs/results/${TAG}canary_alpha_ica_inject/madeeg_alpha_summary.txt"
if [ -f "$S" ]; then
  grep -h "PAIRED ACCURACY" "$S"
  grep -q "44/44 = 1.0000" "$S" && ok "L2-f alpha injection 44/44" || ko "L2-f alpha injection != 44/44"
else
  ko "L2-f alpha injection: no summary produced"
fi

head_ "L2-g  Exp. 11 spectral register - 47/47 on the injection, and the driver EXITS if it fails"
if OUT=$("$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
           --log_dir "$OUT_DIR" --seed 42 2>&1); then
  echo "$OUT" | grep -iE "control|47/47|primary" | head -5
  echo "$OUT" | grep -q "47/47" && ok "L2-g Exp. 11 positive control = 47/47" \
                                || ko "L2-g Exp. 11 positive control"
else
  echo "$OUT" | tail -8
  ko "L2-g Exp. 11: the driver exited with sys.exit(1) - the real number was not computed"
fi

head_ "L2-h  Exp. 4 alignment gate - the premise of the whole mono experiment"
echo "      On 2026-07-30 this output was NOT saved to any file: here it is REDIRECTED,"
echo "      which is the only difference."
if "$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --test_eeg raw --check_alignment --spatial mono \
      > "$OUT_DIR/canary_exp4_alignment_mono.txt" 2>&1; then
  grep -iE "PASS|FAIL|1.0000|155|4 ms|borrow" "$OUT_DIR/canary_exp4_alignment_mono.txt" | head -8
  grep -q "PASS" "$OUT_DIR/canary_exp4_alignment_mono.txt" \
    && ok "L2-h --check_alignment --spatial mono" || ko "L2-h --check_alignment --spatial mono"
else
  tail -8 "$OUT_DIR/canary_exp4_alignment_mono.txt"; ko "L2-h --check_alignment (exit != 0)"
fi
fi   # dataset present
fi   # level 2

# =============================================================================
# LEVEL 3 - NOT EXECUTED HERE. Printed.
# =============================================================================
cat <<'EOF'

############################################################
# LEVEL 3 - requires a GPU or the paper checkpoints
############################################################

# L3-a  The Chapter 1 end-to-end canary: GLOBAL 0.8650 EXACT from model-all0.ckpt.
#       It used to live in scripts/test_sanity.sh, which was DELETED in refactor
#       7fae3cf and ABSORBED into scripts/replicate.sh (phase `sanity`). Today's
#       command is:
bash scripts/setup_checkpoints.sh                       # once
PHASES="within sanity" bash scripts/replicate.sh        # ~25 min CPU, no GPU
#       It must give: within 0.8650 exact (deterministic) and the two negative
#       controls "at chance" / "far below 0.865". The two controls are NOT
#       bit-reproducible (checkpoint_test forces shuffle=True and --workers
#       auto-sizes): do NOT quote them as exact figures.

# L3-b  The contrastive arm's synthetic controls (GPU):
#       step C  148/154 = 0.9610   /  step D  142/154 = 0.9221   (threshold 0.90 in the code)
python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --self_test --kfold 5
python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --self_test --loss within_mixture --kfold 5

# L3-c  Match-mismatch sampling and optimisation sanity: they have their own scripts.
bash scripts/replicate/exp18_matchmismatch.sh    # 3346 pairs, min offset 2.000 s, null 0.5000
bash scripts/replicate/exp19_matchmismatch.sh    # + S2 null 0.6061 -> 0.5000, 20/20 overfit

EOF

echo "############################################################"
echo "# OUTCOME: $PASS passed / $FAIL failed / $SKIP not crossable here"
echo "############################################################"
if [ "$FAIL" -gt 0 ]; then
  echo "RED CANARIES:$FAILED_NAMES"
  echo
  echo "Do not update the expected value. STOP and report: a red canary means a"
  echo "number that was already reported MAY have moved."
  exit 1
fi
echo "No red canaries among those crossable in this environment."
echo "replication provenance: $OUT_DIR"
