#!/usr/bin/env bash
# =============================================================================
# Exp. 7 - configuration sweep on own-vs-other (8 candidates vs the reference).
#
# WHAT IT MEASURES: whether a different configuration of the linear front end
#   (band, mel bands, target rate, estimator) beats the 208/376 reference on the
#   discriminative own-vs-other task, judged with paired McNemar on the SAME 376
#   comparisons and a Bonferroni alpha of 0.05/8 = 0.00625.
# COST       : ~15-40 min CPU. The 7 runs at 64 Hz are ~1 min each (estimated from
#              the Exp. 14 run timestamps of 2026-08-12); C5 and C7 run at
#              --target_fs 256 (~6 GB of design matrix) and are noticeably slower:
#              duration TO BE CONFIRMED on first run.
#              Needs 16 GB of RAM: C5 and C7 are last in the sequence on purpose.
#              A candidate that does not run is declared "not executed", not replaced.
# BUDGET     : NO - own-vs-other runs on held-out SOLO segments (structural assert
#              --train_on raw_solos): no duo loaded, no attention decision, trios
#              untouched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   REF  208/376 = 0.5532   md5(madeeg_ownvsother.csv) = 2eaa926244de340d31907c6deeb04b0c
#   C1 209 / C2 212 / C3 213 / C4 204 / C5 203 / C6 203 / C7 203 / C8 203  (out of 376)
#   discordants / candidate wins: C1 23/45 / C2 28/52 / C3 33/61 /
#   C4 14/32 / C5 34/73 / C6 21/47 / C7 35/75 / C8 33/71
#   OUTCOME: NOT CONCLUSIVE - no candidate passes Bonferroni (not even 0.05).
#   Null 0.500 EXACT by symmetry (every pair evaluated in both directions).
#
# CANARY     : the reference is rerun FIRST and must give exactly 208/376 with an
#              md5 identical to runs/results/ovo_ridge_ica/. If it does not, the
#              script stops and computes no candidate.
# PRE-REGISTRATION: "Chapter 2 - Exp. 7: configuration sweep on own-vs-other,
#              pre-registered criterion (2026-08-11)"
# PROVENANCE : docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt (pinned),
#              runs/results/ovo_sweep_{REF,C1..C8}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # where the replication McNemar lands
TAG=${TAG:-repl_}                             # --training_date prefix: pinned records
                                              # are never overwritten
REF_CSV=${REF_CSV:-$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv}
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

# The pre-registered invariants: they do NOT move. --filters is forced to pooled by
# the --own_vs_other assert; the seed must be identical across runs or the paired
# test would compare different comparisons.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
      --eeg_clean notch_ica --filters pooled --target mel
      --band_low 1 --lags_ms 250 --seed 42)

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

run() {  # run <name> <candidate flags...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "$@" --training_date "${TAG}${name}"
}

acc()  { grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}$1/madeeg_ownvsother_summary.txt"; }
csv()  { echo "$REPO/runs/results/${TAG}$1/madeeg_ownvsother.csv"; }

# ---- REGRESSION CANARY, CROSSED FIRST ---------------------------------------
run ovo_sweep_REF --estimator ridge --n_mels 8 --target_fs 64 --band_high 8
GOT=$(_md5 "$(csv ovo_sweep_REF)")
echo; echo "=== CANARY: the reference must be bit-for-bit the pinned one ==="
acc ovo_sweep_REF
echo "  md5 obtained = $GOT"
echo "  md5 expected = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FAILED] the default path has changed: NO candidate is looked at."
  echo "     STOP and report."
  exit 1
fi
if [ -f "$REF_CSV" ] && ! cmp -s "$(csv ovo_sweep_REF)" "$REF_CSV"; then
  echo "  -> [FAILED] the csv differs from runs/results/ovo_ridge_ica/. STOP."
  exit 1
fi
echo "  -> [PASSED] this licenses the wiring and nothing else."

# ---- THE 8 CANDIDATES - K = 8, closed by the criterion, never widened -------
run ovo_sweep_C1 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 13
run ovo_sweep_C2 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 20
run ovo_sweep_C3 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 40
run ovo_sweep_C4 --estimator ridge     --n_mels 24 --target_fs  64 --band_high  8
run ovo_sweep_C6 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels  8 --target_fs 64 --band_high  8
run ovo_sweep_C8 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels  8 --target_fs 64 --band_high 40
# C5 and C7 last: --target_fs 256, the 16 GB case on record.
run ovo_sweep_C5 --estimator ridge     --n_mels  8 --target_fs 256 --band_high  8
run ovo_sweep_C7 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels 24 --target_fs 256 --band_high 8

# ---- THE PAIRED TEST - one-sided McNemar, candidate vs REF -----------------
MC="$OUT_DIR/exp07_ovo_sweep_mcnemar.txt"
: > "$MC"
for C in C1 C2 C3 C4 C5 C6 C7 C8; do
  "$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_sweep_REF)" "$(csv "ovo_sweep_$C")" | tee -a "$MC"
done

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for C in REF C1 C2 C3 C4 C5 C6 C7 C8; do printf "%-4s " "$C"; acc "ovo_sweep_$C"; done
echo
echo "replication McNemar : $MC"
echo "pinned provenance   : docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt"
echo "NOTE: alpha = 0.05/8 = 0.00625. The threshold is the smallest integer with"
echo "      one-sided exact binomial p < alpha on the OBSERVED discordants: it has"
echo "      no degrees of freedom."
