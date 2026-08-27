#!/usr/bin/env bash
# =============================================================================
# Exp. 15 (EXPLORATORY) - the attentional differential under MFCC-13.
#
# WHAT IT MEASURES: whether the representation that best separates the stems
#   (Exp. 13) and tracks as well as mel (Exp. 14) raises the standardised
#   ATTENTIONAL DIFFERENTIAL D = mean(d)/sd(d) on the duo decision, with a paired
#   sign-flip permutation. The primary is NOT accuracy: the "can this test be won?"
#   check had predicted +1.4 trials against a bar that asks for +4.
# COST       : ~2-3 min CPU (measured from the 2026-08-12 timestamps: first run
#              13:23:20 -> last 13:25:05 = 1m45s, plus the first untimed run).
# BUDGET     : YES - it decides on the DUOS. All already spent: EXPLORATORY BY
#              CONSTRUCTION, written in the title and in the file. The TRIOS are
#              never touched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRIMARY (P2, duo k-fold):  D(mel-8) = +0.1001   D(MFCC-13) = -0.1119
#     difference -0.2120   NULL = 0 EXACT by construction   p = 0.9833
#     (sign-flip permutation, B = 10000, seed 20260812)
#     -> does not reject the null; and the estimate points OPPOSITE to the hypothesis.
#   DETECTABILITY FLOOR (mandatory): MDD = 0.2467 in D at n = 154, power 0.80,
#     alpha 0.05 one-sided; sd of the paired differences 1.2312.
#   DESCRIPTIVE SECONDARIES: accuracy P2 MFCC 65/154 against the mel anchor 86/154
#     (the criterion's prediction was 87.4/154, the Exp. 9 bar was 90/154);
#     P1 MFCC 75/154 against the anchor 74/154; McNemar "mel beats MFCC" 45/69 p = 0.0077.
#   OUTCOME: not conclusive for power (MDD 0.2467 > predicted effect), but the
#   estimate has the opposite sign. The predictive model behind the prediction was
#   RETRACTED the same day (aggregate rho used as if it were per-dimension).
#
# CANARIES   : two, INSIDE the driver, crossed BEFORE the real numbers; the driver
#              exits with an error if one fails.
#              G1 mel anchors decision-identical to Exp. 9: 74/154 and 86/154 with
#                 ZERO discordant decisions, compared TRIAL BY TRIAL.
#              G2 positive control of the MFCC path on the duo decision,
#                 threshold >= 0.95: 1.0000.
# PRE-REGISTRATION: "Chapter 2 - Exp. 15 (EXPLORATORY): the attentional differential
#              under MFCC, pre-registered criterion (2026-08-12)"
# PROVENANCE : docs/provenance/2026-08-12_exp15_mfcc_differential.txt (pinned),
#              runs/results/exp15_*/
#
# The driver does NOT accept --log_dir: it rewrites runs/results/exp15_*. The PINNED
#    provenance file is not touched: --out goes to $OUT_DIR.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp15_mfcc_differential.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp15_differential.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "PASSED|FAILED|D\(|observed statistic|MDD|p = |VERDICT" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-12_exp15_mfcc_differential.txt"
