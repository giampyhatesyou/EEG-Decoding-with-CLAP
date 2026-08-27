#!/usr/bin/env bash
# =============================================================================
# Exp. 16 arm B - does a richer EEG feature extraction help? band_power.
#
# WHAT IT MEASURES: whether adding the `band_power` view (per-channel log power,
#   one block per band) to the multi-view CCA raises tracking on own-vs-other.
#   It is the honest version of "use the best EEG feature extraction":
#   `--eeg_repr spectra` lives in the contrastive model - the family that memorises
#   the song - and is EXCLUDED by the criterion.
# COST       : ~7 min CPU (measured: 4 own-vs-other runs + 1 self-test + 4 McNemar).
# BUDGET     : NO - own-vs-other runs on the SOLOS, free training material.
#              No duo decided, no trio, no split. The TRIOS are never touched.
# GPU        : no.
#
# BUT IT IS NOT FREE IN FAMILY-WISE ERROR RATE: together with Exp. 7 (8 candidates),
#    Exp. 8 (2) and Exp. 14 (2), this is the 13th comparison on the SAME 376
#    decisions. Any sentence of the form "we found something that beats 208/376" must
#    be reported with that count next to it.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   PRELIMINARY GATE: the pinned 209/376 comes from `--cca_views eeg_lagged` ALONE
#     (read in runs/results/ovo_cca_ica/madeeg_ownvsother_summary.txt). Had it
#     already contained band_power, the criterion would have closed the arm without
#     running anything.
#   CANDIDATE (K=1): CCA `eeg_lagged,band_power` -> 202/376 = 0.5372
#     NULL 0.500 EXACT by symmetry (every pair in both directions)
#     one-sided exact binomial p 0.08185   BAR >= 208/376 -> NOT CROSSED.
#   References re-measured today: ridge mel-8 208/376 (p 0.02208, = the bar),
#     single-view CCA 209/376 (p 0.01717).
#   DESCRIPTIVE McNemar, all four directions, none significant:
#     vs ridge mel-8    : cand beats ref 66/138 (p 0.7243), ref beats cand 72/138 (p 0.3353)
#     vs single-view CCA: cand beats ref 43/93  (p 0.7965), ref beats cand 50/93  (p 0.2670)
#   THE CAUSE, and it is the line worth more than the verdict: mean rho(own)/rho(other)
#     goes from 0.0666/0.0112 = 5.9 (single view) to 0.0629/0.0058 = 10.8 (multi-view).
#     THE MEAN SEPARATION NEARLY DOUBLES AND THE ACCURACY DROPS. The decision is an
#     argmax between two scores of the same estimator: it is scale-invariant, so a
#     higher mean rho is NOT a better decoder.
#
# WHAT WAS ACTUALLY TESTED: `band_power` INSIDE the pre-registered 1-8 Hz band, i.e.
#    ONLY delta (1-4) and theta (4-8) - blocks_x=[('eeg_lagged',340),
#    ('band_power',40)] = 20 channels x 2 bands. Alpha and beta are ZERO by
#    construction inside 1-8 Hz and the module discards them, saying so. Widening
#    --band_high is a SEPARATE, declared change, and was NOT done here.
#
# CANARIES   : three, INSIDE the driver, crossed BEFORE the real numbers; the driver
#              exits with an error if one fails.
#              G1 default path: 208/376 + md5 2eaa926244de340d31907c6deeb04b0c
#              G2 (added) single-view CCA re-measured: 209/376 and ZERO discordant
#                 decisions against the archived run of 2026-08-10 (identical md5:
#                 dbbe6400764c94876a569aa4e6ff4c7b)
#              G3 positive control of the MULTI-VIEW CCA path, threshold >= 0.95:
#                 1.0000 (entirely synthetic EEG: not a result)
# PRE-REGISTRATION: "Chapter 2 - Exp. 16: the thesis premise put to the test - CLAP
#              as a target and rich EEG features, pre-registered criterion
#              (2026-08-12)", section B
# PROVENANCE : docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt (pinned),
#              runs/results/exp16b_*/
#
# The driver does NOT accept --log_dir: it rewrites runs/results/exp16b_*. The PINNED
#    provenance file is not touched: --out goes to $OUT_DIR.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp16b_ccaviews_ownvsother.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp16b_ccaviews.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "PASSED|FAILED|/376|MISSES|CLEARS|mean rho|VERDICT|B FAILS|B PASSES" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt"
