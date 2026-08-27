#!/usr/bin/env bash
# =============================================================================
# Exp. 17 - CLAP in the separability gate, with the criterion REMADE invariant to
#           dimensionality. (AUDIO ONLY)
#
# WHAT IT MEASURES: the same question as Exp. 13 with ONE candidate - CLAP-512, the
#   thesis premise - but judged on a new criterion, because the old one was
#   FALSIFIED: the negative control of 2026-08-12 shows that 512 dimensions of NOISE
#   satisfy it (36/36 and gap -0.0002), and CLAP has exactly 512 dimensions. It is
#   not a moved goalpost: when the new criterion was written NO CLAP NUMBER EXISTED.
# COST       : stage 1 ~9 min on the cluster (CPU) + stage 2 ~11 s locally.
# BUDGET     : NO - zero. It opens the HDF5 only for `soli` and the metadata: never
#              ['response'], never a trial, never a label. No duo, no trio.
# GPU        : no. The cluster extraction runs on CPU (the login node has no GPU).
#
# THE CRITERION, all three conditions (written before the code):
#   1. |corr| within-duo below mel-8 in >= 26/36   (P = 0.005665) - unchanged
#   2. within/floor <= 1.80  (>= 60% of the mel-8 distance 3.131 -> noise 0.922)
#   3. within/floor >= 1.20  - MANDATORY MARGIN OVER NOISE: below this threshold the
#      candidate is declared INDISTINGUISHABLE FROM NOISE and does NOT win.
#
# THE GATES, in order, with hard exit:
#   1. Exp. 13 canary: mel-8 0.1772/0.1442, flux 0.2874/0.2417, 30/36 (+/-0.0010)
#   2. the negative control RERUN at 512 dims IN THE SAME RUN: within/floor inside
#      [0.85, 1.05], otherwise the wiring has changed -> the script exits with an
#      error BEFORE the CLAP row is computed
#   3. the extraction's manifest.json (checkpoint + sha256, versions, sr, window,
#      hop, machine): without it the measurement does not start
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   checkpoint 630k-audioset-best.pt  1,863,587,645 bytes
#     sha256 8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037
#   CANARY   mel-8 mean 0.1772 median 0.1442, flux 0.2874/0.2417, 30/36
#   N1 noise-512   within/floor 0.922
#   C7 CLAP-512    see docs/provenance/2026-08-12_exp17_clap_separability.txt
#
# THE LIMIT THAT MUST BE WRITTEN IN EVERY BRANCH OF THE OUTCOME: CLAP's design
#   aperture is 10 s, 160x the window used here, and below 10 s `laion_clap` applies
#   `repeatpad`. WE ARE MEASURING CLAP OUTSIDE THE REGIME IT WAS TRAINED FOR.
#   A failure here is NOT evidence against CLAP at 10 s.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 17: CLAP in the separability gate, criterion
#              REMADE invariant to dimensionality (2026-08-12)"
# PROVENANCE : docs/provenance/2026-08-12_exp17_clap_separability.txt,
#              full extraction log:
#              docs/provenance/2026-08-12_exp17_baldo_extraction.log
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # pinned provenance is never touched
CLAP_DIR=${CLAP_DIR:-$REPO/runs/clap_exp17}   # stage 1's .npy files + manifest.json
OUT="$OUT_DIR/exp17_clap_separability.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ ! -f "$CLAP_DIR/manifest.json" ]; then
  cat <<EOF
manifest.json not found in $CLAP_DIR.

STAGE 1 (extraction) runs on the cluster, not here: laion_clap is not installed in
/opt/miniconda3 and nothing gets installed into /opt/miniconda3 (environment trap #1).
The command, exactly as executed on 2026-08-12 (CPU, no GPU, no SLURM):

  ssh <cluster>
  curl -L -o ~/clap_ckpt/630k-audioset-best.pt \\
       https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
  sha256sum ~/clap_ckpt/630k-audioset-best.pt   # 8053c977...c3f037
  ~/.conda/envs/eeg_attention/bin/python \\
      ~/EEG-Attention-decoding-with-CLAP/src/madeeg_exp16a_clap_extract.py \\
      --madeeg_dir ~/madeeg --out_dir ~/clap_exp17 \\
      --ckpt ~/clap_ckpt/630k-audioset-best.pt

then copy the .npy files and manifest.json back:

  scp -r <cluster>:~/clap_exp17/ $CLAP_DIR

The full log of that execution is in
docs/provenance/2026-08-12_exp17_baldo_extraction.log.
EOF
  exit 1
fi

"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --clap_dir "$CLAP_DIR" --exp17 --out "$OUT"

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
grep -E "CANARY|PASSED|FAILED|GATE 2|noise-512|CLAP-512|VERDICT" "$OUT" || true
echo
echo "replication provenance : $OUT"
echo "pinned provenance      : docs/provenance/2026-08-12_exp17_clap_separability.txt"
echo
echo "Whatever the outcome: CLAP was measured at 62.5 ms, 160x below its design"
echo "   aperture (10 s), where laion_clap applies repeatpad. A failure is NOT"
echo "   evidence against CLAP at 10 s."
