#!/usr/bin/env bash
# =============================================================================
# Exp. 16 arm A - does CLAP separate the two stems? (AUDIO ONLY)
#
# STATUS: STAGE 1 WAS NEVER EXECUTED, and the reason is not compute. `laion_clap`
#    IS installed in /opt/anaconda3, but its default checkpoint is NOT cached
#    locally: `CLAP_Module.load_ckpt()` DOWNLOADS 630k-audioset-best.pt,
#    1,863,587,645 bytes (1.74 GiB), from
#    https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
#    (verified with a HEAD request: content-length 1863587645). Stage 1 REFUSES to
#    trigger the download by itself: it wants --ckpt on a file that already exists.
#
# WHAT IT WILL MEASURE: the same question as Exp. 13 with a seventh candidate -
#   CLAP, which is the thesis premise and which had never been tried as a target.
#   Criterion IDENTICAL to Exp. 13, untouched: >= 26/36 below mel-8 AND gap <= 0.05
#   on the cross-song floor of the SAME representation.
# BUDGET     : NO - audio only, no EEG opened. No duo, no trio.
# GPU        : no.
#
# TWO MANDATORY STAGES (environment trap #1: Anaconda produces floats differing at
# 2e-5 and breaks the md5 canaries; nothing gets installed into /opt/miniconda3):
#   stage 1  EXTRACTION in /opt/anaconda3 -> <out_dir>/<stem-md5>.npy + manifest.json
#   stage 2  MEASUREMENT in /opt/miniconda3 -> the Exp. 13 gate, which reads the .npy
#            as any other representation, and whose CANARY (mel-8 0.1772/0.1442,
#            flux 0.2874/0.2417, 30/36) is the proof that the MEASUREMENT environment
#            has not changed.
#
# THE RECIPE, FIXED BEFORE ANY NUMBER EXISTED:
#   hop 62.5 ms -> 16 Hz series -> Nyquist 8 Hz, exactly the ceiling of the
#     pre-registered band. MEASURED COST on this machine (HTSAT-tiny, batch 32,
#     random weights - stopwatch only, no scientific number): ~62 ms of CPU per
#     window x 8876 windows over the 555 s of stem audio = ~9 minutes, inside the
#     90-minute budget. So the declared fallback (hop 125 ms, band 1-4 Hz, mel-8
#     rerun in the same band) is NOT needed and was NOT used.
#   window 62.5 ms = hop, consecutive and non-overlapping. Three independent reasons,
#     all written in advance: (a) below 10 s laion_clap does `repeatpad`, i.e. it
#     REPEATS the waveform floor(480000/len) times - and 3000 samples divide 480000
#     exactly (160 repetitions), so ZERO padded zeros; (b) a long aperture W
#     attenuates as |sinc(fW)| and at 125 ms the first zero falls EXACTLY on 8 Hz,
#     annihilating the top of the band; at 62.5 ms the first zero is at 16 Hz;
#     (c) with window = hop the frames are disjoint and no correlation between
#     neighbouring frames is manufactured by overlap.
#   AND THE LIMIT THAT COMES WITH IT, declared now and not after the number: CLAP is
#     time-invariant BY DESIGN and its aperture is 10 s. 62.5 ms is 160x shorter.
#     It is not an implementation defect: it is the pre-registered band and CLAP's
#     time scale being 160x apart. Whatever comes out, THAT is part of the result.
#
# THE NUMBER THAT ALREADY EXISTS, AND THAT MUST BE READ BEFORE DOWNLOADING ANYTHING -
#    THE NEGATIVE CONTROL (runs in 11 s, no checkpoint needed):
#      512 dimensions of WHITE NOISE at 16 Hz, one extraction per stem, seed
#      20260812, no shared structure by construction:
#        criterion 1  36/36  (threshold 26/36)  -> SATISFIED
#        criterion 2  gap -0.0002 (threshold <= 0.05) -> SATISFIED
#        within/floor 0.922   (mel-8 3.131, MFCC-13 2.084)
#      So THE PRE-REGISTERED CRITERION IS SATISFIED BY NOISE at this dimensionality.
#      On its own it is not sufficient to distinguish "this representation separates
#      the two stems" from "this representation has many independent dimensions and a
#      small aggregate correlation". The criterion was NOT changed: it was measured,
#      and the `within/floor` column is the number a CLAP row will have to beat before
#      its PASS means anything.
#
# PRE-REGISTRATION: "Chapter 2 - Exp. 16 ... pre-registered criterion (2026-08-12)",
#              section A
# PROVENANCE : docs/provenance/2026-08-12_exp16A_negative_control.txt (exists),
#              docs/provenance/2026-08-12_exp16A_clap_separability.txt (a status
#              file, NOT yet a measurement - it becomes one when stage 1 runs)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: the MEASUREMENT lives here
PY_CLAP=${PY_CLAP:-/opt/anaconda3/bin/python} # the EXTRACTION, and only that, lives here
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
CLAP_DIR=${CLAP_DIR:-$REPO/runs/clap_exp16a}
CKPT=${CKPT:-}                                # the local .pt file. Empty = control only.

mkdir -p "$OUT_DIR"
cd "$REPO"

echo "=== negative control (no checkpoint required, ~11 s) ==="
"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --negative_control --out "$OUT_DIR/exp16a_negative_control.txt"
grep -E "NOISE MEETS BOTH|criterion 1:|w/floor =|CRITERION IS MET" \
     "$OUT_DIR/exp16a_negative_control.txt" || true

if [ -z "$CKPT" ]; then
  echo
  echo "CKPT is not set -> stopping here, which is the intended behaviour."
  echo "Arm A needs 630k-audioset-best.pt (1,863,587,645 bytes) from"
  echo "  https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
  echo "Downloading it is a deliberate decision. Then:  CKPT=/path/630k-audioset-best.pt $0"
  exit 0
fi

echo
echo "=== stage 1 - CLAP extraction in $PY_CLAP (~9 min measured) ==="
"$PY_CLAP" src/madeeg_exp16a_clap_extract.py --madeeg_dir "$MADEEG_DIR" \
           --out_dir "$CLAP_DIR" --ckpt "$CKPT"

echo
echo "=== stage 2 - measurement in $PY (the Exp. 9/13 canary first) ==="
"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --clap_dir "$CLAP_DIR" --out "$OUT_DIR/exp16a_clap_separability.txt"

echo
echo "=== REPLICATION OUTCOME ==="
grep -E "PASSED|FAILED|CLAP|w/floor|VERDICT" "$OUT_DIR/exp16a_clap_separability.txt" || true
echo
echo "A GO here is a HYPOTHESIS, not a result: Exp. 15 showed that this gate can be"
echo "   won by a representation that then makes the real task WORSE, and the negative"
echo "   control above shows that noise satisfies it."
echo "   Read the within/floor column BEFORE the gate column."
