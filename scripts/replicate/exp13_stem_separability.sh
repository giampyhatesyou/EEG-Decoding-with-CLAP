#!/usr/bin/env bash
# =============================================================================
# Exp. 13 — gate di separabilita SOLO AUDIO per il braccio C.
#
# COSA MISURA: sotto quale rappresentazione (6 candidati) i due stem in
#   competizione di un duo diventano scorrelati quanto stem di brani diversi.
# COSTO      : ~10 s CPU (misurato: 9.81 s reali, Mac arm64, 12 ago 2026).
# SGUARDI    : NO — zero. Apre l'HDF5 solo per `soli` e i metadati: mai
#              ['response'], mai un trial, mai un'etichetta.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   CANARY  mel-8  mean 0.1772  median 0.1442
#   CANARY  flux   mean 0.2874  median 0.2417   flux > mel su 30/36
#   C3 MFCC-13  within 0.0533 · pavimento cross-brano 0.0256 · 32/36 · gap +0.0277
#   VERDETTO: GO, unico candidato che passa entrambi i criteri = C3 MFCC-13.
#   Il canarino e' attraversato PRIMA di tutto DAL DRIVER STESSO, che esce con
#   errore se fallisce: nessun candidato viene calcolato se non passa.
#
# CONTRATTO  : vault, "Cap. 2 — Exp. 13: gate di separabilita solo-audio per il
#              braccio C, criterio pre-registrato (12 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-12_exp13_stem_separability.txt (pinnata)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # la provenienza pinnata non si tocca (Comandamenti §9)
OUT="$OUT_DIR/exp13_stem_separability.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
grep -E "CANARY|PASSED|FAILED|C3 MFCC-13|VERDICT" "$OUT" || true
echo
echo "provenienza della replica : $OUT"
echo "provenienza pinnata (12/8): docs/provenance/2026-08-12_exp13_stem_separability.txt"
