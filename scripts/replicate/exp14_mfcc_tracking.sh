#!/usr/bin/env bash
# =============================================================================
# Exp. 14 — il lato TRACKING del gate: MFCC-13 e mel-64 su own-vs-other.
#
# COSA MISURA: se le due rappresentazioni promosse dal gate solo-audio dell'Exp. 13
#   sono anche TRACCIATE dall'EEG, cioe' se scartare l'energia (MFCC esclude il
#   coefficiente 0) costa il tracciamento. Barra pre-registrata: >= 208/376.
# COSTO      : ~3-4 min CPU (misurato dai timestamp del 12/8: cancello 12:49:33 ->
#              ultima run 12:52:55 = 3m22s per 5 run).
# SGUARDI    : NO — own-vs-other gira sui SOLO, materiale di training dichiarato
#              gratuito dal registro degli sguardi. Nessun duo, nessun trio.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   T1 MFCC-13  214/376 = 0.5691  p esatto 0.004224  -> TRACCIA
#   T2 mel-64   212/376 = 0.5638  p esatto 0.007623  -> TRACCIA
#   mel-8 (rif) 208/376 = 0.5532  (= la barra stessa)   null 0.500 esatto
#   ESITO: ramo 1 del contratto — entrambe tracciano; il front end del braccio C
#   e' nominato. Ma il McNemar descrittivo NON le distingue da mel-8.
#   band_pearson(own): MFCC 0.0165 · mel-64 0.0341 · mel-8 0.0360 (la contrazione
#   di scala che l'Exp. 13 aveva lasciato aperta).
#
# CANARINI   : tre, tutti DENTRO il driver e attraversati PRIMA dei numeri veri;
#              il driver esce con errore se uno fallisce (Comandamenti §3).
#              G1 regressione: --target mel --n_mels 8 -> 208/376 con md5
#                 2eaa926244de340d31907c6deeb04b0c (identico a ovo_ridge_ica).
#              G2 identita' di rappresentazione con il candidato C3 dell'Exp. 13:
#                 differenza massima 0.000e+00.
#              G3 controllo positivo sintetico per candidato, soglia >= 0.95:
#                 1.0000 entrambi.
# CONTRATTO  : vault, "Cap. 2 — Exp. 14: il lato tracking del gate — MFCC e mel-64
#              su own-vs-other, criterio pre-registrato (12 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-12_exp14_mfcc_tracking.txt (pinnata) ·
#              runs/results/exp14_*/
#
# ⚠️ Il driver NON accetta --log_dir: riscrive runs/results/exp14_* (sono i record
#    di questo stesso esperimento, e i cancelli si fermano prima se qualcosa non
#    torna). Il file di provenienza PINNATO invece non viene toccato: --out va in
#    $OUT_DIR. Se ti serve intatto anche il record delle run, copia le dir prima.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp14_mfcc_tracking.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp14_tracking.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
grep -E "PASSED|FAILED|TRACKS|does not|BRANCH|VERDICT" "$OUT" || true
echo
echo "provenienza della replica : $OUT"
echo "provenienza pinnata (12/8): docs/provenance/2026-08-12_exp14_mfcc_tracking.txt"
