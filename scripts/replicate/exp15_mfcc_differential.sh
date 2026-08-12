#!/usr/bin/env bash
# =============================================================================
# Exp. 15 (ESPLORATIVO) — il differenziale attenzionale sotto MFCC-13.
#
# COSA MISURA: se la rappresentazione che separa meglio gli stem (Exp. 13) e
#   traccia quanto mel (Exp. 14) alza il DIFFERENZIALE ATTENZIONALE standardizzato
#   D = mean(d)/sd(d) sulla decisione duo, con permutazione appaiata a inversione
#   di segno. Il primario NON e' l'accuratezza: il check "questo test si puo'
#   vincere?" aveva predetto +1.4 trial contro una barra che ne chiede +4.
# COSTO      : ~2-3 min CPU (misurato dai timestamp del 12/8: prima run 13:23:20 ->
#              ultima 13:25:05 = 1m45s, piu' la prima run non temporizzata).
# SGUARDI    : SI' — decide sui DUO. Tutti gia' spesi: ESPLORATIVO PER COSTRUZIONE,
#              scritto nel titolo e nel file. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO (P2, k-fold duo):  D(mel-8) = +0.1001   D(MFCC-13) = -0.1119
#     differenza -0.2120   NULL = 0 ESATTO per costruzione   p = 0.9833
#     (permutazione a inversione di segno, B = 10000, seme 20260812)
#     -> NON rigetta il null; e la stima punta dalla parte OPPOSTA all'ipotesi.
#   PAVIMENTO DI RILEVABILITA' (obbligatorio): MDD = 0.2467 in D a n = 154,
#     potenza 0.80, alpha 0.05 una coda; sd delle differenze appaiate 1.2312.
#   SECONDARIE descrittive: accuratezza P2 MFCC 65/154 contro l'ancora mel 86/154
#     (la predizione del contratto era 87.4/154, la barra Exp. 9 era 90/154);
#     P1 MFCC 75/154 contro l'ancora 74/154; McNemar "mel batte MFCC" 45/69 p = 0.0077.
#   ESITO: 🟡🔴 non conclusivo per potenza (MDD 0.2467 > effetto predetto), ma la
#   stima e' di segno opposto. Il modello predittivo del §0 e' stato RITIRATO lo
#   stesso giorno (rho aggregato usato come se fosse per-dimensione).
#
# CANARINI   : due, DENTRO il driver, attraversati PRIMA dei numeri veri; il driver
#              esce con errore se uno fallisce (Comandamenti §3).
#              G1 ancore mel decision-identiche all'Exp. 9: 74/154 e 86/154 con
#                 ZERO decisioni discordanti, confrontate TRIAL PER TRIAL.
#              G2 controllo positivo del percorso MFCC sulla decisione duo,
#                 soglia >= 0.95: 1.0000.
# CONTRATTO  : vault, "Cap. 2 — Exp. 15 (ESPLORATIVO): il differenziale attenzionale
#              sotto MFCC, criterio pre-registrato (12 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-12_exp15_mfcc_differential.txt (pinnata) ·
#              runs/results/exp15_*/
#
# ⚠️ Il driver NON accetta --log_dir: riscrive runs/results/exp15_*. Il file di
#    provenienza PINNATO non viene toccato: --out va in $OUT_DIR.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp15_mfcc_differential.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp15_differential.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
grep -E "PASSED|FAILED|D\(|observed statistic|MDD|p = |VERDICT" "$OUT" || true
echo
echo "provenienza della replica : $OUT"
echo "provenienza pinnata (12/8): docs/provenance/2026-08-12_exp15_mfcc_differential.txt"
