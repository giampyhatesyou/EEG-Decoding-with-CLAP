#!/usr/bin/env bash
# =============================================================================
# Exp. 12 — il registro spettrale sui duo MONO (materiale fresco per la variabile).
#
# COSA MISURA: la stessa ipotesi dell'Exp. 11 (quale registro e' attenzionato)
#   sui 42 duo MONO, che esistono SOLO nella release raw e vanno ricostruiti da
#   li'; piu' lo stereo-da-raw come verifica di comparabilita' di pipeline e il
#   pooled a 89 coppie.
# COSTO      : ~3 min CPU (stima dai timestamp del 11/8: controllo 16:24:42,
#              primario 16:26:28). Legge la release RAW: piu' I/O dell'Exp. 11.
# SGUARDI    : SI' — decide sui DUO mono. I 309 duo sono TUTTI GIA' SPESI:
#              esplorativo per costruzione. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO   mono 22/42 = 0.5238  (p = 0.4388)
#              soglia pre-scritta il 9/8: 27/42 = 0.6429 -> [NON SUPERATA]
#   null 0.500 ESATTO per simmetria
#   SECONDARIE stereo-da-raw 24/47 = 0.5106 (soglia 30/47) — coincide con
#              l'Exp. 11 sulla release preprocessed: la pipeline e' irrilevante
#              pooled 89: 46/89 = 0.5169 (soglia 53/89)
#              dose-risposta: lontane 10/20 = 0.5000 · vicine 12/22 = 0.5455
#              concentrazione 5/5 (n=16, descrittiva): 10/16 = 0.6250
#   controllo positivo a=1.0  42/42 · negativo (20 semi) 0.5226, banda OK
#   pavimento di sensibilita': a=0.20 38/42 · a=0.10 28/42 · a=0.05 24/42
#
# CANARINO   : (a) canarino di ALLINEAMENTO del percorso raw, gia' attraversato e
#              scritto nel contratto §2 (picco netto a offset 0, distrutto da
#              +-0.12 s); (b) il controllo positivo con soglia >= 0.90 e' eseguito
#              PER PRIMO DAL DRIVER, che esce con errore se non passa.
# CONTRATTO  : vault, "Cap. 2 — Exp. 12: il registro spettrale sui duo MONO
#              (materiale fresco), criterio pre-registrato (11 ago 2026)"
# PROVENIENZA: runs/results/exp12_{primary,control_a10}/
#
# ⚠️ src/madeeg_spectral_attention.py NON E' TRACCIATO DA GIT (stato al 12/8/2026):
#    in un clone pulito il file non c'e' e questo script non gira. DA CONFERMARE
#    con A. se va committato.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # serve anche la release RAW nella stessa dir
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # i record pinnati non si sovrascrivono

SRC="$REPO/src/madeeg_spectral_attention.py"
[ -f "$SRC" ] || { echo "manca $SRC (file untracked: vedi l'avvertenza in testa)"; exit 1; }

mkdir -p "$OUT_DIR"
cd "$REPO"

# --source raw = Exp. 12. Il default (preprocessed) e' l'Exp. 11 e non si tocca.
"$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed 42 --source raw

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
cat "$OUT_DIR/exp12_primary/summary.txt"
echo
echo "uscite della replica : $OUT_DIR/exp12_*"
echo "record pinnati (11/8): runs/results/exp12_*"
