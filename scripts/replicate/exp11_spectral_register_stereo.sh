#!/usr/bin/env bash
# =============================================================================
# Exp. 11 (ESPLORATIVO) — il registro spettrale attenzionato, duo STEREO.
#
# COSA MISURA: se il pattern spaziale-spettrale di potenza EEG (Morlet 1-40 Hz,
#   25 frequenze, LDA shrinkage, LOSO) dice QUALE DEI DUE REGISTRI e' attenzionato,
#   confrontando i due trial della STESSA mixture (47 coppie appaiate).
#   E' la proposta del Relatore (de Vries et al. 2021) eseguita per intero.
# COSTO      : ~3 min CPU (stima dai timestamp del 11/8: primo controllo 15:59:38,
#              primario 16:00:25; lo sweep opzionale aggiunge ~2 min).
# SGUARDI    : SI' — decide sui DUO stereo. I 309 duo sono TUTTI GIA' SPESI:
#              rigirarli non apre materiale nuovo, ma il numero resta ESPLORATIVO
#              PER COSTRUZIONE. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO   registro attenzionato 24/47 = 0.5106  (p = 0.5000)
#              soglia scritta prima 30/47 = 0.6383 -> [NON SUPERATA]
#   null 0.500 ESATTO per simmetria (i due trial della coppia sono scambiabili)
#   controllo positivo a=1.0  47/47 · dose-risposta a=0.5  47/47
#   controllo negativo (etichette permutate, 20 semi) 0.4968, banda [0.40,0.60] OK
#   dose-risposta: lontane 11/22 = 0.5000 · vicine 13/25 = 0.5200  (piatta)
#   sonda confound intensita' (n=13, descrittiva): 3/13 = 0.2308
#   CONTROL_SWEEP=1 in piu': a=0.40 47/47 · 0.20 47/47 · 0.10 38/47 ·
#              0.05 26/47 · 0.02 23/47 · 0.01 22/47 -> pavimento a = 0.05
#
# CANARINO   : il controllo positivo (modulazione 8-13 Hz legata all'etichetta
#              iniettata nell'EEG REALE, soglia >= 0.90) e' attraversato PER PRIMO
#              DAL DRIVER STESSO, che esce con errore se non passa: il numero vero
#              non viene calcolato. Assert anche su n_pairs == 47.
# CONTRATTO  : vault, "Cap. 2 — Exp. 11 (ESPLORATIVO): il registro spettrale
#              attenzionato, criterio pre-registrato (11 ago 2026)"
# PROVENIENZA: runs/results/exp11_{primary,control_a10,control_a05,control_sweep}/
#
# ⚠️ src/madeeg_spectral_attention.py NON E' TRACCIATO DA GIT (stato al 12/8/2026):
#    in un clone pulito il file non c'e' e questo script non gira. DA CONFERMARE
#    con A. se va committato; finche' non lo e', la replica dipende dal working tree.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # i record pinnati non si sovrascrivono
CONTROL_SWEEP=${CONTROL_SWEEP:-0}             # 1 = solo lo sweep di ampiezza (pavimento)

SRC="$REPO/src/madeeg_spectral_attention.py"
[ -f "$SRC" ] || { echo "manca $SRC (file untracked: vedi l'avvertenza in testa)"; exit 1; }

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ "$CONTROL_SWEEP" = 1 ]; then
  "$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
        --log_dir "$OUT_DIR" --seed 42 --control_sweep
  echo; echo "=== ESITO (solo controlli, NON un risultato) ==="
  cat "$OUT_DIR/exp11_control_sweep/summary.txt"
  exit 0
fi

"$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed 42

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
cat "$OUT_DIR/exp11_primary/summary.txt"
echo
echo "uscite della replica : $OUT_DIR/exp11_*"
echo "record pinnati (11/8): runs/results/exp11_*"
echo "pavimento di sensibilita' (sweep di ampiezza): CONTROL_SWEEP=1 bash $0"
