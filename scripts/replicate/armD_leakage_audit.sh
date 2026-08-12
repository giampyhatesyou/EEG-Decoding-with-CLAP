#!/usr/bin/env bash
# =============================================================================
# Braccio D — audit di leakage per-finestra su MAD-EEG (il 92.6% di Niu 2024).
#
# COSA MISURA: quanta accuratezza a finestre di 1 s viene da IMPRONTA DI TRIAL e
#   non da attenzione. Una LDA su log-varianza in 5 bande decodifica etichette
#   PSEUDO-CASUALI per trial: sotto CV per-finestra (finestre dello stesso trial
#   a cavallo dello split) contro CV per-trial. E' un'affermazione sul PROTOCOLLO,
#   non sull'EEG.
# COSTO      : ~2-3 min CPU (stima; una passata su 8 soggetti, LDA su finestre da 1 s).
#              DA CONFERMARE alla prima run: non esiste un tempo misurato agli atti.
# SGUARDI    : SI' — legge i trial duo stereo (EEG + etichette vere). Duo tutti
#              gia' spesi: ESPLORATIVO PER COSTRUZIONE. 🔒 TRIO non toccati.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PSEUDO-etichette (bilanciate): window-CV 0.6209  vs  trial-CV 0.4847
#                                  null 0.5136 PER COSTRUZIONE, stampato accanto
#   etichette VERE:                window-CV 0.3803  vs  trial-CV 0.0895
#                                  null 0.2277 (classe maggioritaria media)
#   ESITO: [LEAKAGE DEMONSTRATED] — regola dichiarata NEL CODICE prima della run:
#   leakage se pseudo window-CV > 0.60 E pseudo trial-CV in [0.40, 0.60].
#
# CANARINO   : il braccio PSEUDO-ETICHETTE E' il controllo positivo dell'audit
#              (etichette a contenuto attentivo NULLO: qualunque accuratezza sopra
#              il suo null e' impronta pura). Il null e' STAMPATO accanto: e' cosi'
#              che la v1 mal calibrata (pseudo sbilanciate, null 0.64) e' stata
#              smascherata — conservata agli atti in
#              runs/results/armD_leakage_audit/*_v1_unbalanced.* (Legacy §4.5).
#              ⚠️ Se il null stampato non e' ~0.5136, NON leggere il numero.
# CONTRATTO  : vault, "Piano — la svolta ... (11 ago 2026)" §4D · report "2026-08-11 (7)"
# PROVENIENZA: runs/results/armD_leakage_audit/leakage_audit_summary.txt +
#              leakage_audit_per_subject.csv
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # il record pinnato non si sovrascrive
SEED=${SEED:-42}

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_leakage_audit.py --madeeg_dir "$MADEEG_DIR" \
      --log_dir "$OUT_DIR" --seed "$SEED"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
cat "$OUT_DIR/armD_leakage_audit/leakage_audit_summary.txt"
echo
echo "uscite della replica : $OUT_DIR/armD_leakage_audit/"
echo "record pinnato (11/8): runs/results/armD_leakage_audit/"
