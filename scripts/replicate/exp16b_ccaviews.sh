#!/usr/bin/env bash
# =============================================================================
# Exp. 16 braccio B — una feature extraction EEG piu' ricca aiuta? band_power.
#
# COSA MISURA: se aggiungere la vista `band_power` (potenza log per canale, un
#   blocco per banda) alla CCA multi-vista alza il tracciamento su own-vs-other.
#   E' la versione onesta della richiesta di A. ("la miglior feature extraction
#   dell'EEG"): `--eeg_repr spectra` vive nel modello contrastivo — la famiglia che
#   memorizza il brano — ed e' ESCLUSO dal contratto.
# COSTO      : ~7 min CPU (misurato: 4 run own-vs-other + 1 self-test + 4 McNemar).
# SGUARDI    : NO — own-vs-other gira sui SOLO, materiale di training gratuito.
#              Nessun duo deciso, nessun trio, nessuno split. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# ⚠️ MA NON E' GRATIS NEL TASSO DI ERRORE PER FAMIGLIA: con Exp. 7 (8 candidati),
#    Exp. 8 (2) ed Exp. 14 (2), questo e' il 13esimo confronto sulle STESSE 376
#    decisioni. Qualunque frase "abbiamo trovato qualcosa che batte 208/376" va
#    riportata con questo conteggio accanto.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   CANCELLO PRELIMINARE: il 209/376 pinnato viene da `--cca_views eeg_lagged` DA
#     SOLO (letto in runs/results/ovo_cca_ica/madeeg_ownvsother_summary.txt). Se
#     avesse gia' contenuto band_power, il contratto §B.2 chiudeva il braccio senza
#     girare niente.
#   CANDIDATO (K=1): CCA `eeg_lagged,band_power` -> 202/376 = 0.5372
#     NULL 0.500 ESATTO per simmetria (ogni coppia in entrambe le direzioni)
#     p binomiale esatto una coda 0.08185   BARRA >= 208/376 -> NON SUPERATA. 🔴
#   RIFERIMENTI ri-misurati oggi: ridge mel-8 208/376 (p 0.02208, = la barra) ·
#     CCA vista singola 209/376 (p 0.01717).
#   McNemar DESCRITTIVI, tutte e quattro le direzioni, nessuna significativa:
#     vs ridge mel-8   : cand batte ref 66/138 (p 0.7243) · ref batte cand 72/138 (p 0.3353)
#     vs CCA 1 vista   : cand batte ref 43/93  (p 0.7965) · ref batte cand 50/93  (p 0.2670)
#   LA CAUSA, ed e' la riga che vale piu' del verdetto: mean rho(own)/rho(other)
#     passa da 0.0666/0.0112 = 5.9 (vista singola) a 0.0629/0.0058 = 10.8
#     (multi-vista). La SEPARAZIONE MEDIA QUASI RADDOPPIA E L'ACCURATEZZA SCENDE.
#     La decisione e' un argmax fra due punteggi dello stesso stimatore: e'
#     invariante di scala, quindi un rho medio piu' alto NON e' un decoder migliore.
#
# ⚠️ COSA E' STATO TESTATO DAVVERO: `band_power` DENTRO la banda pre-registrata
#    1-8 Hz, cioe' SOLO delta (1-4) e theta (4-8) — blocks_x=[('eeg_lagged',340),
#    ('band_power',40)] = 20 canali x 2 bande. Alfa e beta sono ZERO per costruzione
#    dentro 1-8 Hz e il modulo le scarta dichiarandolo. Allargare --band_high e' un
#    cambiamento SEPARATO e dichiarato, e NON e' stato fatto qui.
#
# CANARINI   : tre, DENTRO il driver, attraversati PRIMA dei numeri veri; il driver
#              esce con errore se uno fallisce (Comandamenti §3).
#              G1 percorso di default: 208/376 + md5 2eaa926244de340d31907c6deeb04b0c
#              G2 (aggiunta) CCA vista singola ri-misurata: 209/376 e ZERO decisioni
#                 discordanti contro la run archiviata del 10/8 (md5 identico:
#                 dbbe6400764c94876a569aa4e6ff4c7b)
#              G3 controllo positivo del percorso CCA MULTI-VISTA, soglia >= 0.95:
#                 1.0000 (EEG interamente sintetico: non e' un risultato)
# CONTRATTO  : vault, "Cap. 2 — Exp. 16: la premessa della tesi messa alla prova —
#              CLAP come bersaglio e le feature EEG ricche, criterio pre-registrato
#              (12 ago 2026)" §B
# PROVENIENZA: docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt (pinnata) ·
#              runs/results/exp16b_*/
#
# ⚠️ Il driver NON accetta --log_dir: riscrive runs/results/exp16b_*. Il file di
#    provenienza PINNATO non viene toccato: --out va in $OUT_DIR.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
OUT="$OUT_DIR/exp16b_ccaviews_ownvsother.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" src/madeeg_exp16b_ccaviews.py --madeeg_dir "$MADEEG_DIR" --out "$OUT"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
grep -E "PASSED|FAILED|/376|MISSES|CLEARS|mean rho|VERDICT|B FAILS|B PASSES" "$OUT" || true
echo
echo "provenienza della replica : $OUT"
echo "provenienza pinnata (12/8): docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt"
