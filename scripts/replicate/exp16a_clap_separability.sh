#!/usr/bin/env bash
# =============================================================================
# Exp. 16 braccio A — CLAP separa i due stem? (SOLO AUDIO)
#
# 🔴 STATO AL 12/8: **LO STADIO 1 NON E' STATO ESEGUITO**, e la ragione non e' il
#    budget. `laion_clap` E' installato in /opt/anaconda3, ma il suo checkpoint di
#    default NON e' in cache su questa macchina: `CLAP_Module.load_ckpt()` SCARICA
#    630k-audioset-best.pt, **1 863 587 645 byte (1.74 GiB)**, da
#    https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
#    (verificato con una HEAD: content-length 1863587645). Scaricare 1.74 GiB e' una
#    decisione di A., non di un agente — come lo era il download SparrKULee. Lo
#    stadio 1 RIFIUTA di innescare il download da solo: vuole --ckpt su un file
#    gia' esistente.
#
# COSA MISURERA': la stessa domanda dell'Exp. 13 con un settimo candidato — CLAP,
#   che e' la premessa della tesi e che il Registro §2.5 registra come MAI provato
#   come bersaglio. Criterio IDENTICO all'Exp. 13, non ritoccato: >= 26/36 sotto
#   mel-8 E gap <= 0.05 sul pavimento cross-brano della STESSA rappresentazione.
# SGUARDI    : NO — solo audio, nessun EEG aperto. 🔒 Nessun duo, nessun trio.
# GPU        : no.
#
# DUE STADI, OBBLIGATORI (trappola n.1 Legacy §4: Anaconda produce float diversi a
# 2e-5 e fa fallire i canarini md5; in /opt/miniconda3 non si installa niente):
#   stadio 1  ESTRAZIONE in /opt/anaconda3 -> <out_dir>/<md5-dello-stem>.npy + manifest.json
#   stadio 2  MISURA     in /opt/miniconda3 -> il gate dell'Exp. 13, che legge i .npy
#             come una rappresentazione qualsiasi, e il cui CANARINO (mel-8
#             0.1772/0.1442 · flux 0.2874/0.2417 · 30/36) e' la prova che l'ambiente
#             della MISURA non e' cambiato.
#
# LA RICETTA, FISSATA PRIMA CHE ESISTA QUALUNQUE NUMERO (contratto §A.3):
#   hop 62.5 ms -> serie a 16 Hz -> Nyquist 8 Hz, esattamente il tetto della banda
#     pre-registrata. COSTO MISURATO su questa macchina (HTSAT-tiny, batch 32, pesi
#     casuali — solo cronometro, nessun numero scientifico): ~62 ms di CPU per
#     finestra x 8876 finestre sui 555 s di audio degli stem = **~9 minuti**, dentro
#     il budget di 90. ⟹ **il ripiego dichiarato (hop 125 ms, banda 1-4 Hz, mel-8
#     rigirata nella stessa banda) NON serve e NON e' stato usato.**
#   finestra 62.5 ms = hop, consecutive e non sovrapposte. Tre ragioni indipendenti,
#     tutte scritte prima: (a) sotto i 10 s laion_clap fa `repeatpad`, cioe' RIPETE
#     la forma d'onda floor(480000/len) volte — e 3000 campioni dividono 480000
#     esattamente (160 ripetizioni), quindi ZERO zeri aggiunti; (b) un'apertura lunga
#     W attenua come |sinc(fW)| e a 125 ms il primo zero cade ESATTAMENTE su 8 Hz,
#     annichilendo il tetto della banda; a 62.5 ms il primo zero e' a 16 Hz; (c) con
#     finestra = hop i frame sono disgiunti e nessuna correlazione fra frame vicini
#     e' fabbricata dalla sovrapposizione.
#   ⚠️ E IL LIMITE CHE VIENE CON ESSA, dichiarato ora e non dopo il numero: CLAP e'
#     invariante al tempo PER PROGETTO e la sua apertura e' 10 s. 62.5 ms e' 160x
#     piu' corta. Non e' un difetto dell'implementazione: e' la banda pre-registrata
#     e la scala temporale di CLAP che distano 160x. Qualunque cosa esca, QUELLO fa
#     parte del risultato.
#
# 🔑 IL NUMERO CHE ESISTE GIA', E CHE VA LETTO PRIMA DI SCARICARE QUALSIASI COSA —
#    IL CONTROLLO NEGATIVO (gira in 11 s, non serve nessun checkpoint):
#      512 dimensioni di RUMORE BIANCO a 16 Hz, una estrazione per stem, seme
#      20260812, nessuna struttura condivisa per costruzione:
#        criterio 1  36/36  (soglia 26/36)  -> SODDISFATTO
#        criterio 2  gap -0.0002 (soglia <= 0.05) -> SODDISFATTO
#        w/pavimento 0.922   (mel-8 3.131 · MFCC-13 2.084)
#      ⟹ **IL CRITERIO PRE-REGISTRATO E' SODDISFATTO DAL RUMORE** a questa
#      dimensionalita'. Non e' sufficiente, da solo, a distinguere "questa
#      rappresentazione separa i due stem" da "questa rappresentazione ha molte
#      dimensioni indipendenti e una correlazione aggregata piccola". Il criterio
#      NON e' stato cambiato (Comandamenti §5/§6): e' stato misurato, e la colonna
#      `w/pavimento` e' il numero che una riga CLAP dovra' battere prima che il suo
#      PASS voglia dire qualcosa.
#
# CONTRATTO  : vault, "Cap. 2 — Exp. 16 ... criterio pre-registrato (12 ago 2026)" §A
# PROVENIENZA: docs/provenance/2026-08-12_exp16A_negative_control.txt (esiste) ·
#              docs/provenance/2026-08-12_exp16A_clap_separability.txt (stato, NON
#              ancora una misura — lo diventa quando lo stadio 1 gira)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: la MISURA sta qui
PY_CLAP=${PY_CLAP:-/opt/anaconda3/bin/python} # l'ESTRAZIONE, e solo quella, sta qui
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
CLAP_DIR=${CLAP_DIR:-$REPO/runs/clap_exp16a}
CKPT=${CKPT:-}                                # ← il file .pt locale. Vuoto = solo il controllo.

mkdir -p "$OUT_DIR"
cd "$REPO"

echo "=== controllo negativo (nessun checkpoint richiesto, ~11 s) ==="
"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --negative_control --out "$OUT_DIR/exp16a_negative_control.txt"
grep -E "NOISE MEETS BOTH|criterion 1:|w/floor =|CRITERION IS MET" \
     "$OUT_DIR/exp16a_negative_control.txt" || true

if [ -z "$CKPT" ]; then
  echo
  echo "CKPT non impostata -> mi fermo qui, e va bene cosi'."
  echo "Il braccio A vuole 630k-audioset-best.pt (1 863 587 645 byte) da"
  echo "  https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
  echo "Scaricarlo e' una decisione di A. Poi:  CKPT=/percorso/630k-audioset-best.pt $0"
  exit 0
fi

echo
echo "=== stadio 1 — estrazione CLAP in $PY_CLAP (~9 min misurati) ==="
"$PY_CLAP" src/madeeg_exp16a_clap_extract.py --madeeg_dir "$MADEEG_DIR" \
           --out_dir "$CLAP_DIR" --ckpt "$CKPT"

echo
echo "=== stadio 2 — misura in $PY (il canarino Exp. 9/13 per primo) ==="
"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --clap_dir "$CLAP_DIR" --out "$OUT_DIR/exp16a_clap_separability.txt"

echo
echo "=== ESITO DELLA REPLICA ==="
grep -E "PASSED|FAILED|CLAP|w/floor|VERDICT" "$OUT_DIR/exp16a_clap_separability.txt" || true
echo
echo "⚠️ Un GO qui e' un'IPOTESI, non un risultato: l'Exp. 15 ha mostrato che questo"
echo "   gate puo' essere vinto da una rappresentazione che poi PEGGIORA sul compito"
echo "   vero, e il controllo negativo qui sopra mostra che il rumore lo soddisfa."
echo "   Leggi la colonna w/pavimento PRIMA della colonna gate."
