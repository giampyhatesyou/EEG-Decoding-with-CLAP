#!/usr/bin/env bash
# =============================================================================
# Exp. 17 — CLAP nel gate di separabilita, con il criterio RIFATTO invariante
#           alla dimensionalita. (SOLO AUDIO)
#
# COSA MISURA: la stessa domanda dell'Exp. 13 con UN candidato — CLAP-512, la
#   premessa della tesi — ma giudicato su un criterio nuovo, perche' quello
#   vecchio e' stato FALSIFICATO: il controllo negativo del 12/8 mostra che 512
#   dimensioni di RUMORE lo soddisfano (36/36 e gap -0.0002), e CLAP ha
#   esattamente 512 dimensioni. Non e' un paletto spostato: al momento in cui il
#   criterio nuovo e' stato scritto NON ESISTEVA nessun numero di CLAP.
# COSTO      : stadio 1 ~9' su baldo (CPU) + stadio 2 ~11" sul Mac.
# SGUARDI    : NO — zero. Apre l'HDF5 solo per `soli` e i metadati: mai
#              ['response'], mai un trial, mai un'etichetta. Nessun duo, nessun trio.
# GPU        : no. L'estrazione su baldo gira su CPU (il nodo di login non ha GPU).
#
# IL CRITERIO, tutte e tre le condizioni (contratto §2, scritto prima del codice):
#   1. |corr| within-duo sotto mel-8 in >= 26/36   (P = 0.005665) — invariato
#   2. within/pavimento <= 1.80  (>= 60% della distanza mel-8 3.131 -> rumore 0.922)
#   3. within/pavimento >= 1.20  — MARGINE OBBLIGATORIO SUL RUMORE: sotto questa
#      soglia il candidato e' dichiarato INDISTINGUIBILE DAL RUMORE e NON vince.
#
# I CANCELLI, nell'ordine, con hard-exit (contratto §3):
#   1. canarino Exp. 13: mel-8 0.1772/0.1442 · flux 0.2874/0.2417 · 30/36 (+-0.0010)
#   2. controllo negativo RIGIRATO a 512 dim NELLA STESSA RUN: within/pavimento
#      dentro [0.85, 1.05], altrimenti il cablaggio e' cambiato -> lo script esce
#      con errore PRIMA che la riga CLAP venga calcolata
#   3. il manifest.json dell'estrazione (checkpoint + sha256, versioni, sr,
#      finestra, hop, macchina): senza, la misura non parte
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   checkpoint 630k-audioset-best.pt  1 863 587 645 byte
#     sha256 8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037
#   CANARY   mel-8 mean 0.1772 median 0.1442 · flux 0.2874/0.2417 · 30/36
#   N1 rumore-512  within/pavimento 0.922
#   C7 CLAP-512    vedi docs/provenance/2026-08-12_exp17_clap_separability.txt
#
# ⚠️ IL LIMITE CHE VA SCRITTO IN OGNI RAMO DELL'ESITO: l'apertura di progetto di
#   CLAP e' 10 s, 160x la finestra usata qui, e sotto i 10 s `laion_clap` applica
#   `repeatpad`. STIAMO MISURANDO CLAP FUORI DAL REGIME PER CUI E' ADDESTRATO.
#   Un fallimento qui NON e' evidenza contro CLAP a 10 s.
#
# CONTRATTO  : vault, "Cap. 2 — Exp. 17: CLAP nel gate di separabilita, criterio
#              RIFATTO invariante alla dimensionalita (12 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-12_exp17_clap_separability.txt ·
#              log integrale dell'estrazione:
#              docs/provenance/2026-08-12_exp17_baldo_extraction.log
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # la provenienza pinnata non si tocca (§9)
CLAP_DIR=${CLAP_DIR:-$REPO/runs/clap_exp17}   # gli .npy + manifest.json dello stadio 1
OUT="$OUT_DIR/exp17_clap_separability.txt"

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ ! -f "$CLAP_DIR/manifest.json" ]; then
  cat <<EOF
manifest.json non trovato in $CLAP_DIR.

Lo STADIO 1 (estrazione) gira su baldo, non qui: laion_clap non e' installato in
/opt/miniconda3 e in /opt/miniconda3 non si installa niente (Legacy §4 trappola 1).
Il comando, cosi' com'e' stato eseguito il 12/8 (CPU, nessuna GPU, nessun SLURM):

  ssh baldo
  curl -L -o ~/clap_ckpt/630k-audioset-best.pt \\
       https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
  sha256sum ~/clap_ckpt/630k-audioset-best.pt   # 8053c977...c3f037
  ~/.conda/envs/eeg_attention/bin/python \\
      ~/EEG-Attention-decoding-with-CLAP/src/madeeg_exp16a_clap_extract.py \\
      --madeeg_dir ~/madeeg --out_dir ~/clap_exp17 \\
      --ckpt ~/clap_ckpt/630k-audioset-best.pt

poi si riportano gli .npy e il manifest.json sul Mac:

  scp -r baldo:~/clap_exp17/ $CLAP_DIR

Il log integrale di quella esecuzione e' in
docs/provenance/2026-08-12_exp17_baldo_extraction.log.
EOF
  exit 1
fi

"$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
      --clap_dir "$CLAP_DIR" --exp17 --out "$OUT"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
grep -E "CANARY|PASSED|FAILED|GATE 2|noise-512|CLAP-512|VERDICT" "$OUT" || true
echo
echo "provenienza della replica : $OUT"
echo "provenienza pinnata (12/8): docs/provenance/2026-08-12_exp17_clap_separability.txt"
echo
echo "⚠️ Qualunque sia l'esito: CLAP e' stato misurato a 62.5 ms, 160x sotto la sua"
echo "   apertura di progetto (10 s), dove laion_clap applica repeatpad. Un fallimento"
echo "   NON e' evidenza contro CLAP a 10 s."
