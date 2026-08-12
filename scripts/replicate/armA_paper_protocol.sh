#!/usr/bin/env bash
# =============================================================================
# Braccio A — parita' di protocollo col paper (Cantisani et al., WASPAA 2019).
#
# COSA MISURA: quanto si avvicina al 79 di F1 sui duetti il pacchetto COMPLETO
#   dichiarato dal paper — decoder per-soggetto PER-STRUMENTO, shrinkage 0.1,
#   24 bande mel a 256 Hz, lag 0-250 ms, training sui solo, decisione sul trial
#   intero (~24 s), metrica F1 — girato su tre bande di analisi.
# COSTO      : DA CONFERMARE. E' la famiglia piu' pesante del repo (3 run a
#              --target_fs 256 con --filters per_instrument, ~6 GB di matrice di
#              disegno l'una); i timestamp del 11/8 sono identici e non danno la
#              durata. Prevedi decine di minuti e 16 GB di RAM (Legacy §4.7).
# SGUARDI    : SI' — decide sui DUO stereo. Tutti gia' spesi: ESPLORATIVO PER
#              COSTRUZIONE, dichiarato tale nel summary. 🔒 TRIO non toccati.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   banda 1-8 Hz   (armA_paper_package)  F1 micro = accuratezza 0.5267   n = 150
#   banda 1-40 Hz  (armA_paper_band40)   F1 micro 0.4800
#   banda 0.2-40Hz (armA_paper_band02_40) F1 micro 0.5400
#   null 0.500 sul duo · r(attended) 0.016-0.025 contro la mediana 0.119 del paper
#   ESITO: il 79 NON SI RIPRODUCE da nessun pacchetto fedele ai metodi dichiarati.
#   n = 150 e non 154: 4 trial cadono sotto --filters per_instrument perche' il
#   soggetto 0007 non ha i solo Bo/Fh — noto e dichiarato, non un bug.
#
# CANARINO   : il contratto del braccio A non ne dichiara uno (nessun gate,
#              nessuna affermazione pre-registrata: e' una sonda di protocollo).
#              Il controllo e' la riproduzione stessa dei tre F1 qui sopra.
#              NON aggiungere un --self_test per "avere un controllo": non ha un
#              valore atteso pinnato e produrrebbe un numero nuovo (Comandamenti §10).
# CONTRATTO  : vault, "Piano — la svolta: dal negativo onesto a un modello che
#              decodifica (11 ago 2026)" §4A · report "2026-08-11 (6)"
# PROVENIENZA: runs/results/armA_paper_{package,band40,band02_40}/
#
# ⚠️ SPLIT SAME/DIFF-MELODY (docs/provenance/2026-08-11_armA_same_diff_melody.txt,
#    same 46/89 vs diff 33/61 a 1-8 Hz; same 51/89 vs diff 30/61 a 0.2-40 Hz):
#    NON e' riproducibile da qui.
# DA CONFERMARE: nessun file del repo produce quello split — e' stato calcolato
#    ad hoc dai madeeg_records.csv + madeeg_sequences_raw.yaml e lo script non e'
#    stato salvato (verificato con grep su src/, scripts/, sweeps/ il 12/8).
#    Va riscritto, non indovinato.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono

cd "$REPO"

# Il pacchetto del paper. --eeg_clean none e' imposto dalla release preprocessed
# usata per i solo; --test_eeg raw e --spatial stereo sono il protocollo del paper.
PKG=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --spatial stereo
     --ensemble duo --target mel --n_mels 24 --target_fs 256 --lags_ms 250
     --estimator shrinkage --shrinkage_lambda 0.1 --filters per_instrument
     --eeg_clean none --seed 42 --cv_folds 5)

run() {  # run <nome> <band_low> <band_high>
  echo; echo "--- $1 : banda $2-$3 Hz ---"
  "$PY" src/madeeg_reconstruction.py "${PKG[@]}" --band_low "$2" --band_high "$3" \
        --training_date "${TAG}$1"
}

run armA_paper_package  1   8
run armA_paper_band40   1   40
run armA_paper_band02_40 0.2 40

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in armA_paper_package armA_paper_band40 armA_paper_band02_40; do
  printf "%-22s " "$R"
  grep -h "^F1 over the attended" "$REPO/runs/results/${TAG}$R/madeeg_summary.txt"
done
echo
echo "record pinnati (11/8): runs/results/armA_paper_*"
echo "il paper (WASPAA 2019 Tab. 1, duetti): AE 58 · MAG 74 · MEL 79"
