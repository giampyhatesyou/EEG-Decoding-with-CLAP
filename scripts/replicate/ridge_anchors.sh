#!/usr/bin/env bash
# =============================================================================
# L'ANCORA LINEARE (ridge) — le tre varianti che stanno in RESULTS.md §madeeg_duo.
#
# COSA MISURA: se la famiglia lineare sicura (ricostruzione dello stimolo + argmax
#   della correlazione fra le sorgenti presenti) decide QUALE dei due strumenti
#   di un duo era attenzionato. Tre varianti, che differiscono per UNA cosa
#   ciascuna: come si allena (k-fold sui duo vs solo del protocollo del paper) e
#   che rappresentazione si ricostruisce (log-mel vs inviluppo di Hilbert).
# DATA       : archivio mag-lug 2026 · le due run pinnate sono ri-girate 30/7-08/2026
#              con l'intestazione completa · `raw_solos` 3 lug 2026.
# DOMANDA    : «la famiglia lineare decide l'attenzione sul duo?»
# NULL       : 0.500 — su un duo il caso e' 1/n_present, e ogni mixture compare con
#              ENTRAMBI i suoi strumenti come target (18/18), quindi il prior dello
#              stimolo non e' informazione legittima.
# SOGLIA PRE-REGISTRATA: 88/154 = 0.5714 (minimo intero con binomiale esatta a una
#              coda < 0.05 su n=154). E' la soglia del contratto del 26/7.
# VERDETTO   : 🔴 nessuna delle tre la supera.
#              A1 duos_kfold mel  86/154 = 0.5584  p = 0.0853   <- il MASSIMO STORICO
#                 del progetto sull'attenzione, e NON e' significativo.
#              A2 duos_kfold env  78/154 = 0.5065
#              A3 raw_solos  mel  74/154 = 0.4805  <- sotto il caso
# COSTO      : ~5-10 min CPU per le tre (stima; a 64 Hz sono run leggere).
#              DA CONFERMARE alla prima esecuzione: non c'e' un tempo misurato agli atti.
# SGUARDI    : SI' — decidono sui DUO. I 309 duo sono TUTTI GIA' SPESI dal 30/7:
#              rigirarli non apre materiale nuovo, ma ogni numero sui duo dopo quella
#              data e' ESPLORATIVO PER COSTRUZIONE. 🔒 I TRIO NON SI TOCCANO.
#              ⚠️ A1 e A3 sono 1º SGUARDO PRE-REGOLA: sono stati guardati prima che
#              la convenzione statistica del progetto esistesse. Non si possono
#              ri-etichettare confermativi a posteriori.
# GPU        : no — madeeg_reconstruction.py e' numpy/scipy/sklearn/h5py, nessun torch.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   A1  train_on=duos_kfold  target=mel n_bands=8   -> 0.5584 · F1 micro/macro/weighted
#       0.5584/0.5448/0.5548 · r(att) 0.0202 vs r(best unatt) 0.0118 · inner_val_r 0.0245
#   A2  train_on=duos_kfold  target=envelope n_bands=1 -> 0.5065 · r 0.0139/0.0133 ·
#       inner_val_r 0.0164
#   A3  train_on=raw_solos --test_eeg raw  target=mel  -> 0.4805 · F1 0.4805/0.4849/0.4771 ·
#       r(att) 0.0162 vs r(best unatt) 0.0221 · inner_val_r 0.0582
#
# ⚠️ IL FLAG CHE CAMBIA IL NUMERO, E LA TRAPPOLA CHE HA MORSO:
#    A3 **richiede `--test_eeg raw`**. Il comando documentato SENZA quel flag
#    (cioe' con la release preprocessed al test) da' **0.4870**, non 0.4805 —
#    che e' anche l'accuratezza dell'ASSE 0. Due numeri diversi a una cifra di
#    distanza per un flag non scritto: per questo qui e' esplicito.
#
# ⚠️ A2 (envelope) NON E' PINNATO in results_manifest.tsv, e la ragione e' scritta
#    nel manifest: l'unico run archiviato (`madeeg_ridge_duo_env`) ha un summary
#    che PRECEDE i flag `--estimator` e `--spatial`, quindi il pin non sarebbe
#    VERIFICABILE contro di esso. 0.5065 vive in `_baldo_archive_2026-07-18/` e
#    fuori da RESULTS.md. Rigirandolo con l'intestazione di oggi diventa
#    pinnabile: e' un lavoro di 5 minuti che nessuno ha ancora fatto.
#
# CANARINO   : le tre run SONO i canarini l'una dell'altra nel tempo — i due record
#              pinnati (`madeeg_ridge_{duo,solos}_repro2026-08`) esistono apposta per
#              essere riprodotti byte per byte. Lo script confronta l'md5 di
#              `madeeg_records.csv` con quello dei record pinnati e si ferma se
#              differisce. In piu' il controllo positivo della ridge
#              (`--self_test`, soglia 0.90 DICHIARATA NEL CODICE) gira PER PRIMO.
# CONTRATTO  : vault, "Cap. 2 — criterio pre-registrato (26 lug 2026)" (soglia 88/154)
# PROVENIENZA: RESULTS.md §madeeg_duo <- results_manifest.tsv, pin
#              `madeeg_ridge_duo_repro2026-08` e `madeeg_ridge_solos_repro2026-08` ·
#              A2 in `_baldo_archive_2026-07-18/results/madeeg_ridge_duo_env/`
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 DOSSIER §8.4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
RES="$REPO/runs/results"

mkdir -p "$OUT_DIR"
cd "$REPO"

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
acc()  { grep -h "OVERALL AAD accuracy" "$RES/$1/madeeg_summary.txt"; }
rec()  { echo "$RES/$1/madeeg_records.csv"; }

# Gli invarianti delle tre varianti: identici in tutte, cambia SOLO cio' che la
# variante dichiara di cambiare. Il seed dev'essere lo stesso o i fold cambiano.
BASE=(--madeeg_dir "$MADEEG_DIR" --ensemble duo --estimator ridge --filters pooled
      --eeg_clean none --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250
      --seed 42 --cv_folds 5 --spatial stereo)

# ---- CANARINO 1: il controllo positivo della ridge, PRIMA dei numeri veri ----
# EEG SINTETICO (mistura della sorgente attesa ai lag del modello + rumore, snr 4.0),
# poi la ridge VERA e la decisione VERA. Soglia 0.90 dichiarata nel codice.
echo "--- CANARINO 1: controllo positivo della ridge (--self_test, EEG sintetico) ---"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
      --training_date "${TAG}anchors_ctrl"
S="$RES/${TAG}anchors_ctrl_selftest/madeeg_selftest_summary.txt"
grep -h "AAD accuracy" "$S"
if ! grep -q "\[PASS\]" "$S"; then
  echo "  -> [FALLITO] controllo positivo non superato: i numeri veri NON si guardano."
  exit 1
fi
echo "  -> [PASSATO] licenzia il cablaggio e nient'altro."

# ---- LE TRE VARIANTI ---------------------------------------------------------
echo; echo "--- A1 : duos_kfold, target mel (il massimo storico, 0.5584) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on duos_kfold \
      --target mel --n_mels 8 --training_date "${TAG}ridge_A1_duo_mel"

echo; echo "--- A2 : duos_kfold, target envelope (NON pinnato, vedi l'avvertenza) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on duos_kfold \
      --target envelope --training_date "${TAG}ridge_A2_duo_env"
#   ^ --n_mels NON si passa: `envelope` e' a 1 banda per costruzione.

echo; echo "--- A3 : raw_solos, protocollo del paper. --test_eeg raw E' OBBLIGATORIO ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --train_on raw_solos --test_eeg raw \
      --target mel --n_mels 8 --training_date "${TAG}ridge_A3_solos_mel"

# ---- CANARINO 2: i due record pinnati devono riprodursi byte per byte --------
echo
echo "=== CANARINO 2 — i record pinnati, md5 per md5 ==="
FAILED=0
for P in "${TAG}ridge_A1_duo_mel:madeeg_ridge_duo_repro2026-08" \
         "${TAG}ridge_A3_solos_mel:madeeg_ridge_solos_repro2026-08"; do
  NEW=${P%%:*}; OLD=${P##*:}
  if [ -f "$(rec "$OLD")" ]; then
    A=$(_md5 "$(rec "$OLD")"); B=$(_md5 "$(rec "$NEW")")
    printf "  %-34s pinnato %s\n  %-34s replica %s\n" "$OLD" "$A" "$NEW" "$B"
    if [ "$A" = "$B" ]; then
      echo "  -> [PASSATO]"
    else
      echo "  -> [FALLITO] i record differiscono. Controlla PRIMA l'interprete"
      echo "     (trappola n.1: /opt/anaconda3 muove i float a 2e-5), poi FERMATI e riporta."
      FAILED=1
    fi
  else
    echo "  ⏭️  $OLD non e' su questa macchina (runs/ e' gitignored): niente contro cui confrontare."
  fi
done

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in "${TAG}ridge_A1_duo_mel" "${TAG}ridge_A2_duo_env" "${TAG}ridge_A3_solos_mel"; do
  printf "%-30s " "$R"; acc "$R"
done
echo
echo "soglia pre-registrata (26/7): 88/154 = 0.5714. Nessuna delle tre la supera."
echo "record pinnati : runs/results/madeeg_ridge_{duo,solos}_repro2026-08/"
echo "A2 (envelope)  : _baldo_archive_2026-07-18/results/madeeg_ridge_duo_env/ — fuori da RESULTS.md"
[ "$FAILED" = 0 ] || exit 1
