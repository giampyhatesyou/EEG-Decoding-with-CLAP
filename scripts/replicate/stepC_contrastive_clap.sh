#!/usr/bin/env bash
# =============================================================================
# Passo C — il contrastivo CLAP<->EEG con negativi di BATCH, su MAD-EEG.
#           ▶️ IL TRAINING E' GPU: questo script prepara, verifica e STAMPA.
#
# COSA MISURA: se un modello contrastivo CLIP-style — la stessa aritmetica InfoNCE
#   del Cap. 1, con un campionatore diverso — decide quale dei due strumenti di un
#   duo era attenzionato. E' il trasferimento diretto dell'architettura del Cap. 1
#   su un dataset dove la scorciatoia del Cap. 1 non puo' vincere.
# DATA       : 27 lug 2026 (la run valida). ⚫ Una run precedente del 26/7 e'
#              ANNULLATA, vedi sotto.
# DOMANDA    : «il contrastivo CLAP↔EEG con negativi di batch decide l'attenzione?»
# NULL       : **0.500 per costruzione** — ogni mixture duo compare con ENTRAMBI i
#              suoi strumenti come target (18/18), quindi «quale stem e' attenzionato»
#              non e' una proprieta' dello stimolo.
# SOGLIA PRE-REGISTRATA: **88/154 = 0.5714** (minimo intero con binomiale esatta a
#              una coda < 0.05 su n = 154). Contratto del 26/7, scritto prima del codice.
# VERDETTO   : 🔴 **SOTTO IL CASO, E SPIEGATO** — 58/154 = 0.3766, IC95
#              [0.300, 0.458], p a una coda 0.9992.
#              ⚠️ **L'IC ESCLUDE 0.50**: e' sistematicamente sbagliato, non casuale.
#              🚫 **NON SI GIRA IL SEGNO** ([[Comandamenti]] §6). Invertendo la
#              decisione verrebbe 0.62, e sarebbe un numero senza ipotesi dietro.
#              La spiegazione e' il prior-following: `madeeg_diagnose.py --records`,
#              il comando in fondo a questo script (provenienza pinnata
#              docs/provenance/2026-08-11_diagnose_stepC.txt).
# COSTO      : ▶️ GPU. Preparazione e cancelli su CPU: ~2 min.
# SGUARDI    : SI' — decide sui duo. Al 27/7 era il PRIMO sguardo del braccio
#              contrastivo, quindi CONFERMATIVO. Oggi i 309 duo sono tutti spesi:
#              rigirarlo non apre materiale nuovo. 🔒 I TRIO NON SI TOCCANO.
# GPU        : SI'. [[Comandamenti]] §11: gli agenti preparano il comando, lo lancia A.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO            **58/154 = 0.3766**  IC95 [0.300, 0.458]  p 0.9992
#   CONTROLLO POSITIVO  **148/154 = 0.9610** su EEG SINTETICO
#                       soglia **0.90 DICHIARATA NEL CODICE PRIMA DELLA RUN**
#                       IC95 [0.917, 0.986]
#   iperparametri della run: ensemble=duo folds=5 epochs=10 lr=0.003 batch=8 seed=42
#
# ⚫ **LA RUN 1 DEL 26/7 E' ANNULLATA, e resta agli atti** ([[Comandamenti]] §12).
#    Dava 61/154 = 0.3961. Il bug: la chiave `stim` non era unica fra soggetti,
#    quindi le decisioni NON erano indipendenti e i margini si fondevano fra
#    soggetti diversi. **Quel numero non entra da nessuna parte.** Corretto in
#    `69b07a6` con un assert: l'unita' di decisione e' il trial, con chiave
#    `(subject, stim)`, e c'e' un assert a difenderla.
#
# ⚠️ IL CONTROLLO POSITIVO NON DICE NIENTE SUI DATI VERI. L'EEG e' SINTETICO (un
#    operatore lineare fisso sull'inviluppo della sorgente attesa, snr 4.0). Il
#    148/154 dice che il cablaggio funziona, non che il modello decodifica. Il
#    caveat sta DENTRO il file di summary, sopra i numeri — ed e' stato aggiunto
#    dopo un incidente vero (`7021cec`: un summary diceva `ABOVE CHANCE` senza
#    dire che l'EEG era sintetico).
#
# ⚠️ IL PASSO C **NON FILTRA** IN 1-8 Hz. Il filtro e' del braccio di
#    ricostruzione (`madeeg_reconstruction.py`: 13 occorrenze; il contrastivo: 0).
#    Quindi il 0.3766 NON dipende dalla scelta di banda, e la voce aperta sulla
#    banda 1-8 Hz non lo tocca.
#
# CANARINO   : (1) `src/modules/clip_loss.py` — l'aritmetica InfoNCE e' LA STESSA
#              del Cap. 1: se si muove, si muovono anche 0.865 e 0.943.
#              (2) `--loss` e' ASSERTITO: un valore mal scritto ROMPE, non ricade
#              in silenzio sul default (`batch`).
#              (3) `_self_check` e `_check_soli_row_order` dentro l'adapter del dataset.
#              (4) `--check_rule` sui record archiviati -> **58/154 ESATTO**, e
#              concorde TRIAL PER TRIAL: la regola di decisione ha una sola
#              implementazione (`trial_decision`) ed e' rigirabile offline.
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — criterio
#              pre-registrato (26 lug 2026)" · lettura: "10 Ricerca/Cap. 2 — cosa
#              mostra il passo C (27 lug 2026)"
# PROVENIENZA: RESULTS.md §madeeg_duo <- results_manifest.tsv, pin `madeeg_clap_kfold` ·
#              record in `_baldo_archive_2026-07-18/results/madeeg_clap_{kfold,selftest}/`
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # per i cancelli CPU senza torch
PY_TORCH=${PY_TORCH:-}                        # per clip_loss.py (serve torch)
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}

cd "$REPO"

# ---- CANCELLO 1: l'aritmetica della loss (la stessa del Cap. 1) -------------
echo "=== CANCELLO 1 — src/modules/clip_loss.py (sintetico, nessun dato, nessuna GPU) ==="
if [ -z "$PY_TORCH" ]; then
  # L'env di RIFERIMENTO del canarino e' conda `attention` (torch 2.2.2, numpy<2):
  # va cercato PRIMA di /opt/anaconda3 base (torch 2.11), che resta l'ultima risorsa.
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "/opt/anaconda3/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/modules/clip_loss.py
  echo "  atteso: 0.628491 · 1.2994 · 4.8198  e passo D  0.951610 vs 2.160834"
else
  echo "  ⏭️  NON ATTRAVERSATO (nessun torch trovato). PY_TORCH=... bash \$0"
  echo "      ⚠️ Questo non e' un pass. Prima del training su baldo e' OBBLIGATORIO."
fi

# ---- CANCELLO 2: la regola di decisione, sui record archiviati --------------
echo
echo "=== CANCELLO 2 — --check_rule sui record del passo C: deve dare 58/154 ESATTO ==="
REC="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC" ]; then
  "$PY" src/madeeg_diagnose.py --check_rule "$REC"
else
  echo "  ⏭️  manca $REC"
  echo "      L'archivio di baldo e' FUORI dal repo. ARCHIVE=... per puntarlo altrove."
fi

# ---- CANCELLO 3: il self-check dell'adapter --------------------------------
echo
echo "=== CANCELLO 3 — self-check dell'adapter del dataset contrastivo ==="
# ⚠️ `--ensemble duo` ESPLICITO. Il default dell'adapter e' `both`, che costruisce
#    anche i trial TRIO: sono l'unico holdout rimasto (92 stereo + 93 mono, un
#    colpo solo) e uno smoke test non ne ha bisogno. Deviazione DICHIARATA rispetto
#    al comando che sta nel vault; non puo' muovere nessun numero perche' lo smoke
#    non ne produce.
if [ -d "$MADEEG_DIR" ] && [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/datasets/madeeg_contrastive_dataset.py \
      --madeeg_dir "$MADEEG_DIR" --ensemble duo \
    || echo "  ⚠️ l'adapter e' uscito con errore: NON lanciare il training."
else
  echo "  ⏭️  serve sia il dataset in $MADEEG_DIR sia un interprete con torch."
fi

cat <<'EOF'

###############################################################################
# ▶️ IL TRAINING — LO LANCIA A., SUL NODO GPU. [[Comandamenti]] §11.
###############################################################################
# Su baldo NON c'e' tmux: nohup, cosi' la sessione web puo' cadere senza
# portarsi via la run. `conda activate` NON funziona in shell non interattive:
# si passa l'interprete PER PATH.

REPO=$HOME/EEG-Attention-decoding-with-CLAP     # ⚠️ su baldo il repo si e' chiamato
                                                #    anche ~/akami: verifica il nome
PY=$HOME/.conda/envs/eeg_attention/bin/python
MAD=$HOME/madeeg

cd $REPO && git pull && mkdir -p runs/logs

# (1) PRE-VOLO OBBLIGATORIO — il canarino, con torch vero:
$PY src/modules/clip_loss.py

# (2) IL CONTROLLO POSITIVO, PRIMA del numero vero. Soglia 0.90 NEL CODICE.
#     EEG SINTETICO: dice che il cablaggio funziona, MAI che il modello decodifica.
#     Scrive in madeeg_clap_selftest/, directory SEPARATA ([[Comandamenti]] §9).
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --self_test --kfold 5 \
    > runs/logs/stepC_selftest.log 2>&1 &
#     atteso: 148/154 = 0.9610, IC95 [0.917, 0.986]  -> ABOVE CHANCE
#     ⚠️ se NON passa la soglia 0.90, il numero vero non si guarda. Ci si ferma.

# (3) IL NUMERO VERO — solo dopo che (2) e' passato.
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --kfold 5 \
    > runs/logs/stepC_kfold.log 2>&1 &
#     atteso: 58/154 = 0.3766, IC95 [0.300, 0.458], p 0.9992
#     soglia pre-registrata 88/154 = 0.5714 -> NON DISTINGUIBILE DAL CASO
#     ⚠️ --loss ha default `batch`: il passo C E' il default, non un flag.

# DOVE FINISCE TUTTO:
#   runs/results/madeeg_clap_kfold/madeeg_contrastive_{records.csv,summary.txt}
#   runs/results/madeeg_clap_selftest/...
#   ⚠️ runs/ e' gitignored: per portare i record sul Mac serve un rsync, ed e'
#      il motivo per cui i numeri del passo D non erano pinnabili fino all'11/8.

# (4) POI, SUL MAC, la diagnostica che SPIEGA il 0.3766 (CPU, nessuna GPU).
#     ⚠️ NON esiste uno script `diagnose_prior_stepCD.sh`: la diagnostica e' un
#        comando solo, ed e' esattamente quello che ha prodotto il file pinnato
#        docs/provenance/2026-08-11_diagnose_stepC.txt (`folds=global` e' il default).
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv
#     atteso: 58/154 = 0.3766 · prior-following 89/124 = 0.7177 con null 0.4274 (NON 0.5)
#             · celle 32/53 e 14/71 · Fisher p 4.87e-06 · odds ratio 6.20
EOF
