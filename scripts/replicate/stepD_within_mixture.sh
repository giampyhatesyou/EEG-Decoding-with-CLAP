#!/usr/bin/env bash
# =============================================================================
# Passo D — la loss WITHIN-MIXTURE: allineare l'obiettivo al compito.
#           ▶️ IL TRAINING E' GPU: questo script prepara, verifica e STAMPA.
#
# COSA MISURA: il passo C usa NEGATIVI DI BATCH — la loss di una finestra dipende
#   da chi le e' capitato accanto nel batch, che e' informazione che il compito non
#   ha. Il passo D restringe i negativi ALLA STESSA MIXTURE: per ogni finestra, il
#   solo negativo e' l'altro stem dello stesso duo. La domanda e' se allineare la
#   loss al compito toglie la scorciatoia.
# DATA       : 28 lug 2026.
# DOMANDA    : «allineare la loss al compito toglie la scorciatoia?»
# NULL       : **0.500 per costruzione** sul duo (ogni mixture compare con
#              entrambi i suoi strumenti come target).
# SOGLIA PRE-REGISTRATA: **88/154 = 0.5714** — la STESSA del passo C, non ricalcolata.
# VERDETTO   : 🔴🔴 **PRIMARIO FALLITO + SECONDARIO FALSIFICATO AL CONTRARIO.**
#              Primario 70/154 = 0.4545, IC95 [0.374, 0.537], p 0.8867.
#              E il secondario pre-dichiarato («il prior-following deve COLLASSARE
#              verso 0.4274») e' andato **nella direzione opposta**: 0.7177 -> 0.8145.
#              *«Abbiamo tolto la scorciatoia»* era la riga da scrivere se fosse
#              collassato: **e' falsa, e va scritta come falsa.**
# COSTO      : ▶️ GPU. Cancelli su CPU: ~2 min.
# SGUARDI    : SI', ed e' un **SECONDO SGUARDO sugli STESSI 154 trial**, deciso
#              DOPO che il numero del passo C era noto ⟹ **ESPLORATIVO PER
#              COSTRUZIONE**, e il caveat e' scritto DENTRO il summary della run
#              (le prime 4 righe del file). 🔒 I TRIO NON SI TOCCANO.
# GPU        : SI'. [[Comandamenti]] §11: si prepara il comando, lo lancia A.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO            **70/154 = 0.4545**  IC95 [0.374, 0.537]  p 0.8867
#   CONTROLLO POSITIVO  **142/154 = 0.9221**  IC95 [0.868, 0.959]
#                       soglia **0.90 DICHIARATA NEL CODICE PRIMA DELLA RUN**
#   SECONDARIO 3 (verificato): train_loss della PRIMA epoca **0.665-0.692** contro
#                       log 2 = **0.6931**, che era il valore dichiarato PRIMA —
#                       cioe' la prova che con un solo negativo la loss parte dal
#                       caso giusto. ✅ confermato.
#   iperparametri: ensemble=duo folds=5 epochs=10 lr=0.003 batch=8 seed=42
#                  loss=within_mixture
#
# 🔑 L'ARITMETICA C -> D, che e' il fatto piu' utile del braccio:
#      cella «prior GIUSTO»    32/53 -> **44/53**  (+12)
#      cella «prior SBAGLIATO» 14/71 -> **14/71**  ( 0)
#    Il null e' 0.500 in ENTRAMBE le celle. Tutti i +12 trial guadagnati vengono
#    dalla cella dove il prior dello stimolo era gia' dalla parte giusta, e ZERO
#    dall'altra. Un seguace **puro** del prior prenderebbe ≈68/154 = 0.4416;
#    osservato 70/154. Il passo D non ha imparato a decidere: ha imparato a
#    seguire meglio il prior.
#
# ⚠️ **IL CANCELLO DEI TRIO NON SI E' APERTO.** Richiedeva ≥ 0.5714 **E** una
#    diagnosi collassata: fallite entrambe. I 92 trio stereo + 93 mono restano
#    intatti. Questo e' il motivo per cui esistono ancora.
#
# ⚠️ IL SECONDARIO E' UN RISULTATO, non un dettaglio: era la parte falsificabile
#    del passo D. Che sia andato al contrario e' la ragione per cui il meccanismo
#    del capitolo si chiama «prior-following» e non «rumore».
#
# CANARINO   : (1) `src/modules/clip_loss.py` — e la SECONDA riga del suo
#              self-check e' **esattamente l'obiettivo del passo D**: within-mixture
#              **0.951610** contro batch **2.160834**, verificato contro
#              l'aritmetica calcolata a mano dentro il file. Prova che il flag
#              CAMBIA DAVVERO l'obiettivo (un flag che non cambia niente
#              produrrebbe una run che sembra il braccio nuovo ed e' il vecchio).
#              (2) `--loss` e' ASSERTITO: `within-mixture` col trattino ROMPE.
#              (3) Il self-check verifica anche la PROPRIETA' su cui poggia tutto
#              il passo: sotto within-mixture la loss di una finestra e'
#              INDIPENDENTE dai suoi compagni di batch (e sotto batch NO — e' cosi'
#              che si sa che il controllo discrimina).
#              (4) Directory SEPARATE: `madeeg_clap_kfold_within` e
#              `madeeg_clap_selftest_within`. Un controllo non scrive mai sopra un
#              risultato, e nemmeno un obiettivo diverso ([[Comandamenti]] §9).
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — passo D: loss
#              within-mixture, criterio pre-registrato (28 lug 2026)" · lettura:
#              "10 Ricerca/Cap. 2 — cosa mostra il passo D (28 lug 2026)"
# PROVENIENZA: `_baldo_archive_2026-07-18/results/madeeg_clap_{kfold,selftest}_within/
#              madeeg_contrastive_summary.txt` (verificati l'11/8) ·
#              aritmetica C->D in docs/provenance/2026-08-11_diagnose_step{C,D}.txt
#              ⚠️ **NON PINNATO in results_manifest.tsv**, e la ragione e' scritta
#              nel manifest: gli artefatti non erano su questo Mac. Va pinnato il
#              giorno in cui atterrano.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}

cd "$REPO"

# ---- CANCELLO 1: il self-check della loss, che INCLUDE l'obiettivo del passo D
echo "=== CANCELLO 1 — src/modules/clip_loss.py (la 2ª riga E' l'obiettivo del passo D) ==="
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
  OUT=$("$PY_TORCH" src/modules/clip_loss.py 2>&1); echo "$OUT"
  if echo "$OUT" | grep -q "0.951610" && echo "$OUT" | grep -q "2.160834"; then
    echo "  -> [PASSATO] within-mixture 0.951610 vs batch 2.160834: il flag CAMBIA l'obiettivo."
  else
    echo "  -> [FALLITO] i due valori del passo D non sono quelli pinnati. FERMATI e riporta."
    exit 1
  fi
else
  echo "  ⏭️  NON ATTRAVERSATO (nessun torch trovato). PY_TORCH=... bash \$0"
  echo "      ⚠️ Prima del training su baldo e' OBBLIGATORIO."
fi

# ---- CANCELLO 2: un --loss mal scritto DEVE rompere -------------------------
echo
echo "=== CANCELLO 2 — un --loss mal scritto deve ROMPERE, non ricadere sul default ==="
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  if "$PY_TORCH" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" \
       --loss within-mixture --smoke >/dev/null 2>&1; then
    echo "  -> [FALLITO] '--loss within-mixture' (col trattino) e' stato ACCETTATO."
    echo "     Una run cosi' sembrerebbe il passo D e sarebbe il passo C. FERMATI."
    exit 1
  fi
  echo "  -> [PASSATO] rifiutato, come deve."
else
  echo "  ⏭️  serve un interprete con torch."
fi

# ---- CANCELLO 3: la regola di decisione e' UNA sola -------------------------
echo
echo "=== CANCELLO 3 — --check_rule: la regola del passo D e quella del passo C ==="
echo "    coincidono su 2000 trial duo casuali, e riproducono il passo C a 58/154."
REC="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC" ]; then
  "$PY" src/madeeg_diagnose.py --check_rule "$REC"
else
  echo "  ⏭️  manca $REC (archivio fuori dal repo). ARCHIVE=... per puntarlo altrove."
fi

cat <<'EOF'

###############################################################################
# ▶️ IL TRAINING — LO LANCIA A., SUL NODO GPU. [[Comandamenti]] §11.
###############################################################################
REPO=$HOME/EEG-Attention-decoding-with-CLAP
PY=$HOME/.conda/envs/eeg_attention/bin/python
MAD=$HOME/madeeg

cd $REPO && git pull && mkdir -p runs/logs

# (1) PRE-VOLO OBBLIGATORIO:
$PY src/modules/clip_loss.py     # 0.628491 · 1.2994 · 4.8198 · 0.951610 vs 2.160834

# (2) IL CONTROLLO POSITIVO DEL PASSO D — G2. Soglia 0.90 NEL CODICE, PRIMA.
#     EEG SINTETICO. Directory SEPARATA: madeeg_clap_selftest_within/
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --self_test --loss within_mixture --kfold 5 \
    > runs/logs/stepD_selftest_within.log 2>&1 &
#     atteso: 142/154 = 0.9221, IC95 [0.868, 0.959] -> ABOVE CHANCE

# (3) IL NUMERO VERO — G3. Solo dopo che (2) e' passato.
#     Directory SEPARATA: madeeg_clap_kfold_within/  (mai sopra il passo C)
cd $REPO && nohup $PY src/madeeg_contrastive.py \
    --madeeg_dir $MAD --loss within_mixture --kfold 5 \
    > runs/logs/stepD_kfold_within.log 2>&1 &
#     atteso: 70/154 = 0.4545, IC95 [0.374, 0.537], p 0.8867
#     soglia 88/154 = 0.5714 -> NON DISTINGUIBILE DAL CASO
#     ⚠️ da controllare NEL LOG: la train_loss della PRIMA epoca dev'essere
#        0.665-0.692, contro log 2 = 0.6931 dichiarato prima. E' il secondario 3.

# (4) POI, SUL MAC: la diagnostica che dice cosa e' successo davvero (CPU).
#     ⚠️ E' QUI che il passo D si falsifica: il prior-following NON collassa,
#        sale (0.7177 -> 0.8145). Il primario da solo non lo direbbe.
#     ⚠️ NON esiste uno script `diagnose_prior_stepCD.sh`: sono due comandi, ed e'
#        cosi' che sono stati prodotti i due file pinnati sotto docs/provenance/.
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv
#     -> passo C: 58/154 · prior-following 89/124 = 0.7177 · celle 32/53 e 14/71 · OR 6.20
/opt/miniconda3/bin/python src/madeeg_diagnose.py --madeeg_dir ~/madeeg \
    --records ../_baldo_archive_2026-07-18/results/madeeg_clap_kfold_within/madeeg_contrastive_records.csv
#     -> passo D: 70/154 · prior-following 101/124 = 0.8145 · celle 44/53 e 14/71 · OR 19.90
#     provenienza pinnata: docs/provenance/2026-08-11_diagnose_step{C,D}.txt
EOF
