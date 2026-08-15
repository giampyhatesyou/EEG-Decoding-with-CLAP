#!/usr/bin/env bash
# =============================================================================
# Cap. 1 — WITHIN-SPLIT (stesse canzoni in train e test) + i controlli negativi
#          + il controllo di leakage solo-audio.
#
# COSA MISURA: se il contrastivo EEG<->audio di Akama riproduce la baseline
#   pubblicata quando le STESSE canzoni stanno in training e in test; e, accanto,
#   se l'eval e' onesta (etichette permutate -> caso) e se l'audio DA SOLO
#   identifica gia' il bersaglio (0.996 / 0.967: e' il confound del capitolo).
# DATA       : within pinnato 18/7/2026 · controlli negativi 18/7/2026.
# DOMANDA    : «il modello riproduce la baseline pubblicata sulle stesse canzoni?»
#              e, subito dopo, «cosa dell'accuratezza NON viene dall'EEG?»
# NULL       : 0.25 — assunto dal disegno a 4 slot strumento fissi, non misurato.
# SOGLIA PRE-REGISTRATA: nessuna. Non e' un test: e' una RIPRODUZIONE, e il
#              criterio l'ha fissato il paper (0.865). L'unica soglia scritta e'
#              quella diagnostica 0.30 dentro lo script dei controlli negativi.
# VERDETTO   : ✅ riprodotto — GLOBAL 0.8650 ESATTO dal checkpoint.
# COSTO      : ~25 min CPU (within ~10' + sanity ~15'). Nessun training.
# SGUARDI    : n/a — il registro degli sguardi e' una contabilita' del Cap. 2
#              (i duo di MAD-EEG). Il Cap. 1 gira sul dataset di Akama.
# GPU        : NO per la parte pinnata (tutto dai checkpoint RILASCIATI dagli
#              autori). SI' per il blocco ▶️ in fondo (audio_only / eeg_only:
#              non esistono checkpoint rilasciati per quelli, vanno addestrati).
#
# ⚠️ LA FRASE OBBLIGATORIA IN OGNI CITAZIONE DI QUESTO NUMERO:
#    **within = la divisione confusa.** Le stesse canzoni stanno in train e in
#    test, quindi PIU' ALTO = MEMORIZZA MEGLIO, non «decodifica meglio». Accanto
#    al within va sempre il leave-song-out (0.268) e il controllo solo-audio
#    (0.996): l'ordine di quei tre numeri E' il risultato del capitolo.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   contrastive_raw   within   MACRO 0.875   GLOBAL 0.865   (pin `sanity_none`,
#                              ckpt model-all0, 1 fold, vintage 2026-07)
#                              ⚠️ deterministico: 0.8650 e' ESATTO, bit per bit.
#   controllo `labels`         MACRO 0.227   GLOBAL 0.225
#   controllo `audio_pair`     MACRO 0.339   GLOBAL 0.383
#   ⚠️ I DUE CONTROLLI NON SONO RIPRODUCIBILI BIT-A-BIT (manifest, riga
#      «NOT REPRODUCIBLE BIT-FOR-BIT»): `checkpoint_test` forza `shuffle=True`
#      per shuffle_test_mode != none, e `--workers` si auto-dimensiona sulla
#      macchina, quindi la permutazione cambia da run a run. Rigirati il 26/7
#      hanno dato 0.2500 e 0.3525. In tesi vanno citati come «al livello del
#      caso» e «molto sotto lo 0.865», MAI come cifre esatte.
#   ▶️ (GPU) audio_only_clap within  MACRO 0.996  GLOBAL 0.993   <- il confound
#   ▶️ (GPU) audio_only_raw  within  MACRO 0.967  GLOBAL 0.963
#   ▶️ (GPU) eeg_only        within  MACRO 0.250  GLOBAL 0.471
#   ▶️ (GPU) contrastive_clap within MACRO 0.946  GLOBAL 0.935   (8 semi)
#
# ⚠️ CONFLITTO DI CIFRE ANCORA APERTO (DOSSIER §5.2): in giro nel vault esiste
#    anche la coppia 0.865/0.852 per la stessa riga. La fonte canonica e'
#    `results_manifest.tsv` -> `RESULTS.md`, cioe' 0.875 MACRO / 0.865 GLOBAL.
#    Chi scrive la tesi deve chiudere la voce, non scegliere in silenzio.
#
# CANARINO   : (1) `python src/modules/clip_loss.py` — l'aritmetica InfoNCE che
#              regge OGNI numero del Cap. 1 (0.628491 · 1.2994 · 4.8198). Serve
#              torch: se l'interprete non ce l'ha, lo script lo dichiara e NON
#              finge di averlo attraversato.
#              (2) `python src/run.py --selftest` — i 23 flag del protocollo
#              vivono in due posti (PROTOCOL in run.py e $PROTO in
#              sweeps/sweep_common.sh) e devono coincidere flag per flag.
#              (3) a valle, `report.py` verifica OGNI pin contro l'hparams.yaml
#              del suo run: se uno non torna, RESULTS.md non viene scritto affatto.
# CONTRATTO  : nessuno pre-registrato — il Cap. 1 e' una riproduzione, non un
#              test. La disciplina applicabile e' [[Comandamenti]] §3 (controllo
#              positivo prima) e §9 (un controllo non scrive sopra un risultato).
# PROVENIENZA: RESULTS.md §within e §control <- results_manifest.tsv
#              (pin `sanity_none`, `sanity_labels`, `sanity_audio_pair`)
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}          # solo per i canarini stdlib (selftest/report)
PY_TORCH=${PY_TORCH:-}                        # env con torch: conda `attention` sul Mac
PHASES=${PHASES:-"within sanity report"}      # sottoinsieme delle fasi di replicate.sh

cd "$REPO"

# ---- CANARINO 1: l'aritmetica della loss, PRIMA di tutto -------------------
echo "=== CANARINO 1 — src/modules/clip_loss.py (sintetico, nessun dato) ==="
if [ -z "$PY_TORCH" ]; then
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/modules/clip_loss.py
  echo "  atteso: 0.628491 · 1.2994 · 4.8198  e passo D  0.951610 vs 2.160834"
else
  echo "  ⏭️  NON ATTRAVERSATO: nessun interprete con torch trovato."
  echo "      PY_TORCH=~/miniconda3/envs/attention/bin/python bash \$0"
  echo "      ⚠️ Questo NON e' un pass: e' un canarino non attraversato."
fi

# ---- CANARINO 2: il protocollo vive in due posti e devono coincidere -------
echo
echo "=== CANARINO 2 — src/run.py --selftest (23 flag di protocollo) ==="
"$PY" src/run.py --selftest

# ---- LA RIPRODUZIONE -------------------------------------------------------
# replicate.sh e' resumable: una fase il cui test_breakdown_summary.txt esiste
# gia' viene SALTATA. Per rifarla davvero, rimuovi la sua directory sotto
# runs/results/ (che e' gitignored: non stai cancellando niente di tracciato).
echo
echo "=== LA RIPRODUZIONE — dai checkpoint RILASCIATI, nessun training ==="
if [ ! -f "$REPO/checkpoints/model-all0.ckpt" ]; then
  echo "manca checkpoints/model-all0.ckpt."
  echo "  bash scripts/setup_checkpoints.sh     # li spacchetta da archive/*.7z"
  exit 1
fi
# ⚠️ replicate.sh sorgente sweeps/sweep_common.sh, che attiva l'env conda
#    `eeg_attention` — il nome che l'env ha SU BALDO. Sul Mac l'env del Cap. 1 si
#    chiama `attention`: li' `conda activate eeg_attention` e' un no-op e $PY
#    ricade su `python` nudo. Passa l'interprete per path, sempre:
#      PY=~/miniconda3/envs/attention/bin/python PHASES="within sanity" bash scripts/replicate.sh
# ⚠️ `env`, non un prefisso nudo: `${PY_TORCH:+PY=...}` viene ESPANSO dopo che bash ha
#    gia' deciso cosa e' un'assegnazione, quindi finirebbe per essere il NOME del comando
#    ("PY=/path/python: command not found") e con set -e lo script morirebbe qui.
env PHASES="$PHASES" ${PY_TORCH:+PY="$PY_TORCH"} bash scripts/replicate.sh

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
# report.py --check esce 1 se anche UN pin non e' verificabile (in un clone pulito,
# senza runs/, sono tutti). Con `set -e` + pipefail una pipe lo farebbe morire PRIMA di
# stampare la tabella e la ragione: si cattura l'uscita e la si dichiara.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## within/,$p'
if [ "$RC" != 0 ]; then
  echo "⚠️ report.py --check e' uscito $RC: uno o piu' pin non sono verificabili su questa"
  echo "   macchina (runs/ e' gitignored e i fold vivono dove sono stati girati). Le righe"
  echo "   [BAD PIN] sono qui sopra. NON e' un via libera: un pin non verificabile non e' un pin."
fi
echo
echo "provenienza: RESULTS.md §within e §control <- results_manifest.tsv"
echo "⚠️ within = la divisione confusa: piu' alto = MEMORIZZA meglio, non decodifica meglio."

cat <<'EOF'

###############################################################################
# ▶️ IL RESTO DELLA TABELLA `within` RICHIEDE TRAINING — LO LANCIA A. (§11)
###############################################################################
# Non esistono checkpoint rilasciati per audio_only / eeg_only / contrastive_clap:
# quelle righe di RESULTS.md vengono da run NOSTRE, e per rifarle si addestra.
#
#   ⚠️ audio_only within = IL CONTROLLO DI LEAKAGE, ed e' la prova del confound
#      del Cap. 1: con il solo audio si arriva a 0.996 (clap) / 0.967 (raw).
#      Il brano identifica la risposta. Va citato ACCANTO allo 0.865, mai dopo.

cd $REPO
python src/run.py train --model audio_only --cv within --test --tag clf_audio_raw_within
python src/run.py train --model eeg_only   --cv within --test --tag clf_eeg_within
python src/run.py train --model clap       --cv within --test --tag clapseed_within_42

# audio_only con l'audio CLAP (la riga 0.996) passa da main.py, perche' `--model
# audio_only` di run.py pinna audio_repr=raw:
python src/main.py --objective classify_audio --audio_repr clap --cv_mode within \
    --cv_held_out_id -1 --training_date clf_audio_clap_within   # + $PROTO
#   ^ $PROTO = i 23 flag di sweeps/sweep_common.sh. run.py li emette da solo;
#     a mano si prendono DA LI', non si riscrivono a memoria.

# Poi, per far entrare il fold nella tabella: si aggiunge una riga a
# results_manifest.tsv e si rigenera. report.py verifica il pin contro
# l'hparams.yaml del run e RIFIUTA di scrivere se non torna.
python src/run.py report
EOF
