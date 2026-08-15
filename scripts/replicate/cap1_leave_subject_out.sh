#!/usr/bin/env bash
# =============================================================================
# Cap. 1 — LEAVE-SUBJECT-OUT: il crollo e' per SOGGETTO o per CANZONE?
#
# COSA MISURA: la stessa architettura con un SOGGETTO tenuto fuori invece di una
#   canzone. E' il contrasto che identifica la causa del crollo del Cap. 1:
#   leave-subject-out 0.943 contro leave-song-out 0.258/0.268. Se il modello
#   generalizza fra persone ma non fra brani, quello che ha imparato e' il BRANO.
# DATA       : fold di maggio 2026, verificati 23-26 lug 2026 · ramo raw pinnato 07/2026.
# DOMANDA    : «il crollo e' per soggetto o per canzone?»
# NULL       : 0.25 (4 slot strumento fissi, assunto dal disegno).
# SOGLIA PRE-REGISTRATA: nessuna (riproduzione, non test).
# VERDETTO   : ✅ chiuso — 0.943 su 6 fold (clap) · 0.781 su 3 fold (raw).
# COSTO      : la RIPRODUZIONE della Tabella 2 di Akama dai checkpoint rilasciati
#              e' ~20 min CPU e la fa questo script. Rifare i fold e' ▶️ GPU.
# SGUARDI    : n/a (contabilita' del Cap. 2, non di questo dataset).
# GPU        : NO per la riproduzione dai checkpoint; SI' per rifare i fold.
#
# NUMERI DI RIFERIMENTO ATTESI
#
#   (A) RESULTS.md §subject_out — le due righe della tesi:
#       contrastive_clap  6 fold  MACRO 0.943  GLOBAL 0.946  vintage 2026-05
#       contrastive_raw   3 fold  MACRO 0.781  GLOBAL 0.801  vintage 2026-07
#                                 (pin repro_loso_sub{2,3,7})
#
#   (B) Riproduzione della Tabella 2 di Akama (all-data), dai loro checkpoint —
#       e' quello che stampa `scripts/replicate.sh` in fondo:
#         paper  sub3 0.6458 · sub7 0.8447 · sub2 0.7763 · media 0.7556
#         noi    sub3 0.6642 · sub7 0.8750 · sub2 0.8664 · media 0.8019
#       sub3 e sub7 cadono entro 2-3 punti e riproducono l'INVERSIONE di
#       classifica (sub3 il migliore within -> il peggiore cross-subject; sub7
#       l'opposto). sub2 e' ~9 punti sopra: riproduzione VICINA MA NON ESATTA,
#       e il residuo NON e' ancora spiegato (ipotesi in testa a replicate.sh:
#       composizione delle finestre di test sotto leave-subject-out).
#
# ⚠️ DUE CAVEAT OBBLIGATORI:
#   (a) **5 fold su 6 della riga 0.943 sono PRE-REFACTOR (vintage 2026-05).**
#       Il contrasto 0.943 vs 0.258 confronta DUE EPOCHE DI CODICE. Va citato
#       dichiarando il vintage, non come se fossero la stessa pipeline.
#       I fold 6 e 7 esistono ma non hanno prodotto `test_records.csv`: sono 6,
#       non 8, e la ragione e' quella.
#   (b) **STALENESS APERTA (DOSSIER §5.1):** il vault cita ancora **0.607** come
#       numero «inaffidabile» del ramo raw. Il pin corrente e' **0.781**
#       (repro_loso_sub{2,3,7}). Qualcuno deve chiudere la voce per iscritto.
#
# ⚠️ DISCREPANZA DI TAG, VERIFICATA IL 15/8 E DA RISOLVERE:
#    `scripts/replicate.sh` fase `loso` scrive i tag **akama_loso_sub{3,7,2}**,
#    mentre `results_manifest.tsv` pinna **repro_loso_sub{2,3,7}**. Quindi
#    rilanciare replicate.sh NON aggiorna la riga pinnata: produce una serie
#    PARALLELA. L'intestazione del manifest dice il contrario («the latter is what
#    the committed scripts/replicate.sh produces») ed e' STALE. Non ho cambiato
#    ne' i tag ne' i pin: cambiarli muoverebbe un numero, e non e' compito di
#    questo script. Le due opzioni, entrambe da decidere da A.:
#      1. rinominare i tag in replicate.sh -> `repro_loso_sub{N}` (il numero non
#         cambia, ma la run dev'essere rifatta per riempirli);
#      2. ripinnare il manifest su `akama_loso_sub{N}` DOPO aver verificato che
#         danno le stesse cifre (report.py verifica il pin, non l'uguaglianza).
#    ⚠️ `raw_loso_sub7` (0.408) esiste su disco e CONFLIGGE con repro_loso_sub7
#       (0.875) sullo stesso fold: e' dichiarato NON pinnato nel manifest.
#
# CANARINO   : (1) `src/modules/clip_loss.py`; (2) `src/run.py --selftest`;
#              (3) `report.py` verifica ogni pin contro l'hparams.yaml, e la
#              chiave del fold include `cv_mode` — perche' un leave_subject_out
#              ha gia' oscurato un leave_song_out con lo stesso id numerico
#              (corretto in 10cdc49).
# CONTRATTO  : nessuno pre-registrato (riproduzione).
# PROVENIENZA: RESULTS.md §subject_out <- results_manifest.tsv ·
#              intestazione di scripts/replicate.sh per la tabella (B)
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}
PHASES=${PHASES:-"loso report"}

cd "$REPO"

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
else
  echo "  ⏭️  NON ATTRAVERSATO (nessun torch). PY_TORCH=... bash \$0"
fi

echo
echo "=== CANARINO 2 — src/run.py --selftest ==="
"$PY" src/run.py --selftest

echo
echo "=== (A) I NUMERI PINNATI (nessuna run: si rileggono) ==="
# report.py --check esce 1 se anche UN pin non e' verificabile: con `set -e` + pipefail
# una pipe farebbe morire lo script invece di dirlo. Si cattura l'uscita e la si dichiara.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## subject_out/,/^## within/p'
if [ "$RC" != 0 ]; then
  echo "⚠️ report.py --check e' uscito $RC — pin non verificabili su questa macchina:"
  echo "$CHECK_OUT" | grep "^\[BAD PIN\]" || true
fi

echo
echo "=== (B) RIPRODUZIONE DELLA TABELLA 2 DI AKAMA, dai checkpoint rilasciati ==="
if [ ! -f "$REPO/checkpoints/model-sub3.ckpt" ]; then
  echo "mancano i checkpoint model-sub{3,7,2}.ckpt."
  echo "  bash scripts/setup_checkpoints.sh"
  exit 1
fi
# ⚠️ replicate.sh sorgente sweep_common.sh, che attiva `eeg_attention` (il nome
#    dell'env SU BALDO). Sul Mac l'env del Cap. 1 e' `attention`: passa PY per path.
# ⚠️ `env`, non un prefisso nudo: `${PY_TORCH:+PY=...}` viene ESPANSO dopo che bash ha
#    gia' deciso cosa e' un'assegnazione, quindi diventerebbe il NOME del comando
#    ("PY=/path/python: command not found") e con set -e lo script morirebbe qui.
env PHASES="$PHASES" ${PY_TORCH:+PY="$PY_TORCH"} bash scripts/replicate.sh

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
echo "la tabella paper-vs-noi la stampa replicate.sh qui sopra."
echo "⚠️ i tag scritti sono akama_loso_sub{3,7,2}, NON i pin repro_loso_sub{2,3,7}:"
echo "   vedi la DISCREPANZA DI TAG in testa a questo script."

cat <<'EOF'

###############################################################################
# ▶️ RIFARE I FOLD RICHIEDE GPU — LI LANCIA A. ([[Comandamenti]] §11)
###############################################################################
cd $REPO
python src/run.py sweep --cv subject                  # 4 modelli x 8 soggetti
CUDA_VISIBLE_DEVICES=0 CAP=80m bash sweeps/sweep_subject_out.sh    # equivalente esplicito

# un solo fold, se serve chiudere un buco (es. i fold 6 e 7 senza test_records.csv):
python src/run.py train --model clap --cv subject --held 6 --test --tag cv_sweep_leave_subject_out_6

# ⚠️ nohup, non tmux (su baldo tmux non c'e'):
cd $REPO && nohup python src/run.py sweep --cv subject > runs/logs/lso_subject.log 2>&1 &
python src/run.py report
EOF
