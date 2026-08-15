#!/usr/bin/env bash
# =============================================================================
# I QUATTRO ASSI del protocollo pubblicato (+ ASSE 0) — quale asse muove il segnale?
#
# COSA MISURA: il protocollo di Cantisani et al. (WASPAA 2019) differisce dal
#   nostro su cinque assi indipendenti. Questo script li muove UNO PER VOLTA,
#   partendo dalla base `raw_solos`, e li giudica su `inner_val_r` — la qualita'
#   di ricostruzione su uno split INTERNO al materiale di TRAINING.
# DATA       : 29 lug 2026.
# DOMANDA    : «quale asse del protocollo pubblicato muove il segnale?»
# NULL       : nessuno — `inner_val_r` e' una metrica di RICOSTRUZIONE, non una
#              decisione. Non ha un caso e non si testa contro 0.5.
# SOGLIA PRE-REGISTRATA: nessuna. E' un passo di CONFIGURAZIONE, non un test:
#              per questo si misura su `inner_val_r` e non sull'accuratezza.
# VERDETTO   : ✅ un solo asse migliora davvero — l'ICA (ASSE 1b, +16%).
# COSTO      : ~30-60 min CPU. DA CONFERMARE: le run a --target_fs 256 (ASSE 0 e
#              MAG@256) sono ~6 GB di matrice di disegno l'una. ⚠️ 16 GB di RAM.
# SGUARDI    : ⚠️ SI', COME SCRITTO QUI. `inner_val_r` di per se' e' GRATUITO (e'
#              calcolato su uno split interno al materiale di TRAINING e non tocca
#              nessuna decisione), ma le run che lo hanno prodotto sono run PIENE:
#              costruiscono i trial di test e stampano anche l'accuratezza sul duo.
#              Leggerla e' uno sguardo, e i 309 duo sono gia' tutti spesi.
#              Per il lavoro di CONFIGURAZIONE esiste `--inner_val_only`, che non
#              costruisce nemmeno un trial di test — ma ⚠️ **il suo numero NON e'
#              lo stesso**: media NON pesata su (soggetto x fold) invece che sui
#              TRIAL. Sulla base raw_solos ridge: **0.0562 con `--inner_val_only`
#              contro 0.0582 pesato sui trial**, e la differenza e' dichiarata
#              dentro il summary dello script stesso. Si confronta inner_val_only
#              con inner_val_only, MAI con i numeri pinnati qui sotto.
#              🔒 I TRIO NON SI TOCCANO in nessuna delle due modalita'.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI — `inner_val_r`, base = 0.0582:
#   ASSE 1a  --eeg_clean notch                        0.0582   Δ  0.0000  (esattamente zero)
#   ASSE 1b  --eeg_clean notch_ica                    0.0673   Δ +0.0091  (+16%)  <- L'UNICO
#   ASSE 2   --filters per_instrument                 0.1100   ⚠️ ARTEFATTO, vedi sotto
#   ASSE 3   --estimator shrinkage (lambda 0.1)       0.0349
#   ASSE 4   --target mag  (@64 Hz)                   0.0331   ·  @256 Hz  0.0344
#   ASSE 0   --n_mels 24 --target_fs 256              0.0468   (i due pezzi, separati:
#            24 bande @64 -> 0.0528 · 8 bande @256 -> 0.0492: peggiorano INDIPENDENTEMENTE)
#   combinazioni con ICA: hparams 0.0485 · per_instrument 0.1162 · shrinkage 0.0471
#
#   ⚠️ **ASSE 2 E' UN ARTEFATTO DELLA METRICA, NON UN GUADAGNO DEL DECODER.**
#      Con decoder per-strumento ogni sorgente e' ricostruita dal decoder GIUSTO, e
#      `inner_val_r` misura proprio quello. Sugli STESSI 70 segmenti held-out:
#      pooled 0.0428 vs own 0.0442, e il test equo da' **36/70, p = 0.29**.
#      E' la stessa scelta non dichiarata che aveva prodotto il 49/70 = 0.70 del
#      29/7 (spiegato e ritirato l'11/8): oggi `--own_vs_other` VIETA
#      `--filters per_instrument` con un assert, citando questo artefatto.
#
#   ⚠️ ASSE 0, l'unico numero di ACCURATEZZA dichiarato come sguardo: **0.4870**
#      (F1 micro/macro/weighted 0.4870/0.4899/0.4843), contro la base 0.4805.
#      Gli iperparametri PUBBLICATI (24 mel @ 256 Hz) **peggiorano**: non
#      recuperano il 78-79% del paper. Run pinnata: `madeeg_repro_hparams`.
#
# 🔑 IL CONTROLLO POSITIVO DELL'ICA, che e' la ragione per cui l'ASSE 1b si crede:
#    l'accoppiamento frontale-EOG in 1-8 Hz passa **0.552 -> 0.006**, e la
#    componente rimossa correla **r = 0.94** con l'EOG. Non e' «l'r e' salito»:
#    e' «la cosa che doveva sparire e' sparita».
#    ⚠️ TRAPPOLA PAGATA: il canale *chiamato* `ECG` e' tipizzato **MISC**; i
#    riferimenti veri sono **AUX1 (kind 402)** e **AUX3 (kind 202 = EOG)**.
#    Tipizzando per NOME l'ICA gira, dichiara successo e **non rimuove niente**.
#    L'ICA rimuove componenti solo in **5 soggetti su 8** (0004 e 0005 restano a 0).
#
# CANARINO   : ogni asse ha il DEFAULT sul comportamento preesistente ed e'
#              ASSERTITO (una combinazione sbagliata rompe invece di ricadere in
#              silenzio sul vecchio facendosi passare per il nuovo). I tre numeri
#              di riferimento del braccio — 0.5584 · --self_test PASS · 0.4805 —
#              devono restare bit-identici coi default DOPO qualunque modifica:
#              e' `ridge_anchors.sh` a verificarlo, e questo script lo richiama.
# CONTRATTO  : nessun contratto pre-registrato: e' un passo di configurazione
#              (per questo vive su `inner_val_r`). Nota di metodo nel vault:
#              "Cap. 2 — la banda 1-8 Hz è nostra: il candidato subito dopo il
#              testing (29 lug 2026)".
# PROVENIENZA: report "2026-07-29 (2) — riproduzione baseline: i quattro assi
#              misurati, e l'unico che si muove" ·
#              runs/results/ax_*/ e runs/results/madeeg_repro_hparams/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 DOSSIER §8.4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
RES="$REPO/runs/results"
INNER_ONLY=${INNER_ONLY:-0}                   # 0 = run PIENE: e' cosi' che sono stati prodotti
                                              #     i numeri pinnati, ed e' l'unico modo di
                                              #     riprodurli. Stampano anche l'accuratezza.
                                              # 1 = --inner_val_only: non costruisce nessun trial
                                              #     di test, quindi non PUO' spendere uno sguardo
                                              #     ⚠️ ma da' la media NON pesata (0.0562 sulla
                                              #     base, non 0.0582): NON e' il numero pinnato.

cd "$REPO"

# La base del protocollo del paper: train sui solo raw, test sui duo ricostruiti
# dal raw. Da qui si muove UN asse per volta.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --spatial stereo
      --ensemble duo --target mel --n_mels 8 --target_fs 64 --band_low 1 --band_high 8
      --lags_ms 250 --estimator ridge --filters pooled --eeg_clean none
      --seed 42 --cv_folds 5)
EXTRA=()
[ "$INNER_ONLY" = 1 ] && EXTRA=(--inner_val_only)

run() {  # run <nome> <flag dell'asse...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "${EXTRA[@]}" "$@" \
        --training_date "${TAG}${name}"
}
# run piene -> madeeg_summary.txt ; --inner_val_only -> madeeg_innerval_summary.txt
r() { grep -h "inner_val_r" "$RES/${TAG}$1"/*summary*.txt 2>/dev/null; }

if [ "$INNER_ONLY" = 1 ]; then
  cat <<'WARN'
⚠️ INNER_ONLY=1: nessun trial di test viene costruito, quindi nessuno sguardo puo'
   essere speso. IN CAMBIO il numero non e' quello pinnato: la media e' su
   (soggetto x fold) NON pesata, non sui trial. Base raw_solos: 0.0562, non 0.0582.
   Usalo per CONFIGURARE, e confronta inner_val_only con inner_val_only.
WARN
else
  cat <<'WARN'
⚠️ INNER_ONLY=0 (default): sono run PIENE — e' l'unico modo di riprodurre i numeri
   pinnati. Costruiscono i trial di test e stampano anche l'ACCURATEZZA sui duo.
   Leggerla e' UNO SGUARDO, e i 309 duo sono gia' tutti spesi: ogni numero che ne
   esce e' ESPLORATIVO PER COSTRUZIONE e va etichettato cosi' nel TITOLO della
   sezione, non in nota. 🔒 I TRIO NON SI TOCCANO.
WARN
fi

# ---- BASE + i cinque assi, uno per volta ------------------------------------
run ax_base                                                   # 0.0582
run ax_notch          --eeg_clean notch                       # ASSE 1a  0.0582
run ax_notch_ica      --eeg_clean notch_ica                   # ASSE 1b  0.0673  <- l'unico
run ax_per_instrument --filters per_instrument                # ASSE 2   0.1100  ⚠️ artefatto
run ax_shrinkage      --estimator shrinkage --shrinkage_lambda 0.1   # ASSE 3   0.0349
run ax_mag_fs64       --target mag                            # ASSE 4   0.0331
#   ^ --target mag ignora --n_mels: le bande sono fissate dalla geometria della STFT.
run ax_mag_fs256      --target mag --target_fs 256            #          0.0344
run ax_mel24_fs64     --n_mels 24                             # ASSE 0, pezzo 1  0.0528
run ax_mel8_fs256     --target_fs 256                         # ASSE 0, pezzo 2  0.0492
run ax_hparams        --n_mels 24 --target_fs 256             # ASSE 0 completo  0.0468

# ---- le tre combinazioni con l'ICA (l'unico asse che ha guadagnato) ---------
run ax_ica_hparams    --eeg_clean notch_ica --n_mels 24 --target_fs 256   # 0.0485
run ax_ica_perinstr   --eeg_clean notch_ica --filters per_instrument      # 0.1162 ⚠️
run ax_ica_shrink     --eeg_clean notch_ica --estimator shrinkage --shrinkage_lambda 0.1  # 0.0471

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in ax_base ax_notch ax_notch_ica ax_per_instrument ax_shrinkage ax_mag_fs64 \
         ax_mag_fs256 ax_mel24_fs64 ax_mel8_fs256 ax_hparams \
         ax_ica_hparams ax_ica_perinstr ax_ica_shrink; do
  printf "%-20s " "$R"; r "$R" || echo "(nessun summary)"
done
echo
echo "record pinnati (29/7): runs/results/ax_*/ e runs/results/madeeg_repro_hparams/"
echo "⚠️ ASSE 2 (0.1100 / 0.1162) e' un artefatto della METRICA, non del decoder:"
echo "   sugli stessi 70 segmenti held-out il test equo da' 36/70, p = 0.29."
echo
echo "Lo SGUARDO dichiarato dell'ASSE 0 (accuratezza 0.4870) sta in"
echo "runs/results/madeeg_repro_hparams/ — qui viene ri-prodotto da ${TAG}ax_hparams."
echo "Per configurare senza costruire nessun trial di test:"
echo "  INNER_ONLY=1 bash \$0     # <- media NON pesata: 0.0562 sulla base, non 0.0582"
