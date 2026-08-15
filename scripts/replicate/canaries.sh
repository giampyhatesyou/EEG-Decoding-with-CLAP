#!/usr/bin/env bash
# =============================================================================
# canaries.sh — TUTTI i canarini del progetto, in un colpo solo, nell'ordine.
#
# COSA MISURA: niente di nuovo. Attraversa i cancelli di regressione che
#   proteggono i numeri gia' riportati e dice, per ciascuno, PASSATO / FALLITO /
#   NON ATTRAVERSABILE QUI (con la ragione). E' la prima cosa da lanciare dopo
#   ogni diff, e la prima cosa da riportare quando qualcosa non torna.
# DATA       : scritto il 15 ago 2026, dai canarini censiti nel DOSSIER §9.
# DOMANDA    : un numero gia' riportato si e' mosso?
# NULL       : n/a — nessuno di questi cancelli produce un risultato.
# SOGLIA     : ogni canarino porta la sua, letterale, qui sotto. Nessuna e'
#              negoziabile e nessuna si "aggiorna" perche' il codice e' cambiato:
#              se un canarino non passa, si FERMA e si riporta ([[Comandamenti]] §3/§8).
# VERDETTO   : lo stampa la run. Uscita != 0 se un canarino ATTRAVERSABILE fallisce.
# COSTO      : livello 0 ~10 s CPU · livello 1 ~5 s · livello 2 ~20-30 min CPU.
# SGUARDI    : NO. Nessun livello decide su un duo: i canarini del livello 2 sono
#              own-vs-other (segmenti SOLO held-out), controlli sintetici, e
#              ri-letture di record GIA' SPESI. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no. I canarini che richiedono GPU sono STAMPATI come ▶️, mai lanciati.
#
# I QUATTRO LIVELLI, e perche' sono separati (DOSSIER §8.4, trappole d'ambiente):
#
#   L0  nessun dato, nessun torch. Gira ovunque, anche in un clone pulito
#       (tranne i tre md5, che leggono artefatti sotto runs/, che e' gitignored).
#   L1  serve `torch` -> env conda `attention` sul Mac, `eeg_attention` su baldo.
#       E' il canarino del Cap. 1: `src/modules/clip_loss.py`.
#   L2  serve il dataset MAD-EEG in $MADEEG_DIR (NON e' nel repo, 4.7 GB).
#   L3  serve la GPU, o i checkpoint del paper: ▶️ li lancia A. ([[Comandamenti]] §11).
#
# NUMERI DI RIFERIMENTO ATTESI (DOSSIER §9). Se non escono ESATTAMENTE questi,
# non e' "quasi": e' un canarino rosso.
#
#   L0-a  src/run.py --selftest              -> "23 protocol flags ... PASS"
#   L0-b  src/madeeg_diagnose.py --demo      -> "[demo] PASS  null (base_rate) = 0.375"
#   L0-c  src/models/cca_multiview.py        -> "[cca_multiview] PASS rho_train=1.0000 ...",
#                                               piu' la prova che la view `position` e' INERTE
#   L0-d  toy dell'Exp. 10 (ritirato pre-run) -> 6 regimi (2 shared_w x 3 rumore).
#         ⚠️ VERIFICATO ESEGUENDOLO il 15/8: plain e ortho sono IDENTICI in 3 regimi
#         su 6 e distano <= 0.008 negli altri 3 (0.838/0.845 · 0.845/0.843 ·
#         0.688/0.680) — rumore di campionamento a 400 trial, non uguaglianza esatta.
#         E il toy NON asserisce niente: stampa e basta. Questo cancello verifica
#         quindi solo che giri e con che numeri, NON che i due siano identici.
#   L0-e  sweeps/report.py --check           -> 107 fold pinnati, tabella IDENTICA a
#                                               RESULTS.md e NIENTE scritto su disco
#   L0-f  md5 own-vs-other di default        -> 2eaa926244de340d31907c6deeb04b0c  (+ 208/376)
#   L0-g  md5 provenienza Exp. 13            -> a3c7533cb0d79ee849bb4c8c90ced803
#         md5 provenienza controllo negativo -> f3ebe89b900bde3240d35fb02084bc8f
#   L0-h  statistiche appaiate own-vs-other  -> REF n0/n1/n2 = 36/96/56, p esatto 0.0235
#                                               (e la tavola di soglie Bonferroni del contratto)
#   L1-a  src/modules/clip_loss.py           -> 0.628491 · 1.2994 · 4.8198
#                                               e passo D  0.951610  vs  2.160834
#   L2-a  --check_rule sui record del passo C -> 58/154 ESATTO, concorde trial per trial
#   L2-b  canarino Exp. 13 (solo audio)      -> mel-8 0.1772/0.1442 · flux 0.2874/0.2417 · 30/36
#   L2-c  own-vs-other REF rigirato          -> 208/376 = 0.5532 con md5 2eaa9262...
#   L2-d  ancore duo decision-identiche      -> canary_rawsolos 74/154 · canary_duo 86/154,
#                                               ZERO discordanti (non solo lo stesso totale)
#   L2-e  controllo positivo della ridge     -> --self_test [PASS] (soglia 0.90 nel codice)
#   L2-f  iniezione alfa lateralizzata       -> 44/44 nella configurazione esatta della run
#   L2-g  registro spettrale Exp. 11/12      -> 47/47 e 42/42 (hard-exit dentro il driver)
#   L2-h  cancello di allineamento Exp. 4    -> prestito degli stem r=1.0000 su 272 confronti,
#                                               155/155 wav<->mixture, span entro 4 ms
#   L3-a  Cap. 1 end-to-end da model-all0    -> GLOBAL 0.8650 ESATTO
#   L3-b  controlli sintetici passi C e D    -> 148/154 = 0.9610 · 142/154 = 0.9221 (soglia 0.90)
#   L3-c  campionamento S1 + overfit 20/20   -> vedi exp18/exp19_matchmismatch.sh
#
# ⚠️ TRAPPOLA n.1 (DOSSIER §8.4): l'interprete e' `/opt/miniconda3/bin/python`, MAI
#    `python` nudo. `/opt/anaconda3` da' gli stessi risultati con float diversi a 2e-5
#    e i canarini md5 FALLISCONO — e' gia' successo, all'inizio dell'Exp. 7.
#    L'env conda `attention` non ha `h5py` e non puo' eseguire `madeeg_reconstruction.py`:
#    per questo il livello 1 ha un interprete SUO ($PY_TORCH).
#
# CONTRATTO  : nessuno — non e' un esperimento. La regola che lo impone e'
#              [[Comandamenti]] §3 (controllo positivo attraversato PRIMA del numero
#              vero) e §8 (un flag nuovo non muove un numero vecchio).
# PROVENIENZA: DOSSIER §9 (tavola dei canarini) · docs/provenance/ (i file pinnati)
# =============================================================================
set -uo pipefail        # NON -e: i canarini si contano tutti, non ci si ferma al primo

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1: MAI `python` nudo
PY_TORCH=${PY_TORCH:-}                        # env con torch (conda `attention`); vuoto = cerca
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
ARCHIVE=${ARCHIVE:-$REPO/../_baldo_archive_2026-07-18/results}
LEVELS=${LEVELS:-"0 1 2"}                     # 3 non si esegue mai: si stampa

REF_MD5=2eaa926244de340d31907c6deeb04b0c
EXP13_MD5=a3c7533cb0d79ee849bb4c8c90ced803
NEGCTRL_MD5=f3ebe89b900bde3240d35fb02084bc8f

mkdir -p "$OUT_DIR"
cd "$REPO"

PASS=0; FAIL=0; SKIP=0
FAILED_NAMES=""
_md5()  { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
_has()  { case " $LEVELS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
ok()    { PASS=$((PASS+1)); echo "   ✅ PASSATO  — $1"; }
ko()    { FAIL=$((FAIL+1)); FAILED_NAMES="$FAILED_NAMES
     - $1"; echo "   ❌ FALLITO  — $1"; }
skip()  { SKIP=$((SKIP+1)); echo "   ⏭️  NON ATTRAVERSABILE QUI — $1"; }
head_() { echo; echo "--- $1"; }

# =============================================================================
# LIVELLO 0 — nessun dato, nessun torch.
# =============================================================================
if _has 0; then
echo "############################################################"
echo "# LIVELLO 0 — nessun dataset, nessun torch, nessuna GPU"
echo "############################################################"

head_ "L0-a  src/run.py --selftest — il protocollo Akama vive in 2 posti e devono coincidere"
if OUT=$("$PY" src/run.py --selftest 2>&1); then
  echo "$OUT" | tail -3
  echo "$OUT" | grep -q "PASS" && ok "L0-a run.py --selftest" || ko "L0-a run.py --selftest"
else
  echo "$OUT" | tail -5; ko "L0-a run.py --selftest (uscita != 0)"
fi

head_ "L0-b  madeeg_diagnose.py --demo — il null del prior NON e' 0.5, e il self-check lo prova"
if OUT=$("$PY" src/madeeg_diagnose.py --demo 2>&1); then
  echo "$OUT" | tail -3
  echo "$OUT" | grep -q "\[demo\] PASS" && ok "L0-b diagnose --demo" || ko "L0-b diagnose --demo"
else
  echo "$OUT" | tail -5; ko "L0-b diagnose --demo (uscita != 0)"
fi

head_ "L0-c  cca_multiview.py — la CCA ricostruisce, e la view \`position\` e' INERTE (|Δ| < 1e-9)"
if OUT=$("$PY" src/models/cca_multiview.py 2>&1); then
  echo "$OUT" | tail -5
  echo "$OUT" | grep -q "\[cca_multiview\] PASS" && ok "L0-c cca_multiview self-check" \
                                                 || ko "L0-c cca_multiview self-check"
else
  echo "$OUT" | tail -5; ko "L0-c cca_multiview self-check (uscita != 0)"
fi

head_ "L0-d  toy dell'Exp. 10 — la regola \`ortho\` e' INERTE per algebra (ritirata PRE-RUN)"
echo "      ⚠️ il flag --score_rule ortho esiste ed e' inerte: NON produrre numeri con esso."
if OUT=$("$PY" docs/provenance/2026-08-11_exp10_retired_toy_sweep.py 2>&1); then
  echo "$OUT" | tail -8
  ok "L0-d toy Exp. 10 (gira; 6 regimi, plain vs ortho entro 0.008 — il toy non asserisce)"
else
  echo "$OUT" | tail -5; ko "L0-d toy Exp. 10"
fi

head_ "L0-e  sweeps/report.py --check — 107 pin verificati contro gli hparams, NIENTE scritto"
BEFORE=$(_md5 RESULTS.md)
if OUT=$("$PY" sweeps/report.py --check 2>&1); then
  AFTER=$(_md5 RESULTS.md)
  echo "$OUT" | grep -E "^\| (madeeg_ridge|contrastive_raw|contrastive_clap) " | head -6
  if [ "$BEFORE" = "$AFTER" ]; then
    ok "L0-e report.py --check (pin verificati, RESULTS.md non toccato)"
  else
    ko "L0-e report.py --check HA SCRITTO su RESULTS.md — non doveva"
  fi
else
  echo "$OUT" | tail -8
  ko "L0-e report.py --check (un pin non torna, oppure i run dir non ci sono: vedi sopra)"
fi

head_ "L0-f  md5 del percorso own-vs-other di DEFAULT (artefatto gia' su disco)"
REF_CSV="$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv"
if [ -f "$REF_CSV" ]; then
  GOT=$(_md5 "$REF_CSV")
  echo "      atteso  $REF_MD5"
  echo "      trovato $GOT   ($REF_CSV)"
  [ "$GOT" = "$REF_MD5" ] && ok "L0-f md5 own-vs-other di default" \
                          || ko "L0-f md5 own-vs-other di default"
else
  skip "L0-f: runs/ e' gitignored e $REF_CSV non c'e'. In un clone pulito questo canarino
        si attraversa solo RIGIRANDO il riferimento -> livello 2 (L2-c)."
fi

head_ "L0-g  md5 dei due file di provenienza della separabilita' (Exp. 13 e controllo negativo)"
for PAIR in "docs/provenance/2026-08-12_exp13_stem_separability.txt:$EXP13_MD5" \
            "docs/provenance/2026-08-12_exp16A_negative_control.txt:$NEGCTRL_MD5"; do
  F=${PAIR%%:*}; WANT=${PAIR##*:}
  if [ -f "$F" ]; then
    GOT=$(_md5 "$F")
    printf "      %-58s %s\n" "$(basename "$F")" "$GOT"
    [ "$GOT" = "$WANT" ] && ok "L0-g md5 $(basename "$F")" || ko "L0-g md5 $(basename "$F") (atteso $WANT)"
  else
    skip "L0-g: manca $F"
  fi
done

head_ "L0-h  statistiche appaiate own-vs-other — permutazione esatta a coppie + soglie Bonferroni"
if [ -f "$REPO/runs/results/ovo_sweep_REF/madeeg_ownvsother.csv" ]; then
  if OUT=$("$PY" docs/provenance/2026-08-11_ovo_sweep_paired_stats.py \
             "$REPO/runs/results/ovo_sweep_REF/madeeg_ownvsother.csv" 2>&1); then
    echo "$OUT" | tail -4
    ok "L0-h paired stats (36/96/56, p esatto 0.0235, tavola di soglie del contratto)"
  else
    echo "$OUT" | tail -6; ko "L0-h paired stats"
  fi
else
  skip "L0-h: manca runs/results/ovo_sweep_REF/ (gitignored). Si rigenera col livello 2."
fi
fi   # livello 0

# =============================================================================
# LIVELLO 1 — serve torch. E' IL canarino del Cap. 1.
# =============================================================================
if _has 1; then
echo
echo "############################################################"
echo "# LIVELLO 1 — serve torch (env conda \`attention\`)"
echo "############################################################"

if [ -z "$PY_TORCH" ]; then
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "/opt/anaconda3/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi

head_ "L1-a  src/modules/clip_loss.py — l'aritmetica InfoNCE che regge OGNI numero del Cap. 1"
echo "      Self-check INTERAMENTE SINTETICO: torch.manual_seed(0), randn, nessun dataset,"
echo "      nessuna GPU, confrontato con l'aritmetica calcolata a mano DENTRO il file."
echo "      ⚠️ Il file e' fermo a bb016fd (28 lug 17:02) APPOSTA. Non si tocca."
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  echo "      interprete: $PY_TORCH"
  if OUT=$("$PY_TORCH" src/modules/clip_loss.py 2>&1); then
    echo "$OUT"
    N=$(echo "$OUT" | grep -c "PASS")
    if [ "$N" -ge 2 ] && echo "$OUT" | grep -q "0.628491" && echo "$OUT" | grep -q "0.951610"; then
      ok "L1-a clip_loss self-check (0.628491 · 1.2994 · 4.8198 · 0.951610 vs 2.160834)"
    else
      ko "L1-a clip_loss self-check: i valori stampati NON sono quelli pinnati nel DOSSIER §9"
    fi
  else
    echo "$OUT" | tail -10; ko "L1-a clip_loss self-check (uscita != 0)"
  fi
else
  skip "L1-a: nessun interprete con torch trovato. Passalo esplicitamente:
        PY_TORCH=~/miniconda3/envs/attention/bin/python bash $0
        (su baldo: \$HOME/.conda/envs/eeg_attention/bin/python)"
fi
fi   # livello 1

# =============================================================================
# LIVELLO 2 — serve il dataset MAD-EEG. ~20-30 min CPU.
# =============================================================================
if _has 2; then
echo
echo "############################################################"
echo "# LIVELLO 2 — serve MAD-EEG in $MADEEG_DIR"
echo "############################################################"

if [ ! -f "$MADEEG_DIR/madeeg_preprocessed.yaml" ]; then
  skip "TUTTO il livello 2: manca $MADEEG_DIR/madeeg_preprocessed.yaml.
        Il dataset NON e' nel repo (4.7 GB). Sorgente: Tesi/MAD-MEG_dataset.zip,
        oppure: MADEEG_DIR=... bash scripts/madeeg_setup.sh
        ⚠️ madeeg_setup.sh scarica SOLO la release preprocessed: il percorso
        raw_solos (own-vs-other, Exp. 4/6/9, braccio A) vuole ANCHE
        madeeg_raw.hdf5 + madeeg_raw.yaml + madeeg_sequences_raw.yaml + stimuli/."
else

head_ "L2-a  --check_rule — la regola di decisione del braccio contrastivo, sui record del passo C"
REC_C="$ARCHIVE/madeeg_clap_kfold/madeeg_contrastive_records.csv"
if [ -f "$REC_C" ]; then
  if OUT=$("$PY" src/madeeg_diagnose.py --check_rule "$REC_C" 2>&1); then
    echo "$OUT"
    echo "$OUT" | grep -q "58/154" && ok "L2-a --check_rule = 58/154 esatto" \
                                   || ko "L2-a --check_rule non da' 58/154"
  else
    echo "$OUT" | tail -6; ko "L2-a --check_rule"
  fi
else
  skip "L2-a: manca $REC_C (archivio di baldo fuori dal repo). ARCHIVE=... per puntarlo altrove."
fi

head_ "L2-b  canarino Exp. 13 — separabilita' solo audio (hard-exit dentro il driver)"
if OUT=$("$PY" src/madeeg_stem_separability.py --madeeg_dir "$MADEEG_DIR" \
           --out "$OUT_DIR/canary_exp13.txt" 2>&1); then
  grep -E "CANARY|PASSED|FAILED" "$OUT_DIR/canary_exp13.txt" | head -6
  grep -q "0.1772" "$OUT_DIR/canary_exp13.txt" && grep -q "0.2874" "$OUT_DIR/canary_exp13.txt" \
    && ok "L2-b canarino Exp. 13 (mel-8 0.1772/0.1442 · flux 0.2874/0.2417 · 30/36)" \
    || ko "L2-b canarino Exp. 13: i valori non sono quelli pinnati"
else
  echo "$OUT" | tail -8; ko "L2-b canarino Exp. 13 (il driver e' uscito con errore — e' il suo mestiere)"
fi

head_ "L2-c  own-vs-other REF rigirato da zero — 208/376 CON lo stesso md5"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --own_vs_other --eeg_clean notch_ica \
      --estimator ridge --filters pooled --target mel --n_mels 8 \
      --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 \
      --training_date "${TAG}canary_ovo_ref" >/dev/null 2>&1
NEW_CSV="$REPO/runs/results/${TAG}canary_ovo_ref/madeeg_ownvsother.csv"
if [ -f "$NEW_CSV" ]; then
  grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}canary_ovo_ref/madeeg_ownvsother_summary.txt"
  GOT=$(_md5 "$NEW_CSV")
  echo "      atteso  $REF_MD5"
  echo "      trovato $GOT"
  [ "$GOT" = "$REF_MD5" ] && ok "L2-c own-vs-other REF (208/376 + md5)" \
    || ko "L2-c own-vs-other REF: md5 diverso. PRIMA di tutto, controlla l'INTERPRETE (trappola n.1):
        /opt/anaconda3 da' le stesse 376 decisioni con float diversi a 2e-5 e questo md5 cambia."
else
  ko "L2-c own-vs-other REF: la run non ha prodotto il csv"
fi

head_ "L2-d  le due ancore mel della decisione duo — DECISION-IDENTICHE, non solo pari di totale"
DEC=(--madeeg_dir "$MADEEG_DIR" --estimator ridge --filters pooled --eeg_clean none
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 --cv_folds 5
     --ensemble duo --target mel --n_mels 8)
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" --train_on raw_solos --test_eeg raw --spatial stereo \
      --training_date "${TAG}canary_anchor_rawsolos" >/dev/null 2>&1
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" --train_on duos_kfold --spatial stereo \
      --training_date "${TAG}canary_anchor_duo" >/dev/null 2>&1
for T in "canary_rawsolos:${TAG}canary_anchor_rawsolos:74" "canary_duo:${TAG}canary_anchor_duo:86"; do
  OLD=${T%%:*}; REST=${T#*:}; NEW=${REST%%:*}; K=${REST##*:}
  A="$REPO/runs/results/$OLD/madeeg_records.csv"; B="$REPO/runs/results/$NEW/madeeg_records.csv"
  if [ -f "$A" ] && [ -f "$B" ]; then
    OUT=$("$PY" src/madeeg_diagnose.py --mcnemar "$A" "$B" 2>&1); echo "$OUT" | tail -4
    if echo "$OUT" | grep -q "no discordant comparisons" && echo "$OUT" | grep -q "A correct $K/154"; then
      ok "L2-d ancora $OLD: $K/154, ZERO discordanti"
    else
      ko "L2-d ancora $OLD: non e' decision-identica (o non da' $K/154)"
    fi
  else
    skip "L2-d ancora $OLD: manca $A (gitignored) — non c'e' niente contro cui appaiare."
  fi
done

head_ "L2-e  controllo positivo della ridge — EEG sintetico, soglia 0.90 DICHIARATA NEL CODICE"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
      --training_date "${TAG}canary_ridge" >/dev/null 2>&1
S="$REPO/runs/results/${TAG}canary_ridge_selftest/madeeg_selftest_summary.txt"
if [ -f "$S" ]; then
  grep -h "AAD accuracy" "$S"
  grep -q "\[PASS\]" "$S" && ok "L2-e --self_test della ridge" || ko "L2-e --self_test della ridge"
else
  ko "L2-e --self_test della ridge: nessun summary prodotto"
fi

head_ "L2-f  iniezione alfa lateralizzata — 44/44 nella configurazione ESATTA della run (notch+ICA)"
"$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --test_eeg raw --alpha_li --alpha_inject 1.0 \
      --eeg_clean notch_ica --spatial stereo \
      --training_date "${TAG}canary_alpha_ica" >/dev/null 2>&1
S="$REPO/runs/results/${TAG}canary_alpha_ica_inject/madeeg_alpha_summary.txt"
if [ -f "$S" ]; then
  grep -h "PAIRED ACCURACY" "$S"
  grep -q "44/44 = 1.0000" "$S" && ok "L2-f iniezione alfa 44/44" || ko "L2-f iniezione alfa != 44/44"
else
  ko "L2-f iniezione alfa: nessun summary prodotto"
fi

head_ "L2-g  registro spettrale Exp. 11 — 47/47 sull'iniezione, e lo script ESCE se non passa"
if OUT=$("$PY" src/madeeg_spectral_attention.py --madeeg_dir "$MADEEG_DIR" \
           --log_dir "$OUT_DIR" --seed 42 2>&1); then
  echo "$OUT" | grep -iE "control|47/47|primary" | head -5
  echo "$OUT" | grep -q "47/47" && ok "L2-g controllo positivo Exp. 11 = 47/47" \
                                || ko "L2-g controllo positivo Exp. 11"
else
  echo "$OUT" | tail -8
  ko "L2-g Exp. 11: il driver e' uscito con sys.exit(1) — il numero vero non e' stato calcolato"
fi

head_ "L2-h  cancello di allineamento dell'Exp. 4 — la premessa dell'intero esperimento mono"
echo "      ⚠️ Il 30/7 questo output NON e' stato salvato in nessun file (DOSSIER §9):"
echo "         qui viene REDIRETTO, che e' l'unica differenza."
if "$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" \
      --train_on raw_solos --test_eeg raw --check_alignment --spatial mono \
      > "$OUT_DIR/canary_exp4_alignment_mono.txt" 2>&1; then
  grep -iE "PASS|FAIL|1.0000|155|4 ms|borrow" "$OUT_DIR/canary_exp4_alignment_mono.txt" | head -8
  grep -q "PASS" "$OUT_DIR/canary_exp4_alignment_mono.txt" \
    && ok "L2-h --check_alignment --spatial mono" || ko "L2-h --check_alignment --spatial mono"
else
  tail -8 "$OUT_DIR/canary_exp4_alignment_mono.txt"; ko "L2-h --check_alignment (uscita != 0)"
fi
fi   # dataset presente
fi   # livello 2

# =============================================================================
# LIVELLO 3 — ▶️ NON SI ESEGUE QUI. Si stampa.
# =============================================================================
cat <<'EOF'

############################################################
# LIVELLO 3 — ▶️ LI LANCIA A. ([[Comandamenti]] §11)
############################################################

# L3-a  Il canarino end-to-end del Cap. 1: GLOBAL 0.8650 ESATTO da model-all0.ckpt.
#       ⚠️ Viveva in scripts/test_sanity.sh, CANCELLATO nel refactor 7fae3cf e ASSORBITO
#          dentro scripts/replicate.sh (fase `sanity`). Il DOSSIER §8.1/§9 cita ancora il
#          nome vecchio: il file NON esiste piu'. Il comando di oggi e':
bash scripts/setup_checkpoints.sh                       # una volta sola
PHASES="within sanity" bash scripts/replicate.sh        # ~25 min CPU, nessuna GPU
#       Deve dare: within 0.8650 esatto (deterministico) e i due controlli negativi
#       "al livello del caso" / "molto sotto lo 0.865". ⚠️ I due controlli NON sono
#       riproducibili bit-a-bit (checkpoint_test forza shuffle=True e --workers si
#       auto-dimensiona): NON citarli come cifre esatte.

# L3-b  I controlli sintetici del braccio contrastivo (GPU):
#       passo C  148/154 = 0.9610   ·  passo D  142/154 = 0.9221   (soglia 0.90 nel codice)
python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --self_test --kfold 5
python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --self_test --loss within_mixture --kfold 5

# L3-c  Campionamento match-mismatch e sanita' dell'ottimizzazione: hanno gia' il loro script.
bash scripts/replicate/exp18_matchmismatch.sh    # 3346 coppie, offset min 2.000 s, null 0.5000
bash scripts/replicate/exp19_matchmismatch.sh    # + null S2 0.6061 -> 0.5000, overfit 20/20

EOF

echo "############################################################"
echo "# ESITO: $PASS passati · $FAIL falliti · $SKIP non attraversabili qui"
echo "############################################################"
if [ "$FAIL" -gt 0 ]; then
  echo "CANARINI ROSSI:$FAILED_NAMES"
  echo
  echo "Non aggiornare il valore atteso. FERMATI e riporta ([[Comandamenti]] §3/§8):"
  echo "un canarino rosso vuol dire che un numero gia' riportato PUO' essersi mosso."
  exit 1
fi
echo "Nessun canarino rosso fra quelli attraversabili in questo ambiente."
echo "provenienza della replica: $OUT_DIR"
