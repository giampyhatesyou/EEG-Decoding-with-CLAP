#!/usr/bin/env bash
# =============================================================================
# Exp. 8 — il target di flusso spettrale su own-vs-other.
#
# COSA MISURA: se cambiare la RAPPRESENTAZIONE dello stimolo (log-mel -> onset
#   strength / flusso spettrale per banda) rompe il tetto di own-vs-other, con
#   McNemar appaiato contro il riferimento mel-8 e alpha = 0.05/2 = 0.025.
# COSTO      : ~5 min CPU (stima: 2 self-test + 3 run own-vs-other a 64 Hz,
#              ~1 min l'una dai timestamp delle run Exp. 14 del 12/8).
# SGUARDI    : NO — own-vs-other sui segmenti SOLO held-out (assert strutturale
#              --train_on raw_solos). Nessun duo, nessun trio.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   REF  (mel-8)      208/376 = 0.5532   md5 2eaa926244de340d31907c6deeb04b0c
#   F1   --target flux      243/376 = 0.6463   McNemar vs REF 84/133  p = 0.0015
#   F2   --target flux_mel  238/376 = 0.6330   McNemar vs REF 77/124  p = 0.0045
#   ESITO: 🟢 ENTRAMBI passano Bonferroni (alpha 0.025) — primo verde
#   pre-registrato del progetto. Null 0.500 ESATTO per simmetria.
#   Controlli positivi (EEG sintetico, soglia 0.90 nel codice): 1.0000 entrambi.
#
# CANARINI   : (1) il riferimento mel-8 e' rigirato PER PRIMO e deve dare 208/376
#              con md5 identico a runs/results/ovo_ridge_ica/;
#              (2) i due controlli positivi --self_test si attraversano PRIMA dei
#              numeri veri e lo script si ferma se uno non stampa [PASS].
# CONTRATTO  : vault, "Cap. 2 — Exp. 8: il target di flusso spettrale su
#              own-vs-other, criterio pre-registrato (11 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-11_exp8_flux_mcnemar.txt (pinnata) ·
#              runs/results/{flux_ctrl,fluxmel_ctrl}_selftest, ovo_flux_{regcheck,F1,F2}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
REF_CSV=${REF_CSV:-$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv}
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

# Il riferimento del contratto §2: ogni altro flag resta questo, cambia SOLO --target.
OVO=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
     --eeg_clean notch_ica --estimator ridge --filters pooled
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42)

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
csv()  { echo "$REPO/runs/results/${TAG}$1/madeeg_ownvsother.csv"; }
acc()  { grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}$1/madeeg_ownvsother_summary.txt"; }

# ---- CANARINO 1: il percorso di default, PRIMA di tutto (Comandamenti §3/§8) --
echo "--- canarino di regressione: il riferimento mel-8 ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}ovo_flux_regcheck"
GOT=$(_md5 "$(csv ovo_flux_regcheck)")
acc ovo_flux_regcheck
echo "  md5 ottenuto = $GOT"
echo "  md5 atteso   = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FALLITO] il default e' cambiato: nessun candidato viene girato. FERMATI."
  exit 1
fi
echo "  -> [PASSATO]"

# ---- CANARINO 2: i controlli positivi, PRIMA dei numeri veri -----------------
# EEG sintetico costruito DALLA rappresentazione candidata; soglia 0.90 dichiarata
# nel codice. Scrivono in <tag>_selftest, directory separate (Comandamenti §9).
for T in flux flux_mel; do
  NAME=$([ "$T" = flux ] && echo flux_ctrl || echo fluxmel_ctrl)
  echo; echo "--- controllo positivo --self_test --target $T ---"
  "$PY" src/madeeg_reconstruction.py --madeeg_dir "$MADEEG_DIR" --self_test \
        --target "$T" --training_date "${TAG}${NAME}"
  S="$REPO/runs/results/${TAG}${NAME}_selftest/madeeg_selftest_summary.txt"
  grep -h "AAD accuracy" "$S"
  if ! grep -q "\[PASS\]" "$S"; then
    echo "  -> [FALLITO] controllo positivo non superato: il numero vero NON si guarda."
    exit 1
  fi
  echo "  -> [PASSATO] licenzia il cablaggio e nient'altro."
done

# ---- I DUE CANDIDATI — K = 2, chiuso dal contratto --------------------------
# F1: flux e' a 1 banda, --n_mels non si passa (il contratto lo dice esplicitamente).
echo; echo "--- F1 : --target flux ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target flux \
      --training_date "${TAG}ovo_flux_F1"
echo; echo "--- F2 : --target flux_mel (8 bande, capacity-matched al riferimento) ---"
"$PY" src/madeeg_reconstruction.py "${OVO[@]}" --target flux_mel --n_mels 8 \
      --training_date "${TAG}ovo_flux_F2"

# ---- IL TEST APPAIATO --------------------------------------------------------
MC="$OUT_DIR/exp08_flux_mcnemar.txt"
: > "$MC"
for F in F1 F2; do
  "$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_flux_regcheck)" "$(csv "ovo_flux_$F")" | tee -a "$MC"
done

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in ovo_flux_regcheck ovo_flux_F1 ovo_flux_F2; do printf "%-18s " "$R"; acc "$R"; done
echo
echo "McNemar della replica : $MC"
echo "provenienza pinnata   : docs/provenance/2026-08-11_exp8_flux_mcnemar.txt"
echo "NOTA: alpha = 0.05/2 = 0.025. La soglia e' il minimo intero con binomiale esatta"
echo "      a una coda < alpha sui discordanti OSSERVATI."
