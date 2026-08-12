#!/usr/bin/env bash
# =============================================================================
# Exp. 7 — sweep di configurazione su own-vs-other (8 candidati vs riferimento).
#
# COSA MISURA: se una configurazione diversa del front end lineare (banda, bande
#   mel, frequenza di target, stimatore) batte il riferimento 208/376 sul compito
#   discriminativo own-vs-other, giudicato con McNemar appaiato sugli STESSI 376
#   confronti e alpha di Bonferroni 0.05/8 = 0.00625.
# COSTO      : ~15-40 min CPU. Le 7 run a 64 Hz sono ~1 min l'una (stima dai
#              timestamp delle run Exp. 14 del 12/8); C5 e C7 girano a
#              --target_fs 256 (~6 GB di matrice di disegno) e sono
#              sensibilmente piu' lente: durata DA CONFERMARE alla prima run.
#              ATTENZIONE 16 GB di RAM (Legacy §4.7): C5 e C7 sono gli ultimi
#              della sequenza apposta. Un candidato che non gira si dichiara
#              "non eseguito", non si sostituisce.
# SGUARDI    : NO — own-vs-other gira sui segmenti SOLO held-out (assert
#              strutturale --train_on raw_solos): nessun duo caricato, nessuna
#              decisione di attenzione, trio non toccati.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   REF  208/376 = 0.5532   md5(madeeg_ownvsother.csv) = 2eaa926244de340d31907c6deeb04b0c
#   C1 209 · C2 212 · C3 213 · C4 204 · C5 203 · C6 203 · C7 203 · C8 203  (su 376)
#   discordanti / vittorie del candidato: C1 23/45 · C2 28/52 · C3 33/61 ·
#   C4 14/32 · C5 34/73 · C6 21/47 · C7 35/75 · C8 33/71
#   ESITO: 🟡 NON CONCLUSIVO — nessun candidato passa Bonferroni (nemmeno 0.05).
#   Null 0.500 ESATTO per simmetria (ogni coppia valutata in entrambe le direzioni).
#
# CANARINO   : il riferimento e' rigirato PER PRIMO e deve dare 208/376 esatto con
#              md5 identico a runs/results/ovo_ridge_ica/. Se non lo da', lo script
#              si ferma e non calcola nessun candidato (Comandamenti §3).
# CONTRATTO  : vault, "Cap. 2 — Exp. 7: sweep di configurazione su own-vs-other,
#              criterio pre-registrato (11 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt (pinnata) ·
#              runs/results/ovo_sweep_{REF,C1..C8}/
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}      # dove finisce il McNemar della replica
TAG=${TAG:-repl_}                             # prefisso dei --training_date: i record
                                              # pinnati non si sovrascrivono (Comandamenti §9)
REF_CSV=${REF_CSV:-$REPO/runs/results/ovo_ridge_ica/madeeg_ownvsother.csv}
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

# Invarianti del contratto §2: NON si muovono. --filters e' imposto a pooled
# dall'assert di --own_vs_other; il seed deve essere identico in tutte le run o il
# test appaiato confronterebbe confronti diversi.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
      --eeg_clean notch_ica --filters pooled --target mel
      --band_low 1 --lags_ms 250 --seed 42)

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

run() {  # run <nome> <flag del candidato...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "$@" --training_date "${TAG}${name}"
}

acc()  { grep -h "own-vs-other accuracy" "$REPO/runs/results/${TAG}$1/madeeg_ownvsother_summary.txt"; }
csv()  { echo "$REPO/runs/results/${TAG}$1/madeeg_ownvsother.csv"; }

# ---- CANARINO DI REGRESSIONE, ATTRAVERSATO PER PRIMO (Comandamenti §3) -------
run ovo_sweep_REF --estimator ridge --n_mels 8 --target_fs 64 --band_high 8
GOT=$(_md5 "$(csv ovo_sweep_REF)")
echo; echo "=== CANARINO: il riferimento deve essere bit-per-bit quello pinnato ==="
acc ovo_sweep_REF
echo "  md5 ottenuto = $GOT"
echo "  md5 atteso   = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FALLITO] il percorso di default e' cambiato: NON si guarda nessun candidato."
  echo "     FERMATI e riporta (Legacy §4.8, Comandamenti §3/§8)."
  exit 1
fi
if [ -f "$REF_CSV" ] && ! cmp -s "$(csv ovo_sweep_REF)" "$REF_CSV"; then
  echo "  -> [FALLITO] il csv differisce da runs/results/ovo_ridge_ica/. FERMATI."
  exit 1
fi
echo "  -> [PASSATO] licenzia il cablaggio e nient'altro."

# ---- GLI 8 CANDIDATI — K = 8, chiuso dal contratto, non si allarga ----------
run ovo_sweep_C1 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 13
run ovo_sweep_C2 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 20
run ovo_sweep_C3 --estimator ridge     --n_mels  8 --target_fs  64 --band_high 40
run ovo_sweep_C4 --estimator ridge     --n_mels 24 --target_fs  64 --band_high  8
run ovo_sweep_C6 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels  8 --target_fs 64 --band_high  8
run ovo_sweep_C8 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels  8 --target_fs 64 --band_high 40
# C5 e C7 per ultimi: --target_fs 256, il caso da 16 GB registrato nel vault.
run ovo_sweep_C5 --estimator ridge     --n_mels  8 --target_fs 256 --band_high  8
run ovo_sweep_C7 --estimator shrinkage --shrinkage_lambda 0.1 --n_mels 24 --target_fs 256 --band_high 8

# ---- IL TEST APPAIATO — McNemar a una coda, candidato vs REF ----------------
MC="$OUT_DIR/exp07_ovo_sweep_mcnemar.txt"
: > "$MC"
for C in C1 C2 C3 C4 C5 C6 C7 C8; do
  "$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_sweep_REF)" "$(csv "ovo_sweep_$C")" | tee -a "$MC"
done

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for C in REF C1 C2 C3 C4 C5 C6 C7 C8; do printf "%-4s " "$C"; acc "ovo_sweep_$C"; done
echo
echo "McNemar della replica : $MC"
echo "provenienza pinnata   : docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt"
echo "NOTA: alpha = 0.05/8 = 0.00625. La soglia e' il minimo intero con binomiale"
echo "      esatta a una coda < alpha sui discordanti OSSERVATI: non ha gradi di liberta'."
