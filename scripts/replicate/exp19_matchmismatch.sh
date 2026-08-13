#!/usr/bin/env bash
# =============================================================================
# Exp. 19 — match-mismatch a convergenza (S1) e lo stadio strumento (S2, ESPLORATIVO).
#
# COSA MISURA: niente, di per se'. Attraversa i QUATTRO CANCELLI CPU del contratto
#   (canarino Cap. 1 · riproduzione del campionamento S1 · null di S2 RI-MISURATO dopo
#   il bilanciamento · sanita' dell'ottimizzazione) e poi STAMPA i due comandi ▶️.
#   Il training NON parte: e' un passo GPU e lo lancia A. ([[Comandamenti]] §11).
# COSTO      : ~4 min CPU misurati.
# SGUARDI    : NO. Solo i 105 trial `solo`. Nessun duo, nessun trio, nessuno split
#              per genere/melodia/strumento/soggetto. 🔒 I TRIO NON SI TOCCANO.
#
# COSA CAMBIA RISPETTO ALL'EXP. 18, e SOLO questo (contratto §0 e §5):
#   (1) il BUDGET DI CALCOLO: cap 120 epoche, arresto anticipato sulla TRAIN loss
#       (pazienza 15, delta minimo 0.002). La barra dell'held-out resta 0.70.
#       Cambia il budget, NON il criterio.
#   (2) la RIPARAZIONE DEL NULL di S2 (--balance_pairs), che non e' un iperparametro.
#   NIENTE ALTRO: learning rate 0.003, batch 8, temperatura 0.5, seed 42, architettura
#   ed encoder sono quelli del passo C/D e dell'Exp. 18. Una variabile per volta.
#
# ⚠️ TRAPPOLA D'AMBIENTE, la stessa dell'Exp. 18: qui serve `torch`, che in
#    `/opt/miniconda3` NON C'E'. Si usa `/opt/anaconda3` (torch 2.11.0).
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   CANCELLO 1 — canarino Cap. 1 (`src/modules/clip_loss.py`, sintetico, seed 0):
#     0.628491 · 1.2994 · 4.8198   e passo D  0.951610  vs  2.160834
#   CANCELLO 2 — campionamento S1 INVARIATO rispetto all'Exp. 18:
#     3346 coppie · negativo dallo STESSO wav 3346/3346 · offset minimo 2.000 s ·
#     stesso istante 0/3346 · NULL 0.5000 · e il bilanciamento e' un NO-OP
#     (3346 -> 3346, 2810 -> 2810, 536 -> 536, nulli tutti 0.5000)
#   CANCELLO 3 — null di S2 RI-MISURATO dopo il bilanciamento:
#     held-out  0.6061 -> 0.5000 (regola per identita' dei candidati)
#               0.8898 -> 0.5000 (stessa regola, condizionata anche sul SOGGETTO)
#     costo: held-out 853 -> 188 coppie (13 delle 21 registrazioni), train 3561 -> 2840
#   CANCELLO 4 — sanita' dell'ottimizzazione, 20 esempi, CPU, `--clap_stub`:
#     S1 20/20 = 1.0000 (arresto a epoca 51/120) · S2 20/20 = 1.0000 (epoca 62/120)
#
# ⚠️ `--clap_stub` sostituisce la torre CLAP CONGELATA con una proiezione casuale fissa
#    (checkpoint da 1.74 GiB assente sul Mac, e niente torchaudio). Ogni numero prodotto
#    con quel flag e' un controllo sul CABLAGGIO, mai un risultato — e la frase finisce
#    DENTRO il file di summary ([[Comandamenti]] §9).
#
# ⚠️ S2 E' ESPLORATIVO PER COSTRUZIONE (contratto §0): la decisione di eseguirlo e'
#    stata presa DOPO aver visto il numero di S1. Non puo' sostenere nessuna
#    affermazione confermativa. Il caveat e' anche dentro il summary della run.
#
# CONTRATTO  : vault, "Cap. 2 — Exp. 19: match-mismatch a convergenza (S1) e lo stadio
#              strumento (S2, ESPLORATIVO), criterio pre-registrato (13 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-13_exp19_S1.txt e _S2.txt
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/anaconda3/bin/python}           # vedi la trappola sopra: qui serve torch
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}

# ▶️ I COMANDI PER A., SU BALDO — stampati, non eseguiti.
# ⚠️ La directory del repo su baldo: l'Exp. 18 la documentava come ~/akami, il prompt
#    di questa sessione dice ~/EEG-Attention-decoding-with-CLAP. Qui si usa la seconda
#    ed e' una variabile: se il nome sul nodo GPU e' l'altro, si cambia QUESTA riga.
BALDO_REPO=${BALDO_REPO:-\$HOME/EEG-Attention-decoding-with-CLAP}
BALDO_ENV=${BALDO_ENV:-\$HOME/.conda/envs/eeg_attention/bin/python}
BALDO_CKPT=${BALDO_CKPT:-\$HOME/clap_ckpt/630k-audioset-best.pt}
BALDO_MADEEG=${BALDO_MADEEG:-\$HOME/madeeg}

print_run_command () {
  cat <<EOF

===============================================================================
▶️  IL TRAINING — LO LANCIA A., SUL NODO GPU (L40S). [[Comandamenti]] §11.
    Su baldo NON c'e' tmux: si usa nohup, cosi' la sessione del terminale web
    puo' cadere senza portarsi via la run.
===============================================================================
# (0) UNA VOLTA SOLA, prima di tutto:
REPO=$BALDO_REPO
PY=$BALDO_ENV
CKPT=$BALDO_CKPT
MAD=$BALDO_MADEEG

cd \$REPO && git pull && mkdir -p runs/logs

# (1) PRE-VOLO, ~3 min, OBBLIGATORIO (contratto §3.1-§3.4): gli stessi quattro
#     cancelli girati sul Mac, ma con la torre CLAP VERA. Se il passo 4 non fa
#     20/20, il cablaggio e' rotto e NIENTE va lanciato.
\$PY src/modules/clip_loss.py
\$PY src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir \$MAD \\
    --negative temporal_offset --check_sampling --check_null_by_split
\$PY src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir \$MAD \\
    --negative cross_instrument --check_sampling --check_null_by_split
\$PY src/madeeg_contrastive.py --madeeg_dir \$MAD --loss temporal_offset \\
    --clap_pretrained \$CKPT --overfit 20 --epochs 60 --batch_size 4

# =============================================================================
# ▶️ COMANDO 1 — S1 a convergenza. Copiare e incollare.
# =============================================================================
cd \$REPO && nohup \$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss temporal_offset --clap_pretrained \$CKPT \\
    --epochs 120 --early_stop_patience 15 --early_stop_min_delta 0.002 \\
    --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    > runs/logs/exp19_S1_temporal_offset.log 2>&1 &

# =============================================================================
# ▶️ COMANDO 2 — S2, lo stadio strumento, ESPLORATIVO, col null riparato.
#    Puo' partire subito dopo il primo o in parallelo: sono run indipendenti.
# =============================================================================
cd \$REPO && nohup \$PY src/madeeg_contrastive.py \\
    --madeeg_dir \$MAD --loss cross_instrument --clap_pretrained \$CKPT \\
    --balance_pairs \\
    --epochs 120 --early_stop_patience 15 --early_stop_min_delta 0.002 \\
    --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    > runs/logs/exp19_S2_cross_instrument.log 2>&1 &

# per seguirle:  tail -f \$REPO/runs/logs/exp19_S1_temporal_offset.log
#                tail -f \$REPO/runs/logs/exp19_S2_cross_instrument.log

# DOVE FINISCE TUTTO (nessun percorso inventato: e' resolve_log_dir()):
#   runs/results/madeeg_exp19_temporal_offset/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/results/madeeg_exp19_cross_instrument/madeeg_matchmismatch_{records.csv,summary.txt}
#   runs/logs/exp19_S{1,2}_*.log
#   ⚠️ exp19, NON exp18: la run nuova si scrive ACCANTO a quella vecchia, mai sopra
#      ([[Comandamenti]] §9). Nessun checkpoint viene salvato.
#
# QUANTO DURA — INFERITO, ancorato al dato MISURATO (40 s/epoca su 3346 coppie, L40S):
#   S1  3346 coppie/epoca -> 40 s  x 120 epoche = 80 min  (TETTO: l'arresto
#       anticipato puo' scattare molto prima)
#   S2  3028 coppie/epoca dopo il bilanciamento (2840 train + 188 held-out)
#       -> 40 x 3028/3346 = 36 s/epoca x 120 = 72 min (stesso tetto)
#   piu' ~15 s di costruzione del dataset per run (misurati sul Mac).
#   ⚠️ Se e' piu' lento del previsto la leva e' gratis e NON e' implementata di
#      proposito: CLAP e' CONGELATA e le finestre audio distinte sono ~400 in tutto,
#      quindi i suoi embedding si calcolerebbero UNA VOLTA invece che a ogni epoca.
#      E' codice che il contratto non chiede: si aggiunge solo se il tempo lo giustifica.
===============================================================================
EOF
}

usage () {
  cat <<EOF
uso: bash scripts/replicate/exp19_matchmismatch.sh [--checks]

  (nessun flag)  attraversa i quattro cancelli CPU e STAMPA i due comandi ▶️.
                 Non allena niente: il training e' un passo GPU e lo lancia A.
EOF
}

cd "$REPO"
case "${1:---checks}" in
  --checks)
    echo "=== CANCELLO 1 — canarino del Cap. 1 (sintetico, nessun dato) ==="
    "$PY" src/modules/clip_loss.py

    echo
    echo "=== CANCELLO 2 — campionamento S1 INVARIATO (la prova che il diff non l'ha toccato) ==="
    "$PY" src/datasets/madeeg_solo_matchmismatch.py \
        --madeeg_dir "$MADEEG_DIR" --negative temporal_offset \
        --check_sampling --check_null_by_split

    echo
    echo "=== CANCELLO 3 — null di S2 RI-MISURATO dopo il bilanciamento, PRIMA di ogni training ==="
    "$PY" src/datasets/madeeg_solo_matchmismatch.py \
        --madeeg_dir "$MADEEG_DIR" --negative cross_instrument \
        --check_sampling --check_null_by_split

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== CANCELLO 4 — sanita' dell'ottimizzazione, $NEG ==="
      echo "    20 esempi, CPU, torre CLAP SOSTITUITA: e' un controllo, mai un risultato."
      BAL=""; [ "$NEG" = "cross_instrument" ] && BAL="--balance_pairs"
      "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
          --overfit 20 --clap_stub --epochs 120 --batch_size 4 \
          --early_stop_patience 15 --early_stop_min_delta 0.002 $BAL \
          --training_date "${TAG}exp19_overfit_${NEG}" \
        | grep -E "^\[balance|^\[overfit|^\[null|^\[$NEG|^\[early|reported at|one-sided|per held|written to"
    done

    print_run_command
    ;;
  *)
    usage; exit 2 ;;
esac
