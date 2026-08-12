#!/usr/bin/env bash
# =============================================================================
# Exp. 9 (ESPLORATIVO) — il front end flux sul compito di ATTENZIONE.
#
# COSA MISURA: se il guadagno del flux su own-vs-other (Exp. 8, +35/376) si
#   trasferisce alla decisione di attenzione sul duo, con McNemar appaiato contro
#   le ancore mel sugli stessi trial e alpha = 0.05/2 = 0.025.
# COSTO      : ~5 min CPU (stima: 4 run di decisione a 64 Hz; DA CONFERMARE — i
#              timestamp del 11/8 sono tutti uguali e non danno la durata).
# SGUARDI    : SI' — decide sui DUO. I 309 duo sono TUTTI GIA' SPESI (registro
#              degli sguardi §4.5): rigirarli non apre materiale nuovo, ma ogni
#              numero sui duo resta ESPLORATIVO PER COSTRUZIONE e va etichettato
#              cosi' NEL TITOLO. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   ancora mel P1 (raw_solos) 74/154 = 0.4805   ancora mel P2 (k-fold) 86/154 = 0.5584
#   P1 flux  76/154 = 0.4935   McNemar vs mel 33/64  p = 0.4503  -> NON passa
#   P2 flux  77/154 = 0.5000   McNemar vs mel 26/61  p = 0.9000  -> NON passa
#   ESITO: 🔴 il trasferimento tracking -> attenzione FALLISCE. Null 0.500 per
#   costruzione sul duo; barra assoluta a alpha 0.025 su n=154 = 90/154.
#
# CANARINO   : le due ancore mel sono rigirate PER PRIME e devono riprodurre le
#              DECISIONI PER-TRIAL dei canary pinnati (74/154 e 86/154 con ZERO
#              discordanti, non solo lo stesso totale). Lo script si ferma se no.
# CONTRATTO  : vault, "Cap. 2 — Exp. 9 (ESPLORATIVO): il front end flux sul
#              compito di attenzione, criterio pre-registrato (11 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt (pinnata) ·
#              runs/results/exp9_*/ · ancore runs/results/canary_{rawsolos,duo}/
#
# Il blocco AUDIO-ONLY dello stesso file di provenienza (corr. fra stem: mel
# 0.1772/0.1442, flux 0.2874/0.2417, flux piu' simile in 30/36 duo) NON si
# riproduce da qui: e' il canarino dell'Exp. 13, che lo ristampa a 4 decimali.
#   -> bash scripts/replicate/exp13_stem_separability.sh
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 Legacy §4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
RES="$REPO/runs/results"

mkdir -p "$OUT_DIR"
cd "$REPO"

# I due protocolli dell'Exp. 9. `eeg_clean none` ovunque, come le ancore.
# P1 = protocollo del paper (train sui solo, test sull'EEG raw).
# P2 = protocollo k-fold sui duo.
P1=(--train_on raw_solos --test_eeg raw --spatial stereo)
P2=(--train_on duos_kfold --spatial stereo)
DEC=(--madeeg_dir "$MADEEG_DIR" --estimator ridge --filters pooled --eeg_clean none
     --target_fs 64 --band_low 1 --band_high 8 --lags_ms 250 --seed 42 --cv_folds 5
     --ensemble duo)

rec() { echo "$RES/$1/madeeg_records.csv"; }
acc() { grep -h "OVERALL AAD accuracy" "$RES/$1/madeeg_summary.txt"; }

# ---- CANARINO: le ancore mel, decision-identiche, PRIMA di tutto ------------
echo "--- canarino: ancora mel P1 (protocollo paper) ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P1[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}exp9_gate_rawsolos_mel"
echo "--- canarino: ancora mel P2 (k-fold duo) ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P2[@]}" --target mel --n_mels 8 \
      --training_date "${TAG}exp9_gate_duo_mel"

GATE="$OUT_DIR/exp09_gate_anchors.txt"
: > "$GATE"
for P in "canary_rawsolos:${TAG}exp9_gate_rawsolos_mel:74" "canary_duo:${TAG}exp9_gate_duo_mel:86"; do
  OLD=${P%%:*}; REST=${P#*:}; NEW=${REST%%:*}; K=${REST##*:}
  OUT=$("$PY" src/madeeg_diagnose.py --mcnemar "$(rec "$OLD")" "$(rec "$NEW")")
  echo "$OUT" | tee -a "$GATE"
  # zero discordanti: madeeg_diagnose.py stampa "no discordant comparisons" e non
  # la riga dei discordanti (percorso verificato in src/madeeg_diagnose.py:404).
  if ! echo "$OUT" | grep -q "no discordant comparisons"; then
    echo "  -> [FALLITO] l'ancora $OLD non e' decision-identica. FERMATI e riporta."
    exit 1
  fi
  if ! echo "$OUT" | grep -q "A correct $K/154"; then
    echo "  -> [FALLITO] l'ancora $OLD non da' $K/154. FERMATI e riporta."
    exit 1
  fi
  echo "  -> [PASSATO] $OLD: $K/154, zero decisioni discordanti."
done

# ---- I DUE PRIMARI — K = 2, chiuso dal contratto ----------------------------
echo; echo "--- P1 : flux, protocollo paper, stereo ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P1[@]}" --target flux \
      --training_date "${TAG}exp9_P1_rawsolos_flux"
echo; echo "--- P2 : flux, k-fold duo, stereo ---"
"$PY" src/madeeg_reconstruction.py "${DEC[@]}" "${P2[@]}" --target flux \
      --training_date "${TAG}exp9_P2_duo_flux"

# I SECONDARI S1/S2 del contratto §1 (mel e flux k-fold sui duo MONO) sono
# INESEGUIBILI e sono dichiarati tali agli atti (Legacy §4.3): --spatial mono
# asserisce --train_on raw_solos --test_eeg raw, quindi un k-fold mono non
# esiste nel codice. NON si sostituiscono con un'altra configurazione.
#   "${DEC[@]}" "${P2[@]/--spatial stereo/--spatial mono}"   <-- rompe sull'assert

# ---- IL TEST APPAIATO --------------------------------------------------------
MC="$OUT_DIR/exp09_flux_attention_mcnemar.txt"
: > "$MC"
"$PY" src/madeeg_diagnose.py --mcnemar "$(rec canary_rawsolos)" "$(rec "${TAG}exp9_P1_rawsolos_flux")" | tee -a "$MC"
"$PY" src/madeeg_diagnose.py --mcnemar "$(rec canary_duo)"      "$(rec "${TAG}exp9_P2_duo_flux")"      | tee -a "$MC"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in "${TAG}exp9_gate_rawsolos_mel" "${TAG}exp9_gate_duo_mel" \
         "${TAG}exp9_P1_rawsolos_flux" "${TAG}exp9_P2_duo_flux"; do
  printf "%-34s " "$R"; acc "$R"
done
echo
echo "cancello della replica : $GATE"
echo "McNemar della replica  : $MC"
echo "provenienza pinnata    : docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt"
