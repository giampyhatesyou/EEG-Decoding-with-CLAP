#!/usr/bin/env bash
# =============================================================================
# Exp. 18 — match-mismatch sui SOLO di MAD-EEG: preparazione e cancelli.
#
# COSA MISURA: niente, di per se'. Questo script attraversa i CANCELLI CPU del
#   contratto (canarino del Cap. 1, test del campionamento, sanita' dell'ottimizzazione)
#   e poi STAMPA il comando ▶️ del training. Il training NON parte se non gli passi
#   un flag esplicito: e' un passo GPU e lo lancia A. ([[Comandamenti]] §11).
# COSTO      : ~3 min CPU misurati (2 x check_sampling ~16 s + 2 x overfit ~60 s).
# SGUARDI    : NO. Si aprono SOLO i trial `solo` (105 registrazioni, 420 ripetizioni),
#              che il registro degli sguardi dichiara materiale di training gratuito.
#              Nessun duo deciso, nessun trio, nessuno split per genere/melodia/
#              strumento/soggetto. 🔒 I TRIO NON SI TOCCANO.
# GPU        : solo il training (▶️, sotto). I cancelli qui girano su CPU.
#
# ⚠️ TRAPPOLA D'AMBIENTE, DIVERSA DA TUTTI GLI ALTRI SCRIPT DI QUESTA CARTELLA:
#    qui serve `torch`, che in `/opt/miniconda3` NON C'E'. Si usa `/opt/anaconda3`
#    (torch 2.11.0 + h5py + soundfile + sklearn). Non e' la trappola n.1 del Legacy §4:
#    quella riguarda i cancelli md5 di `madeeg_reconstruction.py`, e qui non ce ne sono —
#    nessun numero di questo script viene confrontato con un md5 pinnato.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   CANARINO Cap. 1 (`src/modules/clip_loss.py`, sintetico, torch.manual_seed(0)):
#     0.628491 · 1.2994 · 4.8198   e passo D  0.951610  vs  2.160834
#   CANCELLO regressione passo C (nessun torch, legge i record archiviati):
#     58/154 = 0.3766, e tutte e 154 le decisioni memorizzate
#   CENSIMENTO dei solo: 105 registrazioni · 420 ripetizioni · 39.4 min di EEG ·
#     96 wav distinti · 7 gruppi brano+tema
#   CAMPIONAMENTO S1 `temporal_offset`: 3346 coppie · negativo dallo STESSO wav
#     3346/3346 · offset minimo 2.000 s (soglia dichiarata 2.0 s) · stesso istante
#     0/3346 · NULL 0.5000 (miglior regola che ignora l'EEG) — esatto per costruzione
#   CAMPIONAMENTO S2 `cross_instrument`: 4414 coppie · stesso brano+tema e strumento
#     DIVERSO 4414/4414 · stesso istante sui due lati 4414/4414 · NULL 0.5063
#   SANITA' DELL'OTTIMIZZAZIONE (contratto §4.3), 20 esempi, CPU, `--clap_stub`:
#     S1 20/20 = 1.0000 · S2 20/20 = 1.0000 (entrambi entro ~10 epoche)
#
# ⚠️ `--clap_stub` sostituisce la torre CLAP CONGELATA con una proiezione casuale
#    fissa: il checkpoint da 1.74 GiB non e' sul Mac (scaricarlo e' decisione di A.,
#    Exp. 16A) e il Mac non ha nemmeno torchaudio. Ogni numero prodotto con quel
#    flag e' un controllo sul CABLAGGIO, mai un risultato — e la frase finisce
#    DENTRO il file di summary, non solo a terminale ([[Comandamenti]] §9).
#
# ⚠️ E IL NUMERO CHE A. DEVE GUARDARE PRIMA DI LANCIARE S2: il contratto §4.2 dichiara
#    "caso 0.500". Per S1 e' vero ed e' misurato (0.5000, esatto per costruzione: ogni
#    coppia di istanti sta nell'indice in ENTRAMBE le direzioni). Per S2 NON lo e':
#    i soggetti non hanno sentito tutti gli stessi solo, quindi la miglior regola che
#    ignora l'EEG vale 0.5063 sull'insieme intero e 0.6061 sulle 21 registrazioni
#    tenute fuori con seed 42. Contro 0.6061 la soglia 0.70 vale +0.094, non +0.20.
#    E' un fatto di metadati (zero sguardi spesi) e va deciso PRIMA della run.
#
# CONTRATTO  : vault, "Cap. 2 — Exp. 18: match-mismatch sui solo di MAD-EEG,
#              criterio pre-registrato (12 ago 2026)"
# PROVENIENZA: docs/provenance/2026-08-12_exp18_matchmismatch.txt (pinnata)
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/anaconda3/bin/python}           # vedi la trappola sopra: qui serve torch
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}
STEP_C_RECORDS=${STEP_C_RECORDS:-$HOME/Library/Mobile\ Documents/com~apple~CloudDocs/Tesi/_baldo_archive_2026-07-18/results/madeeg_clap_kfold/madeeg_contrastive_records.csv}

# ▶️ IL COMANDO PER A. SU BALDO — stampato, non eseguito.
# Ogni iperparametro e' quello del passo C/D (epochs 10, batch 8, lr 0.003, T 0.5,
# seed 42): NIENTE e' stato tarato per questo esperimento, cosi' l'unico fattore che
# cambia fra passo D ed Exp. 18 e' da dove viene il negativo.
BALDO_ENV=${BALDO_ENV:-/home/andrea.giampietro/.conda/envs/eeg_attention/bin/python}
BALDO_CKPT=${BALDO_CKPT:-/home/andrea.giampietro/clap_ckpt/630k-audioset-best.pt}
BALDO_MADEEG=${BALDO_MADEEG:-/home/andrea.giampietro/madeeg}

print_run_command () {
  cat <<EOF

===============================================================================
▶️  IL TRAINING — LO LANCIA A., SU baldo. [[Comandamenti]] §11.
===============================================================================
# (0) prerequisiti su baldo, da verificare UNA VOLTA prima di lanciare:
#     - la release RAW: madeeg_raw.hdf5, madeeg_raw.yaml, madeeg_sequences_raw.yaml
#       e la cartella stimuli/ (scompattata) dentro $BALDO_MADEEG.
#       ⚠️ Il passo C/D usava solo madeeg_preprocessed.*: la release preprocessed
#       NON contiene nessun solo, quindi questi quattro file servono davvero.
#     - il checkpoint CLAP gia' su disco (Exp. 17):
#       $BALDO_CKPT
#       sha256 8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037
#     - torchaudio nell'env (CLAPEncoder ricampiona 44.1 -> 48 kHz al suo interno).

# (1) PRE-VOLO, 2 minuti, obbligatorio: gli stessi cancelli girati qui ma con la
#     torre CLAP VERA. Se questo non fa 20/20, il cablaggio su baldo e' rotto e il
#     training vero non va lanciato (contratto §4.3).
cd ~/akami && git pull
$BALDO_ENV src/datasets/madeeg_solo_matchmismatch.py \\
    --madeeg_dir $BALDO_MADEEG --check_sampling
$BALDO_ENV src/madeeg_contrastive.py \\
    --madeeg_dir $BALDO_MADEEG --loss temporal_offset \\
    --clap_pretrained $BALDO_CKPT \\
    --overfit 20 --epochs 60 --batch_size 4

# (2) S1 — il negativo e' lo STESSO brano a un offset temporale dichiarato.
$BALDO_ENV src/madeeg_contrastive.py \\
    --madeeg_dir $BALDO_MADEEG --loss temporal_offset \\
    --clap_pretrained $BALDO_CKPT \\
    --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    2>&1 | tee ~/akami/runs/logs/exp18_S1_temporal_offset.log

# (3) S2 — SOLO SE S1 attraversa il cancello §4.2 (>= 0.70). Altrimenti il
#     contratto si ferma: nessun misuratore, nessuna affermazione.
$BALDO_ENV src/madeeg_contrastive.py \\
    --madeeg_dir $BALDO_MADEEG --loss cross_instrument \\
    --clap_pretrained $BALDO_CKPT \\
    --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \\
    2>&1 | tee ~/akami/runs/logs/exp18_S2_cross_instrument.log

# DOVE FINISCE TUTTO (nessun percorso da inventare: e' resolve_log_dir()):
#   ~/akami/runs/results/madeeg_exp18_temporal_offset/madeeg_matchmismatch_{records.csv,summary.txt}
#   ~/akami/runs/results/madeeg_exp18_cross_instrument/madeeg_matchmismatch_{records.csv,summary.txt}
#   ~/akami/runs/logs/exp18_S{1,2}_*.log
#   Nessun checkpoint viene salvato: il contratto chiede un'accuratezza held-out,
#   non un modello da riusare. Se A. vuole i pesi, va chiesto e aggiunto.
#
# QUANTO DURA — stima, con la sua ancora MISURATA e il suo "da confermare":
#   ancora misurata: l'estrazione dell'Exp. 17 su baldo ha fatto 8876 forward CLAP
#   in 496.6 s = 17.9 clip/s SU CPU (HTSAT-tiny, ingresso 10 s, batch 32).
#   Questa run chiede 2 candidati per item:
#     S1  (2810 train + 536 held-out) x 2 x 10 epoche = 66 920 forward CLAP
#     S2  (3561 train + 853 held-out) x 2 x 10 epoche = 88 280 forward CLAP
#   -> SU CPU: S1 ~62 min, S2 ~82 min (aritmetica sull'ancora misurata).
#   -> SU GPU: 🔶 DA CONFERMARE ALLA PRIMA RUN. Nessuna misura di CLAP su GPU esiste
#      in questo progetto. A 100-300 clip/s si va sui 4-11 min (S1) e 5-15 min (S2);
#      la costruzione del dataset (lettura del raw + band-pass + 96 wav) costa ~15 s
#      misurati sul Mac e va aggiunta una volta per run.
#   ⚠️ SE LA RUN E' PIU' LENTA DEL PREVISTO, la leva ovvia e' gratis e non e' stata
#      implementata di proposito: CLAP e' CONGELATO e le finestre audio distinte sono
#      ~400 in tutto, quindi i suoi embedding si calcolerebbero UNA VOLTA invece che a
#      ogni epoca. Non l'ho fatto perche' e' codice in piu' che il contratto non chiede;
#      va aggiunto solo se il tempo misurato lo giustifica.
===============================================================================
EOF
}

usage () {
  cat <<EOF
uso: bash scripts/replicate/exp18_matchmismatch.sh [--checks] [--train-s1] [--train-s2]

  (nessun flag)  attraversa i cancelli CPU e STAMPA il comando ▶️. Non allena niente.
  --checks       idem, esplicito.
  --train-s1     lancia S1 in locale. ⚠️ PASSO GPU: normalmente lo lancia A. su baldo.
  --train-s2     lancia S2 in locale. ⚠️ idem, e SOLO se S1 ha passato il cancello §4.2.
EOF
}

mkdir -p "$OUT_DIR"
cd "$REPO"
MODE=${1:---checks}

case "$MODE" in
  --checks)
    echo "=== CANCELLO 1 — canarino del Cap. 1 (sintetico, nessun dato) ==="
    "$PY" src/modules/clip_loss.py

    echo
    echo "=== CANCELLO 2 — regressione del passo C sui record archiviati ==="
    if [ -f "$STEP_C_RECORDS" ]; then
      /opt/miniconda3/bin/python src/madeeg_diagnose.py --check_rule "$STEP_C_RECORDS"
    else
      echo "  SALTATO: record del passo C non trovati in $STEP_C_RECORDS"
    fi

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== CANCELLO 3 — test del campionamento, $NEG ==="
      "$PY" src/datasets/madeeg_solo_matchmismatch.py \
          --madeeg_dir "$MADEEG_DIR" --negative "$NEG" --check_sampling
    done

    for NEG in temporal_offset cross_instrument; do
      echo
      echo "=== CANCELLO 4 — sanita' dell'ottimizzazione (contratto §4.3), $NEG ==="
      echo "    20 esempi, CPU, torre CLAP SOSTITUITA: e' un controllo, mai un risultato."
      "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
          --overfit 20 --clap_stub --epochs 60 --batch_size 4 \
          --training_date "${TAG}exp18_overfit_${NEG}" \
        | grep -E "^\[solo|^\[overfit|^\[Exp|one-sided|per held|pre-registered|written to"
    done

    print_run_command
    ;;
  --train-s1|--train-s2)
    NEG=temporal_offset; [ "$MODE" = "--train-s2" ] && NEG=cross_instrument
    echo "⚠️  Stai lanciando il TRAINING in locale ($NEG). E' un passo ▶️: su baldo lo"
    echo "    lancia A. Qui gira solo se hai CLAP e torchaudio in $PY."
    "$PY" src/madeeg_contrastive.py --madeeg_dir "$MADEEG_DIR" --loss "$NEG" \
        --epochs 10 --batch_size 8 --learning_rate 0.003 --temperature 0.5 --seed 42 \
        --training_date "${TAG}exp18_${NEG}"
    ;;
  *)
    usage; exit 2 ;;
esac
