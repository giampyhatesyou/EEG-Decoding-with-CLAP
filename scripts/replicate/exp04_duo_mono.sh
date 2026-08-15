#!/usr/bin/env bash
# =============================================================================
# Exp. 4 — la meta' MONO dei duo, mai guardata prima. La run CONFERMATIVA.
#
# COSA MISURA: se il massimo delle 16 configurazioni degli assi (86/150 = 0.5733,
#   letto il 29/7 SENZA autorizzazione) replica su materiale mai visto — i 150
#   duo MONO, che esistono solo nella release raw e i cui stem vanno PRESI IN
#   PRESTITO dal gemello stereo dello stesso mixture.
# DATA       : 30 lug 2026. Contratto scritto la sera del 29/7, PRIMA della run.
# DOMANDA    : «il massimo delle 16 configurazioni replica sulla meta' mono?»
# NULL       : 0.500 — per costruzione sul duo (ogni mixture compare con entrambi
#              i suoi strumenti come target).
# SOGLIA PRE-REGISTRATA: **86/150 = 0.5733**, RICALCOLATA prima della run perche'
#              le due pre-registrate (89/155 e 87/151) non erano applicabili: i
#              trial decidibili sono 150, non 155. Minimo intero con binomiale
#              esatta a una coda p < 0.05 (86 -> p = 0.0430).
# VERDETTO   : 🔴 **NON REPLICA** — 79/150 = 0.5267, p = 0.2839, IC95 [0.444, 0.609].
# COSTO      : ~5-10 min CPU (stima: una run a 64 Hz con ICA su 8 soggetti +
#              il cancello di allineamento). DA CONFERMARE alla prima esecuzione.
# SGUARDI    : SI', ed e' **l'UNICO SGUARDO CONFERMATIVO** del progetto sui duo.
#              Con questa run i 309 duo sono diventati TUTTI SPESI (30/7): ogni
#              numero sui duo DOPO quella data e' esplorativo per costruzione.
#              🔒 I TRIO (92 stereo + 93 mono) NON SI TOCCANO: sono l'unico
#              holdout rimasto, un colpo solo, soglia stereo gia' scritta 38/90.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#   PRIMARIO   79/150 = 0.5267   p = 0.2839   IC95 [0.444, 0.609]
#              F1 micro/macro/weighted 0.5267/0.5225/0.5160
#              per-soggetto 0001=0.60 0002=0.70 0003=0.60 0004=0.55 0005=0.70
#                           0007=0.30 0008=0.45 0009=0.20   (range 0.20-0.70)
#              r(attended) 0.0348 vs r(best unattended) 0.0255 · inner_val_r 0.1162
#              n = 150 e non 155: il cancello di recupero dichiara **155 trial
#              costruiti, 150 decidibili**, e nei record `fold` = -1 su tutti e 150.
#              ⚠️ DA CONFERMARE la causa esatta dei 5 mancanti: sul braccio STEREO
#              i 4 mancanti (154 -> 150) sono documentati come effetto di
#              `--filters per_instrument` (il soggetto 0007 non ha i solo Bo/Fh);
#              per il mono il meccanismo e' plausibilmente lo stesso ma non l'ho
#              trovato scritto da nessuna parte. Il conteggio 155/150 SI' e' scritto.
#   CANCELLO   prestito degli stem r = **1.0000** su 272 confronti · wav<->mixture
#              155/155 · span entro **4 ms** · 0 onset mono coincidenti con gli stereo
#
# ⚠️ TRE COSE CHE VANNO SCRITTE OGNI VOLTA CHE SI CITA IL 79/150:
#   (a) **potenza 0.534** contro l'effetto motivante. 79/150 NON significa
#       «l'effetto non c'e'»: significa «a questa n non si vede». Le potenze
#       ricalcolate: 0.534 · 0.774 (contro 0.60) · 0.312 (contro 0.55).
#   (b) **la soglia 86/150 e' LO STESSO INTERO del risultato esplorativo** che
#       l'ha motivata (86/150 = 0.5733). Il criterio chiedeva di replicare una
#       quantita' distorta verso l'alto per costruzione (massimo su 16 confronti;
#       Bonferroni a 16 vorrebbe 93/150 = 0.62). Un criterio cosi' e' severo per
#       la ragione sbagliata, e va detto.
#   (c) **il confound non si e' attivato: il mono va PEGGIO.** Se la decisione
#       fosse spaziale, togliere la separazione dovrebbe far scendere il numero —
#       e infatti scende. Ma scende in un regime dove non c'e' segnale da
#       nessuna parte, quindi non e' evidenza di niente.
#
# ⚫ INCIDENTE DEL 30/7, agli atti ([[Comandamenti]] §12): nella stessa sessione e'
#    stato letto lo split PER GENERE (classica 47/90 = 0.522 · pop 32/60 = 0.533)
#    che il prompt VIETAVA. Non e' un risultato: e' materiale speso. Il contrasto
#    per genere sulla meta' mono e' BRUCIATO. Questo script NON lo ristampa.
#
# CANARINO   : il **cancello di allineamento** `--check_alignment --spatial mono`
#              e' la PREMESSA dell'intero esperimento — se gli stem presi in
#              prestito non sono quelli giusti, il resto non vuol dire niente.
#              Attraversato PER PRIMO qui, e ⚠️ **redirettato in un file**: il
#              30/7 il suo output non e' stato salvato da nessuna parte (DOSSIER
#              §9), quindi era stato verificato il CODICE del cancello, non i suoi
#              numeri. Rigirarlo costa un comando e non spende sguardi.
#              Secondo canarino: `--check_alignment --spatial stereo` FALLISCE ed
#              e' giusto cosi' — 8 trial di `pop_mixtape_BsDr_theme2` sono troncati
#              nella release. E' un difetto del DATASET, non del codice.
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — Exp. 4: criterio
#              pre-registrato sui duo MONO (29 lug 2026, sera)"
# PROVENIENZA: RESULTS.md §madeeg_duo <- results_manifest.tsv, pin
#              `madeeg_exp4_mono_confirm` · report "2026-07-30 — Exp. 4 mono" ·
#              verifica alla fonte "2026-07-30 (2)"
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 DOSSIER §8.4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # il record pinnato non si sovrascrive
RES="$REPO/runs/results"

mkdir -p "$OUT_DIR"
cd "$REPO"

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

# ⚠️ `--spatial mono` ASSERISCE `--train_on raw_solos --test_eeg raw`: i duo mono
#    non hanno EEG nella release preprocessed, non esistono proprio. L'assert c'e'
#    perche' una run che SEMBRA il braccio mono e invece e' quello stereo sarebbe
#    il fallimento peggiore possibile qui: i due bracci differiscono di UN trial.
RAW=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --ensemble duo)

# ---- CANARINO 1: il cancello di allineamento, PRIMA di tutto ----------------
echo "=== CANARINO 1 — cancello di allineamento sul braccio MONO ==="
echo "    (durate EEG/audio, onset dichiarati, e il PRESTITO degli stem validato"
echo "     contro la meta' stereo, dove il campo \`soli\` esiste davvero)"
GATE="$OUT_DIR/exp04_check_alignment_mono.txt"
if "$PY" src/madeeg_reconstruction.py "${RAW[@]}" --check_alignment --spatial mono \
     > "$GATE" 2>&1; then
  grep -iE "PASS|FAIL|1\.0000|155|ms|borrow|onset" "$GATE" | head -12
else
  tail -15 "$GATE"
  echo "  -> [FALLITO] il cancello non passa: il resto dell'esperimento non vuol dire niente."
  exit 1
fi
if ! grep -q "PASS" "$GATE"; then
  echo "  -> [FALLITO] nessun PASS nel cancello. FERMATI e riporta."
  exit 1
fi
echo "  -> [PASSATO] output completo in $GATE"
echo "     (il 30/7 questo file NON esisteva: e' l'unica differenza rispetto ad allora)"

# ---- CANARINO 2: lo STESSO cancello sul braccio stereo DEVE fallire ---------
echo
echo "=== CANARINO 2 — lo stesso cancello su STEREO deve FALLIRE (8 trial troncati) ==="
echo "    Non e' un bug del codice: e' un difetto della release su"
echo "    pop_mixtape_BsDr_theme2. Se qui passasse, il cancello non starebbe"
echo "    controllando niente."
GATE_S="$OUT_DIR/exp04_check_alignment_stereo.txt"
if "$PY" src/madeeg_reconstruction.py "${RAW[@]}" --check_alignment --spatial stereo \
     > "$GATE_S" 2>&1; then
  if grep -q "FAIL" "$GATE_S"; then
    grep -iE "FAIL|BsDr|truncat" "$GATE_S" | head -6
    echo "  -> [PASSATO] il cancello discrimina: fallisce dove deve."
  else
    echo "  -> ⚠️ ATTENZIONE: il cancello stereo NON riporta FAIL. Verifica a mano $GATE_S"
    echo "     prima di credere al braccio mono."
  fi
else
  grep -iE "FAIL|BsDr|truncat" "$GATE_S" | head -6
  echo "  -> [PASSATO] uscita != 0 sul braccio stereo: e' l'esito atteso."
fi

# ---- LA RUN CONFERMATIVA ----------------------------------------------------
# Ogni flag che cambia il numero, esplicito. Questa e' la configurazione VINCENTE
# delle 16 (ICA + per-strumento), applicata al materiale fresco: e' il senso della
# replica, e non e' negoziabile a posteriori.
echo
echo "=== LA RUN CONFERMATIVA — 150 duo MONO, mai guardati prima del 30/7 ==="
"$PY" src/madeeg_reconstruction.py "${RAW[@]}" --spatial mono \
      --eeg_clean notch_ica --filters per_instrument \
      --target mel --n_mels 8 --target_fs 64 \
      --band_low 1 --band_high 8 --lags_ms 250 \
      --estimator ridge --seed 42 --cv_folds 5 \
      --training_date "${TAG}exp4_mono_confirm"

# ---- CANARINO 3: il record pinnato deve riprodursi byte per byte ------------
echo
echo "=== CANARINO 3 — confronto col record pinnato madeeg_exp4_mono_confirm ==="
NEW="$RES/${TAG}exp4_mono_confirm/madeeg_records.csv"
OLD="$RES/madeeg_exp4_mono_confirm/madeeg_records.csv"
if [ -f "$OLD" ]; then
  A=$(_md5 "$OLD"); B=$(_md5 "$NEW")
  echo "  pinnato $A"
  echo "  replica $B"
  if [ "$A" = "$B" ]; then
    echo "  -> [PASSATO]"
  else
    echo "  -> [FALLITO] i record differiscono. Controlla PRIMA l'interprete (trappola n.1),"
    echo "     poi FERMATI e riporta ([[Comandamenti]] §3/§8)."
    exit 1
  fi
else
  echo "  ⏭️  il record pinnato non e' su questa macchina (runs/ e' gitignored)."
fi

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
sed -n '1,10p' "$RES/${TAG}exp4_mono_confirm/madeeg_summary.txt"
echo
echo "soglia pre-registrata: 86/150 = 0.5733 (p = 0.0430)  ->  79/150 NON la supera"
echo "record pinnato (30/7): runs/results/madeeg_exp4_mono_confirm/"
echo "⚠️ potenza 0.534: questo negativo NON e' informativo contro effetti piccoli."
