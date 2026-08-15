#!/usr/bin/env bash
# =============================================================================
# Exp. 6 [G] — il gate SOSTITUTIVO: own-vs-other, ridge contro CCA multi-view.
#              Le QUATTRO configurazioni, non solo il riferimento.
#
# COSA MISURA: su segmenti SOLO held-out, l'estimatore da' allo stesso EEG un
#   punteggio piu' alto contro il PROPRIO audio che contro quello di un altro
#   solo? E' DISCRIMINAZIONE — la quantita' di cui parla l'obiezione di
#   de Cheveigne — su materiale di TRAINING: nessun duo viene caricato e nessuna
#   decisione di attenzione viene presa.
# DATA       : 10 ago 2026. Sostituisce il gate del passo 1, dichiarato mal posto
#              nella pre-registrazione stessa (e il gate fallito resta agli atti).
# DOMANDA    : «la CCA multi-view batte la ridge su una metrica DISCRIMINATIVA?»
# NULL       : **0.500 ESATTO PER SIMMETRIA**, non stimato: le coppie sono mutue e
#              disgiunte, con strumento diverso, e ognuna e' valutata in ENTRAMBE
#              le direzioni. Un estimatore con nient'altro che una preferenza
#              fissa di strumento ne prende esattamente una delle due. n = 376.
# SOGLIA PRE-REGISTRATA: McNemar appaiato, minimo intero con binomiale esatta a
#              una coda < 0.05 sui discordanti OSSERVATI:
#                senza pulizia   80/139     ·   con notch+ICA   76/131
# VERDETTO   : 🔴 **NO-GO alla CCA.** Con notch+ICA: 66/131 discordanti a favore
#              della CCA contro 65 della ridge, **p = 0.5000**. Non e' «la CCA
#              perde»: e' **«la CCA non aggiunge niente»**.
# COSTO      : ~10-15 min CPU per le quattro run (stima: ~1 min l'una a 64 Hz per
#              la ridge, la CCA e' piu' lenta). DA CONFERMARE alla prima esecuzione.
# SGUARDI    : **NO** — own-vs-other gira sui segmenti SOLO held-out (l'assert
#              strutturale impone `--train_on raw_solos`): nessun duo caricato,
#              nessuna decisione di attenzione. 🔒 I TRIO NON SI TOCCANO.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#
#   configurazione        ridge              CCA                McNemar appaiato
#   base (nessuna pulizia) 202/376 = 0.5372  207/376 = 0.5505   72/139  p 0.3673  soglia 80/139
#   notch + ICA            208/376 = 0.5532  209/376 = 0.5559   66/131  p 0.5000  soglia 76/131
#
#   p binomiali a una coda: 0.0819 · 0.0281 · 0.0221 · 0.0172
#   p ESATTI a coppie (permutazione T = n1 + 2·Binom(n0+n2, ½), che e' la forma
#   giusta perche' le 376 decisioni sono 188 coppie, non 376 unita' indipendenti):
#                          0.0851 · 0.0265 · 0.0235 · 0.0230
#   ⚠️ **in tesi si cita il p ESATTO.** La preoccupazione sulla dipendenza era
#      fondata in linea di principio e trascurabile nei fatti: nessun verdetto cambia.
#   intersezione dei confronti fra i due estimatori: **376/376** (la verifica
#   `--mcnemar`), quindi i due sono confrontati sugli STESSI confronti.
#
# 🔑 IL RAMO 🟡 SI ATTIVA DOVE SERVE, ED E' LA PARTE INFORMATIVA:
#    - senza pulizia la **ridge stessa e' al caso** (0.5372, p = 0.0819 > 0.05):
#      li' il test NON HA SENSIBILITA' e non dice niente su nessuno dei due;
#    - con notch+ICA la ridge e' sopra il caso (0.5532, p = 0.0221): li' la
#      sensibilita' e' DIMOSTRATA, e proprio li' la CCA non la batte.
#    Il NO-GO e' quindi pronunciato dove significa qualcosa.
#    ⚠️ **La configurazione di preprocessing NON era pre-registrata.** Sono state
#    girate entrambe e il NO-GO e' stato pronunciato su quella CON sensibilita' —
#    una scelta SFAVOREVOLE all'ipotesi di chi la faceva, ma presa DOPO. Va
#    dichiarata cosi', non nascosta.
#
# 🔑 IL FATTO SUI DATI, che vale piu' del verdetto:
#    **su materiale di training, in-distribution, ENTRAMBI i metodi discriminano
#    appena sopra il caso: ~0.55 contro 0.500, con n = 376.** Non e' un problema
#    di stimatore: e' quanto poco tracciamento decodificabile c'e' in questi 20
#    canali. E' cio' che rende leggibile tutto il resto del Cap. 2.
#
# ⚠️ UN PRECEDENTE NON RIPRODOTTO, e va detto: il **49/70 = 0.70** (p 6.0e-04) del
#    29/7 girava con `--filters per_instrument` e una definizione di *other* che
#    NON e' conservata nel repo. Con decoder per-strumento l'audio PROPRIO viene
#    ricostruito dal decoder GIUSTO e quello *other* da uno SBAGLIATO: e' una
#    scelta metodologica non dichiarata, la stessa che aveva prodotto l'artefatto
#    0.1100 dell'ASSE 2. Oggi `--own_vs_other` **VIETA `--filters per_instrument`
#    con un assert**. Quei 70 confronti e questi 376 non sono lo stesso numero e
#    non vanno in tabella insieme.
#
# CANARINO   : la configurazione **ridge + notch_ica** e' il riferimento di
#              default dell'intero progetto: **208/376 + md5
#              `2eaa926244de340d31907c6deeb04b0c`**. E' la base di Exp. 7, 8, 14,
#              16B e 18. Viene rigirata PER PRIMA e lo script si ferma se l'md5
#              non torna. In piu': `--mcnemar` verifica da solo che i due CSV
#              condividano 376/376 confronti.
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — Exp. 6: gate
#              own-vs-other e alfa appaiata, criterio pre-registrato (10 ago 2026)"
# PROVENIENZA: runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/ ·
#              report "2026-08-10 (2) — il gate mal posto sostituito, e l'alfa
#              appaiata: due negativi con potenza dichiarata" §2
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 DOSSIER §8.4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_DIR=${OUT_DIR:-$REPO/runs/replicate}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
RES="$REPO/runs/results"
REF_MD5=2eaa926244de340d31907c6deeb04b0c

mkdir -p "$OUT_DIR"
cd "$REPO"

_md5() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
csv()  { echo "$RES/${TAG}$1/madeeg_ownvsother.csv"; }
acc()  { grep -h "own-vs-other accuracy" "$RES/${TAG}$1/madeeg_ownvsother_summary.txt"; }

# Gli invarianti: identici in tutte e quattro. `--filters pooled` e' IMPOSTO
# dall'assert di --own_vs_other (vedi il precedente 49/70 qui sopra); il seed
# dev'essere lo stesso o il test appaiato confronterebbe confronti diversi.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --own_vs_other
      --filters pooled --target mel --n_mels 8 --target_fs 64
      --band_low 1 --band_high 8 --lags_ms 250 --seed 42)

run() {  # run <nome> <flag della configurazione...>
  local name="$1"; shift
  echo; echo "--- $name : $* ---"
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" "$@" --training_date "${TAG}${name}"
}

# ---- CANARINO: il riferimento di default, PRIMA di tutto -------------------
run ovo_ridge_ica --estimator ridge --eeg_clean notch_ica
GOT=$(_md5 "$(csv ovo_ridge_ica)")
echo; echo "=== CANARINO — ridge+ICA e' il riferimento di TUTTO il progetto ==="
acc ovo_ridge_ica
echo "  md5 ottenuto = $GOT"
echo "  md5 atteso   = $REF_MD5"
if [ "$GOT" != "$REF_MD5" ]; then
  echo "  -> [FALLITO] il percorso di default e' cambiato."
  echo "     Controlla PRIMA l'interprete: /opt/anaconda3 da' le stesse 376 decisioni"
  echo "     con float diversi a 2e-5 e questo md5 cambia (e' gia' successo, Exp. 7)."
  echo "     Poi FERMATI e riporta ([[Comandamenti]] §3/§8)."
  exit 1
fi
echo "  -> [PASSATO] licenzia il cablaggio e nient'altro."

# ---- LE ALTRE TRE CONFIGURAZIONI --------------------------------------------
# Ogni default della CCA riproduce ESATTAMENTE il percorso dati della ridge (una
# vista EEG = la sua stessa matrice di disegno, una vista stimolo = il suo stesso
# bersaglio), quindi `--estimator cca` senza altri flag e' il confronto alla pari.
run ovo_cca_ica --estimator cca --eeg_clean notch_ica
run ovo_ridge   --estimator ridge --eeg_clean none
run ovo_cca     --estimator cca   --eeg_clean none

# ---- IL TEST APPAIATO — McNemar, dentro ogni configurazione -----------------
# ⚠️ Si appaia ridge contro CCA DENTRO la stessa configurazione di preprocessing.
#    Appaiare ridge-base contro CCA-ICA confonderebbe due assi in un test solo.
MC="$OUT_DIR/exp06_ovo_gate_mcnemar.txt"
: > "$MC"
echo; echo "--- McNemar, configurazione notch+ICA (quella con sensibilita' dimostrata) ---"
"$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_ridge_ica)" "$(csv ovo_cca_ica)" | tee -a "$MC"
echo; echo "--- McNemar, configurazione base (⚠️ qui la ridge stessa e' al caso) ---"
"$PY" src/madeeg_diagnose.py --mcnemar "$(csv ovo_ridge)" "$(csv ovo_cca)" | tee -a "$MC"

# ---- I p ESATTI A COPPIE — quelli che vanno in tesi -------------------------
echo
echo "=== p ESATTI a coppie (188 coppie, non 376 unita' indipendenti) ==="
"$PY" docs/provenance/2026-08-11_ovo_sweep_paired_stats.py \
      "$(csv ovo_ridge_ica)" "$(csv ovo_cca_ica)" "$(csv ovo_ridge)" "$(csv ovo_cca)" \
  | tee "$OUT_DIR/exp06_ovo_gate_paired_stats.txt"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
for R in ovo_ridge ovo_cca ovo_ridge_ica ovo_cca_ica; do printf "%-14s " "$R"; acc "$R"; done
echo
echo "McNemar della replica : $MC"
echo "p esatti della replica: $OUT_DIR/exp06_ovo_gate_paired_stats.txt"
echo "record pinnati (10/8) : runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/"
echo
echo "⚠️ Il verdetto e' 'la CCA non AGGIUNGE niente', non 'la CCA perde'."
echo "⚠️ E il fatto sui dati e' che ENTRAMBI stanno a ~0.55 contro un null di 0.500."
