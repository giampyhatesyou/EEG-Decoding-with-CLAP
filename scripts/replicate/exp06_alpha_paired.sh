#!/usr/bin/env bash
# =============================================================================
# Exp. 6 [H] — L'ALFA LATERALIZZATA APPAIATA. Il vero test dell'intuizione del
#              Relatore. Piu' [H°], il controllo mono.
#
# COSA MISURA: se la lateralizzazione della potenza alfa predice il LATO dello
#   strumento attenzionato. Nessun decoder, nessun audio, nessun training: e' un
#   TEST DEI SEGNI sull'indice di lateralita' alfa (F3/F4, C3/C4, P3/P4, O1/O2),
#   valutato su COPPIE GEMELLE — stesso soggetto, stesso mixture, entrambi i
#   target — dove ogni costante di soggetto, canale, impedenza e sessione si
#   CANCELLA ALGEBRICAMENTE nella differenza.
# DATA       : 10 ago 2026.
# DOMANDA    : «la lateralizzazione alfa predice il LATO dello strumento attenzionato?»
# NULL       : **0.500 ESATTO PER SCAMBIABILITA'**, non stimato: sotto H0 i due
#              gemelli sono scambiabili (tutto identico tranne quale sorgente e'
#              attenzionata), quindi il segno della differenza dei loro LI e'
#              ugualmente probabile nei due versi.
# SOGLIA PRE-REGISTRATA: **28/44 = 0.6364** sul primario (p = 0.0481; 27 darebbe
#              0.0871, quindi nessuna liberta' di scelta) · **27/42 = 0.6429** sul
#              controllo mono (p = 0.0442; 26 darebbe 0.0821). Scritte il 9/8,
#              cioe' PRIMA che esistesse qualunque numero.
# VERDETTO   : 🔴 **SOTTO SOGLIA** — primario 23/44 = 0.5227, p esatta 0.4402.
#              🔴 controllo mono 24/42 = 0.5714, p 0.2204: sotto soglia anche lui,
#              ma **NUMERICAMENTE PIU' ALTO DEL PRIMARIO**.
# COSTO      : ~10-20 min CPU per le sei run (2 vere + 4 controlli). DA CONFERMARE.
# SGUARDI    : SI' — legge i duo. Tutti gia' spesi: ESPLORATIVO PER COSTRUZIONE, e
#              il caveat e' scritto DENTRO il summary della run. 🔒 TRIO non toccati.
# GPU        : no.
#
# NUMERI DI RIFERIMENTO ATTESI (se non escono questi, NON hai replicato):
#
#   PRIMARIO   stereo, coppie gemelle   **23/44 = 0.5227**  p 0.4402  soglia 28/44  🔴
#              per-soggetto 0001=0.50 0002=1.00 0003=0.50 0004=0.25 0005=0.50
#                           0007=0.20 0008=0.60 0009=0.50
#   CONTROLLO  mono, coppie gemelle     **24/42 = 0.5714**  p 0.2204  soglia 27/42  🔴
#   CONTROLLI POSITIVI (iniezione lateralizzata nell'EEG **REALE**, lato preso dal
#   panning VERO, stesso identico codice; soglia **0.90 dichiarata nel codice
#   prima della run**; ampiezza di riferimento 1.0 dichiarata prima, 0.5 e 2.0
#   come dose-risposta e NON come criterio):
#              a = 0.5   43/44 = 0.9773
#              a = 1.0   **44/44 = 1.0000**  <- IL CRITERIO
#              a = 2.0   44/44 = 1.0000
#              a = 1.0 nella configurazione ESATTA della run (notch+ICA)  **44/44**
#
#   ⚠️ n = 44 e non 47: le **3 coppie con target CENTRATO** (panning 0.5,
#      `pop_mixtape_duo_GtVx_theme1_stereo_Vx`) sono escluse **A PRIORI** dal
#      contratto del 9/8. Il codice di oggi le esclude, quindi una ri-esecuzione
#      riproduce i numeri a **n = 44** (`alpha_ctrl2_*`, `alpha_ctrl_ica_inject`).
#      Le directory `alpha_ctrl_a*_inject` sono di PRIMA dell'esclusione e
#      riportano **46/47 · 47/47 · 47/47** a n = 47. Non sono in conflitto con le
#      altre: sono due VINTAGE, e vanno citati come tali. Se una ri-esecuzione di
#      oggi ti da' 47 coppie, non e' il codice corrente.
#
# ⚠️ LA POTENZA, che e' quello che rende leggibile il negativo:
#    **0.371 contro un effetto di 0.60.** Questo negativo NON e' informativo
#    contro effetti piccoli. Dire «l'alfa non c'entra» sarebbe piu' di quanto il
#    dato sostenga; la frase difendibile e' «a n = 44 non si vede, con potenza 0.371».
#    ⚠️ Altrove nel vault la stessa potenza e' citata «contro 0.65»: DA PRECISARE
#    (DOSSIER §5.8). Il valore ricalcolato il 12/8 nell'audit delle soglie e' 0.371
#    **contro 0.60**.
#
# 🔴 IL CONTROLLO E' PIU' ALTO DEL PRIMARIO, E LE DUE PRE-REGISTRAZIONI NON
#    CONCORDANO SU COSA VOGLIA DIRE. Questo va risolto per iscritto, non scelto
#    in silenzio. I due testi, verbatim:
#      · contratto **Exp. 6 §2.4**: «Il mono deve stare AL CASO. **Se il mono
#        batte lo stereo, la lettura spaziale è FALSIFICATA** anche se lo stereo
#        passa» — condizione: mono > stereo. **Soddisfatta** (0.5714 > 0.5227).
#      · contratto **Exp. 5 §4(a)**: «Se il mono **supera la soglia** quanto o più
#        dello stereo, la lettura spaziale è FALSIFICATA anche se il primario
#        passa» — condizione: mono ≥ soglia. **NON soddisfatta** (24/42 < 27/42).
#    E il report del 10/8 §3.3 conclude che «la clausola di falsificazione formale
#    non si attiva — il primario non passa», mentre il DOSSIER §2.D registra
#    «falsificatore pre-scritto SCATTATO». **DA RISOLVERE DA A.**, citando quale
#    delle due formulazioni governa. La frase che regge in entrambi i rami:
#    *«non c'e' nulla, in questi numeri, che assomigli a un effetto spaziale»*.
#
# ⚠️ E MONO E STEREO NON SONO LO STESSO INSIEME DI COPPIE: lo stereo esclude le 3
#    coppie a target centrato (n = 44), il mono no (n = 42, e per costruzione in
#    mono il panning e' 0.5 ovunque). **NON e' un confronto appaiato** e non va
#    presentato come tale.
#
# 📌 IL PUNTO PIU' LARGO, che vale per tutto il Cap. 2: **l'alfa lateralizzata e'
#    un correlato TONICO, e la famiglia sicura decide per CORRELAZIONE
#    WITHIN-TRIAL** — che centra via qualunque costante di trial. Le due forme non
#    si incontrano senza cambiarne una. E' anche perche' la vista `position` della
#    CCA e' inerte (|Δ| < 1e-9, asserito nel codice).
#
# CANARINO   : il controllo positivo per iniezione gira PRIMA del numero vero e
#              nella CONFIGURAZIONE ESATTA della run (notch+ICA), non solo in una
#              configurazione comoda. Soglia 0.90 nel codice. I controlli scrivono
#              in directory separate (`<tag>_inject`) col caveat DENTRO il file,
#              sopra i numeri ([[Comandamenti]] §9).
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — Exp. 6: gate
#              own-vs-other e alfa appaiata, criterio pre-registrato (10 ago 2026)"
#              (le soglie vengono dal contratto Exp. 5 del 9/8)
# PROVENIENZA: runs/results/alpha_real_{stereo,mono}/madeeg_alpha_summary.txt ·
#              runs/results/alpha_ctrl2_a{0.5,1.0,2.0}_inject/ ·
#              runs/results/alpha_ctrl_ica_inject/ ·
#              report "2026-08-10 (2)" §3
#              ⚠️ tutte sotto runs/, che e' GITIGNORED: da un clone pulito questi
#              numeri non esistono finche' non si rigira questo script.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # trappola n.1 DOSSIER §8.4: MAI `python` nudo
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # il dataset NON e' nel repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # i record pinnati non si sovrascrivono
RES="$REPO/runs/results"

cd "$REPO"

# ⚠️ `--alpha_li` ASSERISCE `--train_on raw_solos`: legge il record RAW continuo.
#    Non allena niente — quel flag e' solo cio' che carica la release raw.
#    E ASSERISCE `--spatial != both`: il mono e' il braccio di CONTROLLO e
#    metterlo insieme al primario lo distruggerebbe.
BASE=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw
      --alpha_li --ensemble duo --seed 42)

pair() { grep -hE "PAIRED ACCURACY|threshold|twin pairs" \
         "$RES/${TAG}$1/madeeg_alpha_summary.txt" 2>/dev/null; }

# ---- I CONTROLLI POSITIVI, PRIMA DEL NUMERO VERO ---------------------------
# Iniezione lateralizzata di ampiezza nota nell'EEG REALE, lato dal panning VERO,
# STESSO codice. Se il cablaggio non ritrova un effetto che c'e' per costruzione,
# il numero vero non vuol dire niente.
echo "=== CONTROLLI POSITIVI — iniezione lateralizzata (soglia 0.90 nel codice) ==="
for A in 1.0 0.5 2.0; do
  echo; echo "--- iniezione a = $A (dose-risposta; 1.0 e' IL criterio) ---"
  # tag `ctrl2` come le directory pinnate del vintage CORRENTE (n = 44).
  # Le `alpha_ctrl_a*_inject` sono il vintage a n = 47, PRIMA dell'esclusione
  # dei 3 target centrati: non si sovrascrivono e non si confrontano.
  "$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
        --eeg_clean none --alpha_inject "$A" \
        --training_date "${TAG}alpha_ctrl2_a${A}"
  pair "alpha_ctrl2_a${A}_inject"
done

echo; echo "--- iniezione a = 1.0 nella CONFIGURAZIONE ESATTA DELLA RUN (notch+ICA) ---"
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
      --eeg_clean notch_ica --alpha_inject 1.0 \
      --training_date "${TAG}alpha_ctrl_ica"
S="$RES/${TAG}alpha_ctrl_ica_inject/madeeg_alpha_summary.txt"
pair "alpha_ctrl_ica_inject"
if ! grep -q "44/44 = 1.0000" "$S"; then
  echo "  -> [FALLITO] il controllo positivo nella configurazione della run non da' 44/44."
  echo "     Il numero vero NON si guarda ([[Comandamenti]] §3). FERMATI e riporta."
  exit 1
fi
echo "  -> [PASSATO] licenzia il cablaggio e nient'altro. Non dice niente sui dati veri."

# ---- IL PRIMARIO E IL SUO CONTROLLO ----------------------------------------
echo
echo "=== [H] PRIMARIO — coppie gemelle STEREO (⚠️ questo E' uno sguardo sui duo) ==="
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial stereo \
      --eeg_clean notch_ica --training_date "${TAG}alpha_real_stereo"

echo
echo "=== [H°] CONTROLLO — coppie gemelle MONO (deve restare AL CASO) ==="
"$PY" src/madeeg_reconstruction.py "${BASE[@]}" --spatial mono \
      --eeg_clean notch_ica --training_date "${TAG}alpha_real_mono"

echo
echo "=== ESITO DELLA REPLICA (confronta con i RIFERIMENTI in testa a questo script) ==="
echo "--- [H] PRIMARIO stereo ---";  pair "alpha_real_stereo"
echo "--- [H°] CONTROLLO mono ---";  pair "alpha_real_mono"
echo
echo "record pinnati (10/8): runs/results/alpha_real_{stereo,mono}/ (⚠️ gitignored)"
echo
echo "⚠️ potenza 0.371 contro un effetto di 0.60: questo negativo NON e' informativo"
echo "   contro effetti piccoli. E il controllo mono e' PIU' ALTO del primario:"
echo "   vedi in testa a questo script le DUE formulazioni pre-registrate che"
echo "   danno risposte diverse su cosa questo significhi. La voce e' APERTA."
echo "🚫 Il segno NON si gira ([[Comandamenti]] §6): la direzione fisiologica"
echo "   (desincronizzazione alfa CONTROLATERALE) e' fissata nel codice e nel"
echo "   summary PRIMA della run, e non e' stata toccata dopo."
