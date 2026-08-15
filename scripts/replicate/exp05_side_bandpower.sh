#!/usr/bin/env bash
# =============================================================================
# Exp. 5 — il LATO attenzionato da band-power. ⚫ MAI GIRATO COME SCRITTO.
#
# Questo script NON esegue niente e non produce nessun numero. Esiste perche' il
# contratto dell'Exp. 5 e' agli atti, tre delle sue soglie sono ancora in uso, e
# una cartella di replica che semplicemente NON CONTIENE l'Exp. 5 farebbe pensare
# a una dimenticanza invece che a una decisione ([[Comandamenti]] §12: chi
# sbaglia ritira per iscritto, non cancella).
#
# COSA MISURAVA: se la potenza in banda (alfa in particolare) permette di
#   decodificare da quale LATO proveniva lo strumento attenzionato — e se questo
#   accade nella resa STEREO (±45°) e NON nella resa MONO.
# DATA       : contratto scritto il **9 ago 2026**, prima di qualunque riga di
#              codice e di qualunque numero. ARCHIVIATO l'11 ago 2026.
# ORIGINE    : ESTERNA, per iscritto, datata — la mail del Relatore del ~8/8/2026
#              («split up the EEG signal in different frequency bands, or better
#              even, use the frequency decomposition as the features for the
#              decoder»), con rimando a de Vries, Marinato & Baldauf (2021).
#              L'ipotesi non nasce dai nostri dati. Questo non la rende
#              confermativa (i duo erano gia' spesi), ma toglie il sospetto che
#              sia stata scelta dopo aver visto cosa conveniva.
# DOMANDA    : «il lato dello strumento attenzionato si legge dalla decomposizione
#              in frequenza?»
# NULL       : 0.5 teorico, ma il null OPERATIVO era **75/149 = 0.5034** — il
#              tiratore costante che risponde sempre «sinistra». E' quello il
#              pavimento, e andava stampato accanto alla statistica.
# SOGLIA PRE-REGISTRATA: **86/149 = 0.5772** sul primario (p = 0.0356; 85 avrebbe
#              dato 0.0505, quindi non c'era liberta' di scelta).
# VERDETTO   : ⚫ **ARCHIVIATO SENZA NUMERO.** La decisione sul LATO e' stata
#              eliminata il 10/8 e con essa la soglia 86/149. Nessuna run
#              corrisponde a questo contratto.
# SGUARDI    : zero. Non essendo mai stato eseguito, non ha speso niente.
# GPU        : n/a.
#
# 🔑 COSA SOPRAVVIVE, ED E' LA PARTE CHE CONTA — tre soglie scritte il 9/8, cioe'
#    DUE GIORNI PRIMA che esistesse qualunque numero, e poi riprese da altri
#    esperimenti. E' la prova materiale che quelle soglie non sono state scelte
#    dopo aver visto i dati:
#
#      30/47  ->  ripresa dall'**Exp. 11** (registro spettrale, duo stereo)
#      27/42  ->  ripresa dall'**Exp. 12** (duo mono) e dall'**Exp. 6 [H°]**
#      53/89  ->  ripresa dall'**Exp. 12** (pooled)
#      28/44  ->  la variante dell'**Exp. 6 [H]** dopo l'esclusione dei 3 target
#                 centrati (44 coppie a lati opposti invece di 47)
#
# 🔑 E LA SCOPERTA DI METADATI che ha reso possibile tutto il resto, contata il
#    9/8 e che NON spende sguardi (e' un conteggio di metadati, non di EEG):
#    `wav_info.panning` in `madeeg_preprocessed.yaml` da', per ogni trial e per
#    ogni strumento, **0.2 = sinistra · 0.8 = destra · 0.5 = centro**. Quindi:
#      149 duo stereo con lato del target definito  (75 L · 74 R)
#        5 duo stereo con target al CENTRO -> esclusi A PRIORI dal contratto
#          (tutti `pop_mixtape_duo_GtVx_theme1_stereo_Vx`: la voce e' centrata)
#       47 coppie gemelle stereo · di cui **44 a lati opposti**
#       42 coppie gemelle mono
#       22 mixture con la gemella in ENTRAMBE le rese
#
# ⚠️ IL CONFOUND DICHIARATO PRIMA, e che nessun successore ha rimosso: nei brani
#    **pop** (`BsDr`, `GtVx`) il panning e' fisso, quindi **lato ≡ strumento** e un
#    decoder del lato puo' essere un decoder dello strumento. Nei classici no:
#    10 combinazioni (brano, strumento) su 20 compaiono con lo strumento
#    attenzionato sia a sinistra sia a destra, per 79 trial — ⚠️ ma il ribaltamento
#    e' **fra soggetti, non dentro il soggetto** (0 soggetti hanno sentito lo
#    stesso strumento dai due lati), quindi quel contrasto e' BETWEEN-SUBJECT.
#
# ⚠️ NESSUN FILE DEL REPO IMPLEMENTA L'EXP. 5 COME SCRITTO. Verificato il 15/8:
#    non esiste un decoder del LATO da band-power. Le cose che gli somigliano, e
#    che NON sono la stessa cosa:
#      - `--alpha_li` (Exp. 6 [H]) decide il lato, ma con l'INDICE DI LATERALITA'
#        alfa su coppie gemelle, non con un decoder addestrato su band-power;
#      - `src/madeeg_spectral_attention.py` (Exp. 11/12) e' band-power + LDA, ma
#        decide il **REGISTRO** (acuto/grave), non il **LATO**;
#      - `--cca_views lateralization` (modello 9) e' una FEATURE, non un test, e
#        dentro 1-8 Hz e' identicamente zero (l'alfa e' fuori banda).
#    Scrivere «l'Exp. 5 e' stato fatto da X» sarebbe falso per tutti e tre.
#
# CONTRATTO  : vault, "90 Archivio/Contratti eseguiti/Cap. 2 — Exp. 5: lato
#              attenzionato da band-power, criterio pre-registrato (9 ago 2026)"
#              (archiviato l'11/8, NON riscritto)
# PROVENIENZA: nessuna — non esiste nessun file di risultato, e non deve esistere.
#              I successori: exp06_alpha_paired.sh · exp11_spectral_register_stereo.sh
#              · exp12_spectral_register_mono.sh
# =============================================================================
set -euo pipefail

cat <<'EOF'
⚫ Exp. 5 — ARCHIVIATO, MAI GIRATO COME SCRITTO. Questo script non esegue niente.

Non c'e' un comando da rilanciare perche' non c'e' codice che implementi il test
del contratto: la decisione sul LATO e' stata eliminata il 10/8 e con essa la
soglia 86/149. Il contratto resta agli atti, non riscritto.

Cosa lanciare INVECE, se quello che cerchi e' uno dei suoi successori:

  il LATO, su coppie gemelle, con l'indice di lateralita' alfa   (Exp. 6 [H])
    bash scripts/replicate/exp06_alpha_paired.sh
    -> primario 23/44 = 0.5227 (soglia 28/44) · controllo mono 24/42 (soglia 27/42)

  il REGISTRO (acuto/grave), band-power Morlet + LDA, duo STEREO  (Exp. 11)
    bash scripts/replicate/exp11_spectral_register_stereo.sh
    -> 24/47 = 0.5106 (soglia 30/47, scritta il 9/8)

  lo stesso sui duo MONO e sul pooled                             (Exp. 12)
    bash scripts/replicate/exp12_spectral_register_mono.sh
    -> mono 22/42 (soglia 27/42) · pooled 46/89 (soglia 53/89), entrambe del 9/8

  band_power come VISTA della CCA su own-vs-other                 (Exp. 16 braccio B)
    bash scripts/replicate/exp16b_ccaviews.sh
    -> 202/376 = 0.5372, barra 208/376 non superata
    ⚠️ e li' `band_power` dentro 1-8 Hz e' SOLO delta+theta: alfa e beta sono zero
       per costruzione e il modulo le scarta. Quel 🔴 vale per delta+theta, NON
       per l'idea del Relatore, che richiede una banda piu' larga — cioe' una
       PRE-REGISTRAZIONE NUOVA, non un flag in piu' su una run vecchia.
EOF
exit 0
