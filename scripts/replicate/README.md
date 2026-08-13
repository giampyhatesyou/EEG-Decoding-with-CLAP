# `scripts/replicate/` — uno script per esperimento

Uno script per ogni esperimento del Cap. 2, con **i comandi esatti** presi dai file
di provenienza e dai contratti pre-registrati nel vault. Ogni script porta in testa,
come commento: cosa misura in una frase, quanto ci mette, **se spende sguardi**, e
**i numeri di riferimento attesi** — così chi lo lancia sa subito se ha replicato.

> **Il Cap. 1 non è qui.** Sta in **`scripts/replicate.sh`** (già esistente, non
> toccato): riproduce tutti i numeri di Akama dai checkpoint rilasciati, ~45 min CPU,
> nessuna GPU. Prima serve `bash scripts/setup_checkpoints.sh`.
> Quello usa l'ambiente conda `attention` (torch); questa cartella usa miniconda base.

---

## Le tre trappole d'ambiente che fanno fallire tutto

Sono pagate, non ipotetiche ([[Legacy — la strada trovata e come proseguirla (11 ago 2026)]] §4).

1. **L'interprete: `/opt/miniconda3/bin/python`, mai `python` nudo.**
   Anaconda base (`/opt/anaconda3/bin/python`) dà gli stessi risultati con float
   diversi a 2e-5 → **i canarini md5 falliscono**. L'env conda `attention` non ha
   `h5py` e non può nemmeno eseguire `madeeg_reconstruction.py`.
   Tutti gli script qui usano `PY=${PY:-/opt/miniconda3/bin/python}`.
2. **`--spatial mono` richiede `--train_on raw_solos --test_eeg raw`** (assert nel
   codice: i duo mono non esistono nella release preprocessed). È la trappola che ha
   reso **ineseguibili** i secondari S1/S2 dell'Exp. 9 — dichiarati tali, non sostituiti.
3. **`--eeg_clean` vale solo con `--train_on raw_solos`** (assert: la release
   preprocessed è già pulita dagli autori).

Più due che costano tempo invece che correttezza: **16 GB di RAM** — le run a
`--target_fs 256` (Exp. 7 C5/C7, braccio A) sono ~6 GB di matrice di disegno l'una;
e **la shell dei task è zsh**, dove `$VAR` con più flag non fa word-splitting: per
questo qui ci sono script su file e array bash, non one-liner con variabili.

## Variabili comuni (default sensati, sovrascrivibili da env)

| variabile | default | cosa fa |
|---|---|---|
| `PY` | `/opt/miniconda3/bin/python` | l'interprete. Non cambiarlo senza rileggere la trappola 1 |
| `MADEEG_DIR` | `$HOME/madeeg` | il dataset MAD-EEG, **non è nel repo** (sorgente: `Tesi/MAD-MEG_dataset.zip`, 4.7 GB) |
| `OUT_DIR` | `<repo>/runs/replicate` | dove finiscono provenienza e confronti della replica |
| `TAG` | `repl_` | prefisso dei `--training_date`: **i record pinnati non si sovrascrivono** ([[Comandamenti]] §9) |

## La tabella

| script | cosa misura | costo | sguardi? | numero di riferimento | dove vive il numero |
|---|---|---|---|---|---|
| `exp07_ovo_config_sweep.sh` | 8 configurazioni del front end lineare contro il riferimento, su own-vs-other, McNemar appaiato α=0.00625 | ~15–40 min CPU (C5/C7 a 256 Hz: **da confermare**) | **NO** | REF **208/376 = 0.5532** · max C3 **213/376** · 🟡 nessuno passa | `runs/results/ovo_sweep_{REF,C1..C8}/` · `docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt` |
| `exp08_flux_ovo.sh` | il flusso spettrale come target di ricostruzione, su own-vs-other, α=0.025 | ~5 min CPU (stima) | **NO** | F1 **243/376 = 0.6463** (p 0.0015) · F2 **238/376** (p 0.0045) · 🟢 | `runs/results/ovo_flux_{F1,F2}/` · `docs/provenance/2026-08-11_exp8_flux_mcnemar.txt` |
| `exp09_flux_attention.sh` | se il guadagno flux si trasferisce alla **decisione di attenzione** sul duo | ~5 min CPU (**da confermare**) | **SÌ** (duo, già spesi) | P1 **76/154** · P2 **77/154** (mel: 74 · 86) · 🔴 | `runs/results/exp9_*/` · `docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt` |
| `exp11_spectral_register_stereo.sh` | il **registro** attenzionato da potenza Morlet + LDA, 47 coppie stereo, LOSO | ~3 min CPU (stima) | **SÌ** (duo, già spesi) | **24/47 = 0.5106**, soglia 30/47 non superata · 🔴 | `runs/results/exp11_*/` |
| `exp12_spectral_register_mono.sh` | lo stesso sui 42 duo **mono** ricostruiti dalla release raw, + stereo-da-raw e pooled 89 | ~3 min CPU (stima) | **SÌ** (duo, già spesi) | mono **22/42 = 0.5238** (soglia 27/42) · stereo-da-raw 24/47 · pooled 46/89 · 🔴 | `runs/results/exp12_*/` |
| `armA_paper_protocol.sh` | il pacchetto **completo del paper** (per-strumento, shrinkage 0.1, 24 mel @ 256 Hz, F1, trial interi) su tre bande | **da confermare** — la famiglia più pesante (3 run a 256 Hz) | **SÌ** (duo, già spesi) | F1 **0.5267 / 0.4800 / 0.5400** su n=150 · il 79 non si riproduce | `runs/results/armA_paper_{package,band40,band02_40}/` |
| `armD_leakage_audit.sh` | quanta accuratezza a finestre di 1 s è **impronta di trial** (audit del 92.6% di Niu) | ~2–3 min CPU (**da confermare**) | **SÌ** (duo, già spesi) | pseudo window-CV **0.6209** vs trial-CV **0.4847**, null 0.5136 · leakage dimostrato | `runs/results/armD_leakage_audit/leakage_audit_summary.txt` |
| `exp13_stem_separability.sh` | separabilità dei due stem **solo audio**, 6 rappresentazioni, contro un pavimento cross-brano | **~10 s CPU (misurato)** | **NO** — nessun EEG aperto | canarino mel-8 **0.1772/0.1442**, flux **0.2874/0.2417**, 30/36 · C3 MFCC-13 **0.0533** vs pav. 0.0256, 32/36 · GO | `docs/provenance/2026-08-12_exp13_stem_separability.txt` |
| `exp14_mfcc_tracking.sh` | se MFCC-13 e mel-64 sono **tracciate** dall'EEG (own-vs-other), barra ≥ 208/376 | ~3–4 min CPU (misurato dai timestamp) | **NO** | T1 **214/376 = 0.5691** · T2 **212/376** · entrambe tracciano | `docs/provenance/2026-08-12_exp14_mfcc_tracking.txt` · `runs/results/exp14_*/` |
| `exp15_mfcc_differential.sh` | il **differenziale attenzionale standardizzato** D sotto MFCC contro mel, permutazione appaiata | ~2–3 min CPU (misurato dai timestamp) | **SÌ** (duo, già spesi) | D(mel) **+0.1001** vs D(MFCC) **−0.1119**, p **0.9833**, MDD **0.2467** · accuratezza 65/154 vs 86/154 | `docs/provenance/2026-08-12_exp15_mfcc_differential.txt` · `runs/results/exp15_*/` |
| `exp16a_clap_separability.sh` | **CLAP** come settimo candidato del gate di separabilità, in due stadi (estrazione in `/opt/anaconda3`, misura in `/opt/miniconda3`) | ~9 min CPU **misurati** per l'estrazione + 11 s per la misura | **NO** — nessun EEG aperto | 🔴 **stadio 1 non eseguito**: il checkpoint (1.74 GiB) non è in cache e scaricarlo è decisione di A. · **controllo negativo già misurato: 512 dim di rumore soddisfano ENTRAMBI i criteri** (36/36, gap −0.0002, w/pav. **0.922** contro 3.131 di mel-8) | `docs/provenance/2026-08-12_exp16A_negative_control.txt` |
| `exp16b_ccaviews.sh` | se `band_power` (potenza log per canale, δ+θ) aggiunta alla CCA alza il tracciamento su own-vs-other, barra ≥ 208/376 | ~7 min CPU (misurato) | **NO** | **202/376 = 0.5372** (p 0.0819) contro ridge 208 e CCA 1-vista 209 · 🔴 **barra non superata** · McNemar non distingue in nessuna delle 4 direzioni | `docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt` · `runs/results/exp16b_*/` |
| `exp17_clap_separability.sh` | **CLAP** nel gate di separabilità, col criterio **rifatto invariante alla dimensionalità** (il vecchio è falsificato: il rumore-512 lo soddisfa). Estrazione ▶️ su baldo (CPU), misura sul Mac | 9 min 35 s estrazione (misurati, baldo CPU) + 24 s misura | **NO** — nessun EEG aperto | rumore-512 w/pav. **0.922** · mel-8 **3.131** · MFCC-13 **2.084** · **CLAP-512 3.837** → 🔴 **NO-GO**, criterio 2 non superato (e col criterio VECCHIO CLAP sarebbe passato) | `docs/provenance/2026-08-12_exp17_clap_separability.txt` · log integrale `..._exp17_baldo_extraction.log` · manifest `..._exp17_clap_extraction_manifest.json` |
| `exp19_matchmismatch.sh` | **preparazione** dell'Exp. 19: stesso compito dell'Exp. 18 con un **budget di epoche** (cap 120, arresto anticipato sulla **train** loss, pazienza 15) e il **null di S2 riparato**. Attraversa i quattro cancelli e **stampa i due comandi ▶️** (`nohup`, log in `runs/logs/`). ⚠️ **Non allena niente** | ~2 min 20 s CPU (misurati) | **NO** — solo i 105 trial `solo` | canarino Cap. 1 identico prima/dopo · campionamento S1 **invariato** (3346/3346, offset min 2.000 s, null 0.5000) e bilanciamento **no-op provato** su S1 · **null S2 held-out 0.6061 → 0.5000** (e 0.8898 → 0.5000 con la regola condizionata sul soggetto), costo 853 → 188 coppie · overfit **20/20 = 1.0000** in entrambi gli stadi, arresto anticipato a epoca 51 e 62 di 120 | `docs/provenance/2026-08-13_exp19_S1.txt` · `..._S2.txt` · `runs/results/madeeg_exp19_*` |
| `exp18_matchmismatch.sh` | **preparazione** dell'Exp. 18 (match-mismatch sui **solo**): canarino Cap. 1, regressione passo C, test del campionamento, sanità dell'ottimizzazione, poi **stampa il comando ▶️**. ⚠️ **Non allena niente senza un flag esplicito** | ~2 min 22 s CPU (misurati) | **NO** — solo i 105 trial `solo`, materiale di training gratuito | S1 3346 coppie, negativo dallo **stesso wav** 3346/3346, offset min **2.000 s**, stesso istante **0/3346**, null **0.5000** · S2 4414 coppie, strumento diverso 4414/4414, null **0.5063** (held-out **0.6061**, NON 0.500) · overfit **20/20 = 1.0000** in entrambi gli stadi | `docs/provenance/2026-08-12_exp18_matchmismatch.txt` · `runs/results/madeeg_exp18_*` |
| `run_all_cheap.sh` | in sequenza **solo** exp13 · exp07 · exp08 · exp14 | ~25–50 min CPU | **NO** | — | — |

**Come si legge la colonna «sguardi»:** *NO* = il registro degli sguardi dichiara
l'esperimento gratuito (solo audio, oppure own-vs-other su segmenti solo held-out).
*SÌ (duo, già spesi)* = decide sui duo. I 309 duo sono **tutti spesi**: rigirarli non
apre materiale nuovo, ma ogni numero sui duo è **esplorativo per costruzione** e va
etichettato così nel titolo. 🔒 **Nessuno script di questa cartella tocca i trio**
(92 stereo + 93 mono, l'unico holdout rimasto, un colpo solo, soglia stereo già
scritta 38/90).

## Regole che gli script rispettano, e che chi li modifica non rinegozia

- **Il canarino per primo.** Dove esiste, viene attraversato prima di qualunque
  numero vero e lo script si ferma se fallisce ([[Comandamenti]] §3). Exp. 7 e 8
  controllano il md5 `2eaa926244de340d31907c6deeb04b0c` del percorso di default;
  Exp. 9 e 15 controllano che le ancore mel siano **decision-identiche** trial per
  trial (74/154 e 86/154, zero discordanti); Exp. 11/12/13/14/15 hanno i cancelli
  dentro i driver, che escono con errore.
- **Un controllo non scrive sopra un risultato** ([[Comandamenti]] §9): i
  `--training_date` sono prefissati con `$TAG` e la provenienza va in `$OUT_DIR`.
  Due eccezioni dichiarate: i driver `madeeg_exp1{4,5}_*.py` non accettano
  `--log_dir` e riscrivono `runs/results/exp1{4,5}_*` — i file di provenienza
  pinnati restano intatti. È scritto in testa a quei due script.
- **Niente invocazioni plausibili** ([[Comandamenti]] §10): quello che non è
  ricostruibile con certezza è commentato con `# DA CONFERMARE:` e la ragione.
  Oggi c'è un solo caso: lo **split same/diff-melody del braccio A** — nessun file
  del repo lo produce, era ad hoc e lo script non è stato salvato.
- **Niente GPU** ([[Comandamenti]] §11). Nessuno script qui ne ha bisogno; il braccio C
  (match-mismatch su SparrKULee) è ▶️ di A. e i suoi comandi stanno nella Legacy §5.

## Cosa NON è coperto da questa cartella

- **Cap. 1** → `scripts/replicate.sh` (e `python src/run.py report` per la tabella).
- **Exp. 6 [H]** (alfa lateralizzata appaiata) → `runs/results/alpha_real_{stereo,mono}/`,
  script non scritto.
- **Passi C e D** (contrastivo CLAP su MAD-EEG) → GPU, `src/madeeg_contrastive.py`;
  i numeri vivono in `_baldo_archive_2026-07-18/`. La diagnostica del prior è
  `python src/madeeg_diagnose.py --records ...` → `docs/provenance/2026-08-11_diagnose_step{C,D}.txt`.
- **Exp. 10** — ritirato **prima** della run (insensibile per algebra). Il flag
  `--score_rule ortho` esiste ed è inerte: **non produrre numeri con esso**.
- **Exp. 16** — coperto: braccio B eseguito (`exp16b_ccaviews.sh`), braccio A fermo
  allo stadio 1 per un download di 1.74 GiB che è decisione di A.
  (`exp16a_clap_separability.sh`, che gira comunque il controllo negativo).
  ⚠️ **Il braccio A è stato poi eseguito dall'Exp. 17** (`exp17_clap_separability.sh`),
  che è il seguito da usare: stesso stadio 1, ma il criterio dell'Exp. 16A è
  **ritirato perché falsificato dal rumore-512** e sostituito da quello invariante
  alla dimensionalità. `exp16a_clap_separability.sh` resta agli atti, non si rilancia
  per giudicare CLAP.
- **Exp. 18** — solo la **preparazione** è coperta (`exp18_matchmismatch.sh`): il
  training è ▶️ di A. su baldo e lo script lo stampa senza lanciarlo. ⚠️ Questo è
  l'unico script della cartella che usa `/opt/anaconda3/bin/python`: gli serve
  `torch`, che in `/opt/miniconda3` non c'è. Non è la trappola n.1 — qui non c'è
  nessun cancello md5 da far combaciare.
- **Exp. 19** — idem, `exp19_matchmismatch.sh`, e usa lo stesso `/opt/anaconda3`.
  ⚠️ **Non sostituisce l'Exp. 18: lo estende.** Cambia il **budget di calcolo**
  (cap 120 epoche, arresto anticipato sulla **train** loss) e ripara il **null di
  S2** (`--balance_pairs`, held-out 0.6061 → 0.5000); la barra dell'held-out resta
  **0.70**, e lr/architettura/temperatura non si toccano. Entrambi i flag sono
  **spenti per default**, quindi `exp18_matchmismatch.sh` continua a riprodurre
  l'Exp. 18 cifra per cifra.
