# `scripts/replicate/` — uno script per esperimento

Uno script per ogni esperimento del progetto, con **i comandi esatti** presi dai file
di provenienza e dai contratti pre-registrati nel vault. Ogni script porta in testa,
come commento: cosa misura in una frase, quanto ci mette, **se spende sguardi**, e
**i numeri di riferimento attesi** — così chi lo lancia sa subito se ha replicato.
La tabella qui sotto è solo la navigazione: **il dettaglio sta nell'intestazione di
ogni script**, che è la fonte, non un riassunto.

> **Aggiornato il 15 ago 2026:** il Cap. 1 ora è coperto anche qui
> (`cap1_*.sh`, che avvolgono `scripts/replicate.sh`), e sono stati aggiunti gli
> esperimenti che il DOSSIER §8.3 elencava come «senza script di replica»:
> ancore ridge, i quattro assi, Exp. 4, Exp. 5 (archiviato), Exp. 6 [G] e [H],
> passi C e D, più `canaries.sh` che attraversa **tutti** i cancelli in un colpo.

---

## Le tre trappole d'ambiente che fanno fallire tutto

Sono pagate, non ipotetiche (DOSSIER §8.4).

1. **L'interprete: `/opt/miniconda3/bin/python`, mai `python` nudo.**
   Anaconda base (`/opt/anaconda3/bin/python`) dà gli stessi risultati con float
   diversi a 2e-5 → **i canarini md5 falliscono** (è successo davvero, all'inizio
   dell'Exp. 7). L'env conda `attention` non ha `h5py` e non può nemmeno eseguire
   `madeeg_reconstruction.py`.
   Tutti gli script qui usano `PY=${PY:-/opt/miniconda3/bin/python}`.
   **Eccezione dichiarata:** `exp18`/`exp19_matchmismatch.sh` usano `/opt/anaconda3`
   perché serve `torch` — e lì non c'è nessun cancello md5 da far combaciare.
   Gli script che attraversano il canarino del Cap. 1 hanno un **secondo**
   interprete, `$PY_TORCH`, che cerca l'env conda `attention` (torch 2.2.2,
   numpy<2) e ricade su `/opt/anaconda3` solo come ultima risorsa.
2. **`--spatial mono` richiede `--train_on raw_solos --test_eeg raw`** (assert nel
   codice: i duo mono non esistono nella release preprocessed). È la trappola che ha
   reso **ineseguibili** i secondari S1/S2 dell'Exp. 9 — dichiarati tali, non sostituiti.
3. **`--eeg_clean` vale solo con `--train_on raw_solos`** (assert: la release
   preprocessed è già pulita dagli autori).

Più due che costano tempo invece che correttezza: **16 GB di RAM** — le run a
`--target_fs 256` (Exp. 7 C5/C7, braccio A, ASSE 0) sono ~6 GB di matrice di disegno
l'una; e **la shell dei task è zsh**, dove `$VAR` con più flag non fa word-splitting:
per questo qui ci sono script su file e array bash, non one-liner con variabili.

## Variabili comuni (default sensati, sovrascrivibili da env)

| variabile | default | cosa fa |
|---|---|---|
| `PY` | `/opt/miniconda3/bin/python` | l'interprete. Non cambiarlo senza rileggere la trappola 1 |
| `PY_TORCH` | *cercato* | l'env con `torch` per il canarino `clip_loss.py`. Vuoto = lo cerca; se non lo trova **dichiara che non l'ha attraversato**, non finge |
| `MADEEG_DIR` | `$HOME/madeeg` | il dataset MAD-EEG, **non è nel repo** (sorgente: `Tesi/MAD-MEG_dataset.zip`, 4.7 GB) |
| `OUT_DIR` | `<repo>/runs/replicate` | dove finiscono provenienza e confronti della replica |
| `TAG` | `repl_` | prefisso dei `--training_date`: **i record pinnati non si sovrascrivono** ([[Comandamenti]] §9) |
| `ARCHIVE` | `<repo>/../_baldo_archive_2026-07-18/results` | i record dei passi C/D, che vivono fuori dal repo |
| `LEVELS` | `0 1 2` | solo in `canaries.sh`: quali livelli di canarino attraversare |

---

## La tavola di navigazione

`null` = il valore contro cui si giudica. `soglia` = il criterio **scritto prima**
della run. `CPU?` = ✅ gira su CPU · **+MAD** serve il dataset · **+CKPT** servono i
checkpoint del paper · ▶️ è un passo GPU che **lancia A.** ([[Comandamenti]] §11).

### Cap. 1 — il braccio Akama

| esperimento | script | null | soglia | verdetto | contratto nel vault | provenienza | canarino | CPU? |
|---|---|---|---|---|---|---|---|---|
| **within-split** + controlli negativi | `cap1_within_split.sh` | 0.25 (4 slot fissi, assunto dal disegno) | nessuna — è una riproduzione, il criterio l'ha fissato il paper (0.865) | ✅ riprodotto: MACRO 0.875 / GLOBAL **0.8650 esatto**; `labels` 0.225, `audio_pair` 0.383 (⚠️ **non** bit-riproducibili) | nessuno pre-registrato (riproduzione) | `RESULTS.md` §within e §control ← `results_manifest.tsv` | `clip_loss.py` · `run.py --selftest` · `report.py` verifica ogni pin | ✅ +CKPT · le righe audio_only/eeg_only/clap sono ▶️ |
| **leave-song-out** | `cap1_leave_song_out.sh` | 0.25 | nessuna (riproduzione) | ✅ chiuso 20/20 fold — **0.268**, cioè al caso. eeg_only 0.247 · audio_only_raw 0.181 · contrastive_raw 0.142 | nessuno pre-registrato | `RESULTS.md` §song_out ← manifest (80 pin) | `clip_loss.py` · `run.py --selftest` · pin↔`hparams.yaml` | ✅ (rilegge i pin) · rifare i fold è ▶️ |
| **leave-subject-out** | `cap1_leave_subject_out.sh` | 0.25 | nessuna (riproduzione) | ✅ 0.943 (6 fold clap) · 0.781 (3 fold raw). Tabella 2 di Akama: sub3 0.6642 · sub7 0.8750 · sub2 0.8664 | nessuno pre-registrato | `RESULTS.md` §subject_out ← manifest · intestazione di `scripts/replicate.sh` | idem | ✅ +CKPT · rifare i fold è ▶️ |

### Cap. 2 — MAD-EEG, famiglia lineare

| esperimento | script | null | soglia | verdetto | contratto nel vault | provenienza | canarino | CPU? |
|---|---|---|---|---|---|---|---|---|
| **ancore ridge** (A1/A2/A3) | `ridge_anchors.sh` | 0.500 sul duo, per costruzione | **88/154 = 0.5714** | 🔴 nessuna la supera: A1 **0.5584** (max storico, p 0.0853) · A2 0.5065 · A3 0.4805 | «Cap. 2 — criterio pre-registrato (26 lug 2026)» | `RESULTS.md` §madeeg_duo ← manifest, pin `madeeg_ridge_{duo,solos}_repro2026-08` · A2 in `_baldo_archive_2026-07-18/` | md5 dei due record pinnati + `--self_test` della ridge (soglia 0.90 nel codice) | ✅ +MAD |
| **i quattro assi + ASSE 0** | `axes_paper_protocol.sh` | nessuno — `inner_val_r` è ricostruzione, non decisione | nessuna: è configurazione, non test | ✅ un solo asse guadagna: **ICA +0.0091 (+16%)**. ⚠️ ASSE 2 (0.1100) è **artefatto della metrica**: test equo 36/70, p 0.29 | nessun contratto; nota di metodo «la banda 1-8 Hz è nostra (29 lug 2026)» | report «2026-07-29 (2)» · `runs/results/ax_*` · `madeeg_repro_hparams` | ogni asse ha il default sul comportamento preesistente ed è **assertito**; le tre ancore restano bit-identiche (→ `ridge_anchors.sh`) | ✅ +MAD, **16 GB RAM** |
| **Exp. 4** — duo mono (unico sguardo confermativo) | `exp04_duo_mono.sh` | 0.500 per costruzione | **86/150 = 0.5733** (ricalcolata prima della run) | 🔴 **non replica** — 79/150 = 0.5267, p 0.2839, ⚠️ potenza 0.534 | «Cap. 2 — Exp. 4: criterio pre-registrato sui duo MONO (29 lug 2026, sera)» | `RESULTS.md` §madeeg_duo, pin `madeeg_exp4_mono_confirm` · report «2026-07-30» | `--check_alignment --spatial mono` (r=1.0000 su 272 confronti) **e** lo stesso su stereo che **deve fallire** · md5 del record pinnato | ✅ +MAD (release raw) |
| **Exp. 5** — lato da band-power | `exp05_side_bandpower.sh` | 0.5 teorico, **0.5034 operativo** (75/149, tiratore costante) | **86/149 = 0.5772** | ⚫ **archiviato senza numero.** Nessun file del repo lo implementa; sopravvivono tre soglie (30/47 · 27/42 · 53/89) riprese da Exp. 11/12 e 6[H] | «Cap. 2 — Exp. 5: lato attenzionato da band-power (9 ago 2026)» (archiviato l'11/8, non riscritto) | **nessuna, e non deve esistere** | n/a | ✅ (non esegue niente) |
| **Exp. 6 [G]** — own-vs-other, 4 config | `exp06_ovo_gate.sh` | **0.500 esatto per simmetria** (188 coppie × 2 direzioni) | McNemar: 80/139 base · **76/131** con ICA | 🔴 **NO-GO alla CCA**: 66/131 vs 65, p 0.5000. «la CCA non aggiunge niente», non «perde» | «Cap. 2 — Exp. 6: gate own-vs-other e alfa appaiata (10 ago 2026)» | `runs/results/ovo_{ridge,cca,ridge_ica,cca_ica}/` · report «2026-08-10 (2)» §2 | **md5 `2eaa9262…` + 208/376** rigirato per primo · `--mcnemar` verifica 376/376 confronti condivisi | ✅ +MAD (release raw) |
| **Exp. 6 [H]** — alfa lateralizzata appaiata | `exp06_alpha_paired.sh` | **0.500 esatto per scambiabilità** | **28/44 = 0.6364** primario · **27/42** controllo mono | 🔴 sotto soglia: 23/44 (p 0.4402) · mono 24/42, **più alto del primario**. ⚠️ potenza 0.371 | idem Exp. 6 (soglie dal contratto Exp. 5 del 9/8) | `runs/results/alpha_real_{stereo,mono}/` · `alpha_ctrl2_a*_inject/` · report «2026-08-10 (2)» §3 | iniezione lateralizzata **44/44 nella configurazione esatta della run** (soglia 0.90 nel codice), attraversata prima del numero vero | ✅ +MAD (release raw) |
| Exp. 7 — sweep di 8 configurazioni | `exp07_ovo_config_sweep.sh` | 0.500 esatto | Bonferroni McNemar α = 0.05/8 = 0.00625 | 🟡 nessuna passa. REF **208/376 = 0.5532** · max C3 213/376 | contratto Exp. 7 | `docs/provenance/2026-08-11_ovo_sweep_mcnemar.txt` · `runs/results/ovo_sweep_*` | md5 `2eaa9262…` | ✅ +MAD |
| Exp. 8 — flusso spettrale su own-vs-other | `exp08_flux_ovo.sh` | 0.500 esatto | α = 0.025 | 🟢 F1 **243/376** (p 0.0015) · F2 238/376 | contratto Exp. 8 | `docs/provenance/2026-08-11_exp8_flux_mcnemar.txt` | md5 `2eaa9262…` | ✅ +MAD |
| Exp. 9 — il flux sull'**attenzione** | `exp09_flux_attention.sh` | 0.500 sul duo | barra assoluta **90/154** a α 0.025 | 🔴 il trasferimento fallisce: 76/154 e 77/154 | contratto Exp. 9 (11 ago 2026) | `docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt` | ancore mel **74/154 e 86/154 decision-identiche** (zero discordanti) | ✅ +MAD |
| Exp. 11 — registro spettrale, duo stereo | `exp11_spectral_register_stereo.sh` | 0.500 esatto per simmetria | **30/47** (scritta il 9/8) | 🔴 24/47 = 0.5106 | contratto Exp. 11 | `runs/results/exp11_*` | positivo **47/47**, hard-exit dentro il driver | ✅ +MAD |
| Exp. 12 — lo stesso sui duo mono | `exp12_spectral_register_mono.sh` | 0.500 | **27/42** mono · **53/89** pooled | 🔴 22/42 · 46/89 | contratto Exp. 12 | `runs/results/exp12_*` | positivo **42/42**, hard-exit | ✅ +MAD (serve anche la raw) |
| braccio A — parità di protocollo col paper | `armA_paper_protocol.sh` | 0.500 sul duo | nessuna: è una sonda di protocollo | F1 **0.5267 / 0.4800 / 0.5400** su n=150 — il 79 del paper **non si riproduce** | Piano «la svolta» (11 ago 2026) §4A | `runs/results/armA_paper_*` | nessuno dichiarato (è la riproduzione stessa dei tre F1) | ✅ +MAD, **16 GB RAM** |
| braccio D — audit di leakage a finestre | `armD_leakage_audit.sh` | **0.5136 per costruzione** sul braccio pseudo (⚠️ **non** 0.500: il DOSSIER §8.1 dice «null 0.500 esatto», il file dice `null 0.5136 by construction` — **voce da chiudere**) | regola d'interpretazione scritta nel codice **prima** della run: leakage dimostrato se pseudo window-CV > 0.60 **e** pseudo trial-CV in [0.40, 0.60] | ✅ leakage dimostrato: pseudo window-CV **0.6209** vs trial-CV **0.4847** (etichette vere: 0.3803 vs 0.0895, null 0.2277) | — | `runs/results/armD_leakage_audit/leakage_audit_summary.txt` | il braccio pseudo **è** il controllo positivo dell'audit; v1 conservata come `*_v1_unbalanced*` | ✅ +MAD |

### Cap. 2 — MAD-EEG, separabilità e front end

| esperimento | script | null | soglia | verdetto | contratto nel vault | provenienza | canarino | CPU? |
|---|---|---|---|---|---|---|---|---|
| Exp. 13 — separabilità solo audio | `exp13_stem_separability.sh` | pavimento cross-brano (0.0256 per C3) | due criteri pre-registrati | ✅ **GO**, unico candidato che passa entrambi: C3 MFCC-13 (0.0533 vs 0.0256, 32/36) | contratto Exp. 13 (12 ago 2026) | `docs/provenance/2026-08-12_exp13_stem_separability.txt` | mel-8 **0.1772/0.1442** · flux **0.2874/0.2417** · 30/36 (±0.0010), **hard-exit** | ✅ +MAD (~10 s misurati) |
| Exp. 14 — MFCC tracciate dall'EEG | `exp14_mfcc_tracking.sh` | 0.500 esatto | barra **≥ 208/376** | ✅ entrambe tracciano: T1 214/376 · T2 212/376 | contratto Exp. 14 | `docs/provenance/2026-08-12_exp14_mfcc_tracking.txt` | md5 `2eaa9262…` · identità con C3 a **0.000e+00** · due positivi sintetici 1.0000 | ✅ +MAD |
| Exp. 15 — differenziale attenzionale | `exp15_mfcc_differential.sh` | D = 0 | **MDD 0.2467** | 🟡 non conclusivo per potenza: D(mel) +0.1001 vs D(MFCC) −0.1119, p 0.9833 | contratto Exp. 15 | `docs/provenance/2026-08-12_exp15_mfcc_differential.txt` | ancore decision-identiche **e** correlazioni sottostanti a 0.000e+00 | ✅ +MAD |
| Exp. 16A — CLAP nel gate (criterio ritirato) | `exp16a_clap_separability.sh` | — | ⚠️ **criterio ritirato**, falsificato dal rumore-512 | ⚫ agli atti. **Non si rilancia per giudicare CLAP** — usa l'Exp. 17. Gira comunque il controllo negativo (0.922) | — | `docs/provenance/2026-08-12_exp16A_{clap_separability,negative_control}.txt` | md5 `f3ebe89b…` | ✅ (stadio 2) · stadio 1 ▶️ |
| Exp. 16B — `band_power` come vista CCA | `exp16b_ccaviews.sh` | 0.500 esatto | barra **≥ 208/376** | 🔴 202/376 = 0.5372, barra non superata | contratto Exp. 16 braccio B | `docs/provenance/2026-08-12_exp16B_ccaviews_ownvsother.txt` | md5 `2eaa9262…` · CCA 1-vista ri-misurata 209/376 con 0 discordanti | ✅ +MAD |
| Exp. 17 — CLAP col criterio invariante | `exp17_clap_separability.sh` | rumore-512 come pavimento (0.922) | criterio 2, invariante alla dimensionalità | 🔴 **NO-GO**: CLAP 3.837 · mel-8 3.131 · MFCC-13 2.084 | contratto Exp. 17 (12 ago 2026) | `docs/provenance/2026-08-12_exp17_clap_separability.txt` + log e manifest | rumore-512 **nella stessa run, prima** della riga CLAP · md5 `a3c7533c…`/`f3ebe89b…` | ✅ (stadio 2, 24 s) · estrazione ▶️ |

### Cap. 2 — il braccio contrastivo (GPU)

| esperimento | script | null | soglia | verdetto | contratto nel vault | provenienza | canarino | CPU? |
|---|---|---|---|---|---|---|---|---|
| **passo C** — CLAP↔EEG, negativi di batch | `stepC_contrastive_clap.sh` | **0.500 per costruzione** | **88/154 = 0.5714** | 🔴 **sotto il caso e spiegato**: 58/154 = 0.3766, IC95 [0.300, 0.458] esclude 0.50. 🚫 il segno **non si gira** | «Cap. 2 — criterio pre-registrato (26 lug 2026)» | `RESULTS.md` §madeeg_duo, pin `madeeg_clap_kfold` · `_baldo_archive_2026-07-18/` | `clip_loss.py` · `--loss` assertito · `_self_check` dell'adapter · **`--check_rule` → 58/154 esatto** | cancelli ✅ · training ▶️ |
| **passo D** — loss within-mixture | `stepD_within_mixture.sh` | **0.500 per costruzione** | **88/154 = 0.5714** (la stessa, non ricalcolata) | 🔴🔴 primario 70/154 = 0.4545 **e** secondario **falsificato al contrario**: il prior-following sale 0.7177 → 0.8145 | «Cap. 2 — passo D: loss within-mixture (28 lug 2026)» | `_baldo_archive_.../madeeg_clap_{kfold,selftest}_within/` · `docs/provenance/2026-08-11_diagnose_step{C,D}.txt`. ⚠️ **non pinnato** nel manifest | `clip_loss.py` 2ª riga = **l'obiettivo del passo D** (0.951610 vs 2.160834) · un `--loss` mal scritto **rompe** · directory separate | cancelli ✅ · training ▶️ |
| Exp. 18 — match-mismatch sui solo (preparazione) | `exp18_matchmismatch.sh` | S1 **0.5000** · S2 0.5063 (held-out **0.6061**) | 0.70 sull'held-out | preparazione soltanto: **non allena senza flag esplicito** | contratto Exp. 18 | `docs/provenance/2026-08-12_exp18_matchmismatch.txt` | canarino Cap. 1 · campionamento provato coppia per coppia · overfit 20/20 | ✅ ma con `/opt/anaconda3` (serve torch) · training ▶️ |
| Exp. 19 — budget di epoche + null di S2 riparato | `exp19_matchmismatch.sh` | S2 **0.6061 → 0.5000** dopo `--balance_pairs` | 0.70 (invariata) | preparazione soltanto: **stampa i due comandi ▶️** | contratto Exp. 19 | `docs/provenance/2026-08-13_exp19_S{1,2}.txt` | i quattro cancelli §3.1–§3.4, canarino Cap. 1 identico prima **e** dopo il diff | ✅ con `/opt/anaconda3` · training ▶️ |

### Trasversali

| cosa | script | null | soglia | verdetto | contratto | provenienza | canarino | CPU? |
|---|---|---|---|---|---|---|---|---|
| **tutti i canarini, nell'ordine** | `canaries.sh` | n/a — nessun cancello produce un risultato | ognuno porta la sua, letterale, in testa allo script | lo stampa la run; **uscita ≠ 0** se un canarino attraversabile fallisce | nessuno: non è un esperimento ([[Comandamenti]] §3/§8) | DOSSIER §9 · `docs/provenance/` | **è** il file dei canarini | L0 ✅ · L1 serve torch · L2 +MAD · L3 ▶️ |
| exp13 · exp07 · exp08 · exp14 in sequenza | `run_all_cheap.sh` | — | — | — | — | — | quelli dei quattro | ✅ +MAD (~25–50 min) |

**Come si legge la colonna «sguardi»** (che sta nell'intestazione di ogni script, non
qui): *NO* = il registro degli sguardi dichiara l'esperimento gratuito (solo audio,
oppure own-vs-other su segmenti solo held-out). *SÌ* = decide sui duo. I 309 duo sono
**tutti spesi** dal 30/7: rigirarli non apre materiale nuovo, ma ogni numero sui duo è
**esplorativo per costruzione** e va etichettato così nel titolo. Spendono sguardi:
`ridge_anchors` · `axes_paper_protocol` · `exp04` · `exp06_alpha_paired` · `exp09` ·
`exp11` · `exp12` · `exp15` · `armA` · `armD` · `stepC` · `stepD`.
🔒 **Nessuno script di questa cartella tocca i trio** (92 stereo + 93 mono, l'unico
holdout rimasto, un colpo solo, soglia stereo già scritta 38/90).

---

## Da zero: ricreare l'ambiente, i dati, l'ordine

### 1. Gli ambienti (sono **tre**, e non sono intercambiabili)

| a cosa serve | interprete | cosa ci vuole dentro |
|---|---|---|
| canarini `torch` (`clip_loss.py`), Cap. 1 | conda **`attention`** — `eeg_attention` su baldo | `torch==2.2.2`, `numpy<2`, `pytorch_lightning==1.9.5` → `pip install -r requirements.txt` |
| analisi MAD-EEG su CPU (tutto `madeeg_*.py`) | **`/opt/miniconda3/bin/python`** | numpy · scipy · sklearn · **h5py** · mne · librosa · pandas · pyyaml |
| Exp. 16A/17/18/19 (serve `torch` **e** `laion_clap`) | **`/opt/anaconda3/bin/python`** | torch + `laion_clap` 1.1.6 |

Perché tre e non uno: l'env `attention` **non ha `h5py`** e non può eseguire
`madeeg_reconstruction.py`; `/opt/miniconda3` non ha `torch`. Non è una comodità, è
la ragione per cui gli script hanno `$PY` **e** `$PY_TORCH` separati.

```bash
# env del Cap. 1 — 3.9 è la versione dichiarata dal README di root; requirements.txt
# pinna torch 2.2.2 + numpy 1.26.4 + pytorch_lightning 1.9.5, che è il vincolo vero.
conda create -n attention python=3.9 && conda activate attention
pip install -r requirements.txt
python src/run.py --selftest        # deve dire: 23 protocol flags ... PASS
python src/modules/clip_loss.py     # deve dire: 0.628491 · 1.2994 · 4.8198 · 0.951610 vs 2.160834
```

### 2. Dove vanno i dati

- **MAD-EEG non è nel repo** (4.7 GB). Sorgente: `Tesi/MAD-MEG_dataset.zip`, oppure
  `MADEEG_DIR=~/madeeg bash scripts/madeeg_setup.sh`.
  ⚠️ `madeeg_setup.sh` scarica **solo la release preprocessed**: il percorso
  `raw_solos` (own-vs-other, Exp. 4/6/9, braccio A, assi) vuole **anche**
  `madeeg_raw.hdf5` + `madeeg_raw.yaml` + `madeeg_sequences_raw.yaml` + `stimuli/`.
  Lo script se ne accorge e lo dice; non lo indovina.
- **Checkpoint del paper** (Cap. 1): `bash scripts/setup_checkpoints.sh`, che li
  spacchetta da `archive/*.7z`. I `.ckpt` sono gitignored.
- **Checkpoint CLAP** `630k-audioset-best.pt` (1.74 GiB), sha256
  `8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`: scaricarlo è
  una decisione di A., non degli script.
- **`runs/` è gitignored.** Da un clone pulito i record dei run **non esistono**: gli
  script che li rileggono lo dichiarano e non fingono. L'unica eccezione è
  `!docs/provenance/*.log`.

### 3. L'ordine in cui si esegue

```bash
# 0. SEMPRE per primo, dopo ogni diff. Livello 0 non ha bisogno di niente.
LEVELS=0 bash scripts/replicate/canaries.sh
PY_TORCH=~/miniconda3/envs/attention/bin/python LEVELS="0 1" bash scripts/replicate/canaries.sh

# 1. Cap. 1, dai checkpoint rilasciati (nessun training)
bash scripts/setup_checkpoints.sh
bash scripts/replicate/cap1_within_split.sh          # ~25 min CPU
bash scripts/replicate/cap1_leave_song_out.sh        # rilegge gli 80 pin, secondi
bash scripts/replicate/cap1_leave_subject_out.sh     # ~20 min CPU

# 2. Cap. 2 gratuito (nessuno sguardo speso): solo audio e own-vs-other
bash scripts/replicate/exp13_stem_separability.sh    # ~10 s — il gate di separabilità
bash scripts/replicate/exp06_ovo_gate.sh             # il riferimento 208/376 + md5
bash scripts/replicate/exp07_ovo_config_sweep.sh
bash scripts/replicate/exp08_flux_ovo.sh
bash scripts/replicate/exp14_mfcc_tracking.sh
bash scripts/replicate/exp16b_ccaviews.sh
bash scripts/replicate/exp17_clap_separability.sh    # stadio 2

# 3. Cap. 2 che SPENDE SGUARDI (duo già spesi, ma il caveat va scritto nel titolo)
bash scripts/replicate/ridge_anchors.sh
bash scripts/replicate/axes_paper_protocol.sh        # 16 GB RAM
bash scripts/replicate/exp04_duo_mono.sh
bash scripts/replicate/exp06_alpha_paired.sh
bash scripts/replicate/exp09_flux_attention.sh
bash scripts/replicate/exp11_spectral_register_stereo.sh
bash scripts/replicate/exp12_spectral_register_mono.sh
bash scripts/replicate/exp15_mfcc_differential.sh
bash scripts/replicate/armA_paper_protocol.sh
bash scripts/replicate/armD_leakage_audit.sh

# 4. Livello 2 dei canarini (rigira il riferimento e le ancore da zero)
LEVELS=2 bash scripts/replicate/canaries.sh

# 5. Il braccio contrastivo: cancelli su CPU, training ▶️
bash scripts/replicate/stepC_contrastive_clap.sh     # prepara e STAMPA
bash scripts/replicate/stepD_within_mixture.sh       # prepara e STAMPA
bash scripts/replicate/exp18_matchmismatch.sh
bash scripts/replicate/exp19_matchmismatch.sh
```

### 4. Cosa richiede A. e la GPU (▶️)

Gli agenti **preparano il comando, non lo lanciano** ([[Comandamenti]] §11). Ogni
script qui stampa il suo blocco ▶️ e si ferma. La lista completa:

| ▶️ | dove viene stampato | cosa produce |
|---|---|---|
| `PHASES="within sanity" bash scripts/replicate.sh` dopo `setup_checkpoints.sh` | `canaries.sh` L3-a · `cap1_within_split.sh` | GLOBAL **0.8650 esatto** end-to-end. ⚠️ è CPU, ma vuole i checkpoint |
| `run.py train --model {audio_only,eeg_only,clap}` + `main.py --objective classify_audio --audio_repr clap` | `cap1_within_split.sh` | le righe `within` che **non** hanno checkpoint rilasciati, incluso il **controllo di leakage 0.996** |
| `run.py sweep --cv song` (4 modelli × 20 canzoni) | `cap1_leave_song_out.sh` | rifà i fold di `song_out`, ~80 min/fold |
| `run.py sweep --cv subject` | `cap1_leave_subject_out.sh` | rifà i fold di `subject_out` |
| `madeeg_contrastive.py --self_test [--loss within_mixture] --kfold 5` | `canaries.sh` L3-b · stepC/stepD | i controlli sintetici **148/154** e **142/154** (soglia 0.90 nel codice) |
| `madeeg_contrastive.py --kfold 5 [--loss within_mixture]` | stepC/stepD | i numeri veri 58/154 e 70/154 |
| i due comandi di training dell'Exp. 18/19 | `exp18`/`exp19_matchmismatch.sh` | S1/S2 match-mismatch |
| estrazione CLAP (stadio 1) | `exp16a`/`exp17_clap_separability.sh` | gli embedding + il manifest con lo sha256 del checkpoint |

Dopo un ▶️ la diagnostica torna su CPU: i due comandi
`madeeg_diagnose.py --madeeg_dir ~/madeeg --records …` per i passi C e D sono
stampati in fondo a `stepD_within_mixture.sh` e riproducono
`docs/provenance/2026-08-11_diagnose_step{C,D}.txt`.

---

## Regole che gli script rispettano, e che chi li modifica non rinegozia

- **Il canarino per primo.** Dove esiste, viene attraversato prima di qualunque
  numero vero e lo script si ferma se fallisce ([[Comandamenti]] §3). Exp. 6/7/8/14/16B
  controllano il md5 `2eaa926244de340d31907c6deeb04b0c` del percorso di default;
  Exp. 9 e 15 controllano che le ancore mel siano **decision-identiche** trial per
  trial (74/154 e 86/154, zero discordanti); Exp. 11/12/13 hanno i cancelli
  dentro i driver, che escono con errore prima di calcolare il numero vero.
- **Un controllo non scrive sopra un risultato** ([[Comandamenti]] §9): i
  `--training_date` sono prefissati con `$TAG` e la provenienza va in `$OUT_DIR`.
  Due eccezioni dichiarate: i driver `madeeg_exp1{4,5}_*.py` non accettano
  `--log_dir` e riscrivono `runs/results/exp1{4,5}_*` — i file di provenienza
  pinnati restano intatti perché si passa `--out`. È scritto in testa a quei due script.
- **Un canarino non attraversato non è un canarino passato.** Gli script che
  cercano `$PY_TORCH` e non lo trovano stampano `⏭️ NON ATTRAVERSATO` e lo dicono
  esplicitamente. Non c'è una quarta categoria fra passato, fallito e non attraversato.
- **Niente invocazioni plausibili** ([[Comandamenti]] §10): quello che non è
  ricostruibile con certezza è commentato con `DA CONFERMARE:` e la ragione.
  Oggi i casi aperti sono: lo **split same/diff-melody del braccio A** (nessun file
  del repo lo produce, lo script non è stato salvato); i **costi in minuti** di
  buona parte degli script del Cap. 2 (i timestamp dell'11/8 sono identici e non
  danno la durata); la **causa esatta** dei 5 trial mancanti sul braccio mono
  dell'Exp. 4 (il conteggio 155/150 è scritto, il meccanismo no).
- **Niente GPU** ([[Comandamenti]] §11). Nessuno script qui ne ha bisogno per
  girare: dove serve, il comando viene **stampato**, non lanciato.

## Cosa NON è coperto da questa cartella

- **Exp. 10** — ritirato **prima** della run (insensibile per algebra). Il flag
  `--score_rule ortho` esiste ed è inerte: **non produrre numeri con esso**.
  Il toy che lo dimostra è `docs/provenance/2026-08-11_exp10_retired_toy_sweep.py`
  e lo attraversa `canaries.sh` L0-d.
- **C1-8 (`--eeg_repr spectra`)** — 0.9350 / 0.4975 / 0.4713: mai aggregato, mai
  pinnato, **nessun canarino**. Non ha uno script e non deve averne uno finché quei
  numeri non entrano nel manifest.
- **Lo split same/diff-melody del braccio A** — 🔴 nessun file del repo lo produce.
  I numeri (0.5169/0.5410 e 0.5730/0.4918) hanno il file di provenienza
  (`docs/provenance/2026-08-11_armA_same_diff_melody.txt`) **ma non il codice**.
  Va riscritto, non indovinato.
- **La run vera dell'Exp. 18 S1** — girata su baldo; `exp18_matchmismatch.sh` copre
  solo la preparazione, e ⚠️ il suo risultato **non ha un file** in `docs/provenance/`.
