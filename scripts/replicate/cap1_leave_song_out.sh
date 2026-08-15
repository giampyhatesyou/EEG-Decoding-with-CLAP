#!/usr/bin/env bash
# =============================================================================
# Cap. 1 — LEAVE-SONG-OUT: il crollo su canzoni mai viste, e le tre baseline
#          che gli stanno accanto.
#
# COSA MISURA: se il contrastivo regge quando la canzone di test non e' mai stata
#   vista in training. 20 fold (una canzone tenuta fuori per volta) x 4 modelli.
#   E' IL numero del Cap. 1: 0.865 within contro 0.268 leave-song-out.
# DATA       : chiuso 20/20 il 26 lug 2026.
# DOMANDA    : «il modello regge su canzoni mai viste?»
# NULL       : 0.25 (4 slot strumento fissi, assunto dal disegno).
# SOGLIA PRE-REGISTRATA: nessuna. Non c'era un test: c'era una riproduzione, e il
#              disegno a 20 fold era fissato dal paper. Il verdetto e' descrittivo.
# VERDETTO   : ✅ chiuso 20/20 fold — e il numero e' 0.268, cioe' AL CASO.
# COSTO      : ▶️ GPU, ~80 min per fold x 20 fold x 4 modelli. Lo sweep e'
#              RIESUMABILE: rilanciarlo salta i fold gia' fatti.
#              Sul Mac questo script NON allena: verifica i pin e stampa i comandi.
# SGUARDI    : n/a (contabilita' del Cap. 2, non di questo dataset).
# GPU        : SI', per rifare i fold. NO per rileggere i numeri gia' pinnati.
#
# NUMERI DI RIFERIMENTO ATTESI (RESULTS.md §song_out, 20 fold ciascuno):
#   contrastive_clap  MACRO 0.268   per-classe [v d b o] 0.57 0.07 0.35 0.08
#   eeg_only          MACRO 0.247
#   audio_only_raw    MACRO 0.181
#   contrastive_raw   MACRO 0.142
#
# ⚠️ TRE CAVEAT OBBLIGATORI, tutti e tre da scrivere accanto al numero:
#   (a) **VINTAGE MISTO** sulla riga contrastive_clap: 6 fold sono di codice
#       2026-05 e 14 di codice 2026-07. Sono DUE EPOCHE DI CODICE nella stessa
#       media. La tabella omogenea a 14 fold da' **0.258** e sta in
#       `40 Codice-Repo/Risultati e provenienza.md`. La decisione di stesura
#       (quale delle due va in tesi) e' APERTA dal 26/7 (DOSSIER §5.3).
#   (b) **SOTTO IL CASO NON E' UN ERRORE DI MISURA**: 0.142 e 0.181 sono
#       prior-following su fold mono-classe. Va detto cosi', non «il modello
#       sbaglia».
#   (c) **`contrastive_raw` e' passato 0.142 -> 0.154 -> 0.142** per un dedup
#       arbitrario, prima che i pin esistessero. E' esattamente l'incidente che
#       ha prodotto `results_manifest.tsv`. Il numero di oggi e' 0.142.
#
# ⚠️ Canzone 36 (drum) NON e' pinnata: fuori dal disegno 5x4, vintage maggio
#    (manifest, riga «Deliberately NOT pinned»). 20 fold vuol dire 20, non 21.
#
# CANARINO   : (1) `src/modules/clip_loss.py` (l'aritmetica InfoNCE);
#              (2) `src/run.py --selftest` (23 flag: lo sweep e replicate.sh
#                  devono passare lo STESSO $PROTO);
#              (3) `report.py` verifica ogni pin contro l'hparams.yaml del run —
#                  in particolare la chiave del fold include `cv_mode` e
#                  `eeg_repr`, perche' due volte un fold ne ha oscurato un altro
#                  (10cdc49: un leave_subject_out ha oscurato un leave_song_out
#                  con lo stesso id; 26/7: un fold `spectra` ha oscurato il suo
#                  gemello `raw`).
# CONTRATTO  : nessuno pre-registrato (riproduzione).
# PROVENIENZA: RESULTS.md §song_out <- results_manifest.tsv (80 pin: 4 modelli
#              x 20 canzoni) · tabella omogenea a 14 fold nel vault,
#              `40 Codice-Repo/Risultati e provenienza.md`
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}

cd "$REPO"

echo "=== CANARINO 1 — src/modules/clip_loss.py (sintetico, nessun dato) ==="
if [ -z "$PY_TORCH" ]; then
  for C in "$HOME/miniconda3/envs/attention/bin/python" \
           "$HOME/anaconda3/envs/attention/bin/python" \
           "/opt/miniconda3/envs/attention/bin/python" \
           "/opt/anaconda3/envs/attention/bin/python" \
           "$HOME/.conda/envs/eeg_attention/bin/python"; do
    [ -x "$C" ] && PY_TORCH="$C" && break
  done
fi
if [ -n "$PY_TORCH" ] && [ -x "$PY_TORCH" ]; then
  "$PY_TORCH" src/modules/clip_loss.py
else
  echo "  ⏭️  NON ATTRAVERSATO (nessun torch). PY_TORCH=... bash \$0"
fi

echo
echo "=== CANARINO 2 — src/run.py --selftest ==="
"$PY" src/run.py --selftest

echo
echo "=== I NUMERI PINNATI (nessun training: si rileggono i 80 fold gia' fatti) ==="
# report.py --check esce 1 se anche UN pin non e' verificabile. Con `set -e` + pipefail
# una pipe farebbe morire lo script PRIMA dell'avvertenza qui sotto — cioe' proprio nel
# caso che l'avvertenza esiste per spiegare. Si cattura l'uscita e la si dichiara.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## song_out/,/^## subject_out/p'
echo "⚠️ se qui sopra manca una riga, il suo run dir non e' su questa macchina:"
echo "   runs/ e' gitignored e i fold vivono solo dove sono stati girati (baldo)."
if [ "$RC" != 0 ]; then
  echo "⚠️ report.py --check e' uscito $RC — pin non verificabili. Le righe [BAD PIN]:"
  echo "$CHECK_OUT" | grep "^\[BAD PIN\]" || true
fi

cat <<'EOF'

###############################################################################
# ▶️ RIFARE I FOLD RICHIEDE GPU — LI LANCIA A. ([[Comandamenti]] §11)
###############################################################################
# Lo sweep e' RIESUMABILE: la chiave del "gia' fatto" e' l'identita' PIENA del
# fold (objective | audio_repr | eeg_repr | held-out id, dentro un cv_mode).
# Rilanciarlo dopo un'interruzione riparte da dove era, non da zero.

cd $REPO
python src/run.py sweep --cv song                 # 4 modelli x 20 canzoni

# equivalente esplicito, se serve controllare il cap per fold o la GPU:
CUDA_VISIBLE_DEVICES=0 CAP=80m bash sweeps/sweep_song_out.sh

# ⚠️ `conda activate` NON funziona in shell non interattive (nohup/sbatch/cron):
#    diventa un no-op e `python` nudo e' quello di sistema, SENZA torch —
#    e l'errore esce minuti dopo il lancio, non subito. sweep_common.sh risolve
#    l'interprete PER PATH proprio per questo. Un PY=... esplicito vince sempre.
#
# Su baldo NON c'e' tmux: si usa nohup, cosi' la sessione web puo' cadere senza
# portarsi via la run.
cd $REPO && nohup python src/run.py sweep --cv song > runs/logs/lso_song.log 2>&1 &

# Quando i fold sono finiti: si pinna e si rigenera. report.py rifiuta di
# scrivere la tabella se anche un solo pin non torna col suo hparams.yaml.
python src/run.py report
EOF
