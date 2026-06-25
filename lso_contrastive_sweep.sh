#!/usr/bin/env bash
# RESUMABLE leave-song-out sweep for the CONTRASTIVE models (the full EEG+audio model):
#   contrastive raw (biggest gap: only 1 fold so far), then contrastive clap.
# Mirror of the controls sweep; trains a fresh model per (model x held-out song),
# tests on the held-out song, aggregates MACRO at the end.
set -uo pipefail
cd "$(dirname "$0")/src"   # entrypoints main.py/checkpoint_test.py live in src/

# --- env (no-op if already in eeg_attention) ---
if [ "${CONDA_DEFAULT_ENV:-}" != "eeg_attention" ]; then
  for c in /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
           "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && . "$c" && break
  done
  conda activate eeg_attention 2>/dev/null || true
fi
PY=python
echo "[env] $(date) python=$(command -v $PY) env=${CONDA_DEFAULT_ENV:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
$PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC

BUDGET=${BUDGET:-25200}     # 7h launch cutoff
CAP=${CAP:-90m}            # per-fold cap (contrastive is heavier, esp. clap)

PROTO="--dataset preprocessing_eegmusic --test_dataset preprocessing_eegmusic_test \
--max_epochs 1000 --batch_size 8 --eeg_length 768 --loss_function clip_loss \
--eeg_normalization MetaAI --clamp_value 20 --learning_rate 0.003 --supervised 1 \
--dim_reduction 1 --split_seed 42 --detach_z_c 0 --window_size 1280 --stride 256 \
--test_window_size 768 --test_stride 256 --start_position 0 --key all \
--attention_values 4 5 --devices 1 --shifting_time 0 --seed 42 --cv_mode leave_song_out"

# --- done-set: only true leave_song_out, contrastive (resume across sessions) ---
$PY - <<'PYC' > /tmp/lso_done_ct.txt
import glob, os
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_mode':'?','cv_held_out_id':'?'}
    try:
        for l in open(os.path.join(d,'hparams.yaml')):
            s=l.strip()
            for k in c:
                if s.startswith(k+':'): c[k]=s.split(':',1)[1].strip().strip('\'"')
    except Exception: pass
    return c
seen=set()
for f in glob.glob('../results/*/nmed-CL-*/version_*/test_records.csv'):
    c=hp(os.path.dirname(f))
    if c['cv_mode']!='leave_song_out': continue
    if not c['objective'].startswith('contrastive'): continue
    seen.add(f"contrastive|{c['audio_repr']}|{c['cv_held_out_id']}")
print('\n'.join(sorted(x for x in seen if x)))
PYC
echo "[resume] contrastive LSO folds already done: $(wc -l < /tmp/lso_done_ct.txt)"
is_done(){ grep -qxF "contrastive|$1|$2" /tmp/lso_done_ct.txt; }

run_fold(){   # $1 repr  $2 song  $3 tag
  local repr="$1" song="$2" tag="$3"
  if is_done "$repr" "$song"; then echo "  skip(done): contrastive|$repr|$song"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  (contrastive|$repr|$song)  elapsed $((SECONDS/60))m ====="
  echo "  TRAIN"
  timeout $CAP $PY -u main.py $PROTO --audio_repr "$repr" \
        --cv_held_out_id "$song" --training_date "$tag"
  if compgen -G "../results/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO --audio_repr "$repr" \
        --cv_held_out_id "$song" --training_date "$tag" --test_breakdown 1
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}

declare -A SONGS=( [vocal]="2 3 5 33 45 38 39" [drum]="7 18 59 112 145 36 115" \
                   [bass]="44 50 55 68 140 86 141" [others]="8 16 43 62 120 63 135" )
CLASSES="vocal drum bass others"
NROUND=7

# Phase 1: contrastive RAW (biggest gap). Phase 2: contrastive CLAP (deepen).
for repr in raw clap; do
  for ((r=0;r<NROUND;r++)); do
    for cls in $CLASSES; do
      arr=(${SONGS[$cls]}); s=${arr[$r]:-}; [ -z "$s" ] && continue
      [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop"; break 3; }
      run_fold "$repr" "$s" "ct_${repr}_lso_song$s"
    done
  done
done

echo; echo "######## CONTRASTIVE AGGREGATE (true leave-song-out only) ########"
$PY - <<'PYC'
import glob, os, pandas as pd
TASK={0:'vocal',1:'drum',2:'bass',3:'others'}
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_mode':'?','cv_held_out_id':'?'}
    try:
        for l in open(os.path.join(d,'hparams.yaml')):
            s=l.strip()
            for k in c:
                if s.startswith(k+':'): c[k]=s.split(':',1)[1].strip().strip('\'"')
    except Exception: pass
    return c
byfold={}
for f in glob.glob('../results/*/nmed-CL-*/version_*/test_records.csv'):
    c=hp(os.path.dirname(f))
    if c['cv_mode']!='leave_song_out' or not c['objective'].startswith('contrastive'): continue
    try: df=pd.read_csv(f)
    except Exception: continue
    if 'correct' not in df or 'task' not in df or len(df)==0: continue
    byfold[(c['audio_repr'],c['cv_held_out_id'])]=df
groups={}
for (rep,song),df in byfold.items(): groups.setdefault(rep,[]).append((int(song),df))
for rep in sorted(groups):
    items=groups[rep]; alld=pd.concat([d for _,d in items]); per=alld.groupby('task')['correct'].mean()
    print(f"\n===== contrastive/{rep}  ({len(items)} folds, {len(alld)} windows) =====")
    for t,v in per.items(): print(f"   held-out {TASK[int(t)]:7}: acc={v:.3f}")
    print(f"   MACRO={per.mean():.3f}  GLOBAL={alld['correct'].mean():.3f}  (chance 0.25)")
    print(f"   songs={sorted(s for s,_ in items)}  classes={[TASK[int(t)] for t in per.index]}")
PYC
echo "######## DONE ########"
