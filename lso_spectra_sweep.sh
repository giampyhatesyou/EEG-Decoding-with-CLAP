#!/usr/bin/env bash
# SpectraCLIP experiment, end-to-end and RESUMABLE.
# Matrix: eeg_repr {raw, spectra} x objective {classify_eeg, contrastive(audio=raw)}
#         x eval {within, leave_song_out}. For each cell it trains a fresh model and
#         tests it; LSO sweeps several held-out songs per class.
# Resume: scans results/ (eeg_repr absent in old hparams -> treated as raw) and SKIPS
#         any (objective,audio_repr,eeg_repr,cv_mode,held) already done.
# 2 GPUs: launch twice with SHARD=0 and SHARD=1 (NSHARD=2) -> even/odd songs, no overlap.
set -uo pipefail
cd "$(dirname "$0")/src"   # main.py / checkpoint_test.py live in src/

if [ "${CONDA_DEFAULT_ENV:-}" != "eeg_attention" ]; then
  for c in /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
           "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && . "$c" && break
  done
  conda activate eeg_attention 2>/dev/null || true
fi
PY=python
NSHARD=${NSHARD:-1}; SHARD=${SHARD:-0}     # GPU sharding (default: single GPU does all)
BUDGET=${BUDGET:-25200}                    # 7h launch cutoff
CAP=${CAP:-80m}                            # per-training-fold cap
echo "[env] $(date) env=${CONDA_DEFAULT_ENV:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SHARD=$SHARD/$NSHARD"
$PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC

PROTO="--dataset preprocessing_eegmusic --test_dataset preprocessing_eegmusic_test \
--max_epochs 1000 --batch_size 8 --eeg_length 768 --loss_function clip_loss \
--eeg_normalization MetaAI --clamp_value 20 --learning_rate 0.003 --supervised 1 \
--dim_reduction 1 --split_seed 42 --detach_z_c 0 --window_size 1280 --stride 256 \
--test_window_size 768 --test_stride 256 --start_position 0 --key all \
--attention_values 4 5 --devices 1 --shifting_time 0 --seed 42"

# done-set keyed by (objective, audio_repr, eeg_repr, cv_mode, held); eeg_repr absent -> raw
$PY - <<'PYC' > /tmp/spectra_done.txt
import glob, os
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','eeg_repr':'raw','cv_mode':'?','cv_held_out_id':'?'}
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
    if c['cv_mode'] not in ('within','leave_song_out'): continue
    seen.add(f"{c['objective']}|{c['audio_repr']}|{c['eeg_repr']}|{c['cv_mode']}|{c['cv_held_out_id']}")
print('\n'.join(sorted(seen)))
PYC
echo "[resume] folds already done: $(wc -l < /tmp/spectra_done.txt)"
is_done(){ grep -qxF "$1" /tmp/spectra_done.txt; }

run_fold(){   # $1 obj  $2 audio_repr  $3 eeg_repr  $4 cv_mode  $5 held  $6 tag
  local obj="$1" arepr="$2" erepr="$3" cvm="$4" held="$5" tag="$6"
  local key="$obj|$arepr|$erepr|$cvm|$held"
  if is_done "$key"; then echo "  skip(done): $key"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  ($key)  elapsed $((SECONDS/60))m ====="
  local objflag=""; [ "$obj" != "contrastive" ] && objflag="--objective $obj"
  local eegflag=""; [ "$erepr" != "raw" ] && eegflag="--eeg_repr $erepr"
  local cvflag="--cv_mode $cvm --cv_held_out_id $held"
  echo "  TRAIN"
  timeout $CAP $PY -u main.py $PROTO $cvflag --audio_repr "$arepr" $eegflag $objflag --training_date "$tag"
  if compgen -G "../results/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO $cvflag --audio_repr "$arepr" $eegflag $objflag --training_date "$tag" --test_breakdown 1
    echo "$key" >> /tmp/spectra_done.txt
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}

tag_for(){    # obj audio eeg cvmode held -> tag
  local pre=""; [ "$3" != "raw" ] && pre="${3}_"
  local o; case "$1" in classify_eeg) o="clf_eeg";; contrastive) o="ct_$2";; esac
  if [ "$4" = "within" ]; then echo "${pre}${o}_within"; else echo "${pre}${o}_lso_song$5"; fi
}

# Cells (objective:audio_repr:eeg_repr). contrastive holds audio=raw to isolate eeg_repr.
CELLS="classify_eeg:raw:raw classify_eeg:raw:spectra contrastive:raw:raw contrastive:raw:spectra"
declare -A SONGS=( [vocal]="2 3 5 33 45 38 39" [drum]="7 18 59 112 145 36 115" \
                   [bass]="44 50 55 68 140 86 141" [others]="8 16 43 62 120 63 135" )
CLASSES="vocal drum bass others"

# --- Phase 1: within (1 per cell), only on shard 0 to avoid duplication ---
if [ "$SHARD" -eq 0 ]; then
  for cell in $CELLS; do
    obj="${cell%%:*}"; rest="${cell#*:}"; arepr="${rest%%:*}"; erepr="${rest#*:}"
    [ $SECONDS -ge $BUDGET ] && break
    run_fold "$obj" "$arepr" "$erepr" within -1 "$(tag_for "$obj" "$arepr" "$erepr" within -1)"
  done
fi

# --- Phase 2: leave-song-out, round-robin by class, sharded by song index ---
idx=0
for ((r=0;r<7;r++)); do
  for cls in $CLASSES; do
    arr=(${SONGS[$cls]}); s=${arr[$r]:-}; [ -z "$s" ] && continue
    if [ $((idx % NSHARD)) -eq "$SHARD" ]; then
      for cell in $CELLS; do
        obj="${cell%%:*}"; rest="${cell#*:}"; arepr="${rest%%:*}"; erepr="${rest#*:}"
        [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m -> stop"; break 3; }
        run_fold "$obj" "$arepr" "$erepr" leave_song_out "$s" \
                 "$(tag_for "$obj" "$arepr" "$erepr" leave_song_out "$s")"
      done
    fi
    idx=$((idx+1))
  done
done

echo; echo "######## SPECTRA AGGREGATE (within + leave_song_out) ########"
$PY - <<'PYC'
import glob, os, pandas as pd
TASK={0:'vocal',1:'drum',2:'bass',3:'others'}
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','eeg_repr':'raw','cv_mode':'?','cv_held_out_id':'?'}
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
    if c['cv_mode'] not in ('within','leave_song_out'): continue
    if c['audio_repr'] not in ('raw','?'): continue   # this experiment fixes audio=raw
    try: df=pd.read_csv(f)
    except Exception: continue
    if 'correct' not in df or 'task' not in df or len(df)==0: continue
    byfold[(c['objective'],c['eeg_repr'],c['cv_mode'],c['cv_held_out_id'])]=df
groups={}
for (obj,erepr,cvm,held),df in byfold.items(): groups.setdefault((obj,erepr,cvm),[]).append(df)
print("{:34}{:>6}{:>8}{:>8}".format("objective / eeg_repr / eval","folds","MACRO","GLOBAL"))
for k in sorted(groups):
    dfs=groups[k]; alld=pd.concat(dfs); per=alld.groupby('task')['correct'].mean()
    print("{:34}{:>6}{:>8.3f}{:>8.3f}".format("/".join(k), len(dfs), per.mean(), alld['correct'].mean()))
print("chance=0.25 ; compare spectra vs raw at eval=leave_song_out (the honest metric)")
PYC
echo "######## DONE ########"
