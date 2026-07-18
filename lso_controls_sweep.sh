#!/usr/bin/env bash
# Focused, RESUMABLE leave-song-out sweep for the CONTROLS only:
#   audio-only (classify_audio raw), eeg-only (classify_eeg), then audio-only (clap).
# Purpose: fill in the missing folds so each control has a solid per-class MACRO
#          (error bars) on truly-unseen songs.
# Resume : scans results/ and SKIPS any (objective,audio_repr,song) already tested.
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"   # conda env + $PY + $PROTO
cd "$(dirname "$0")/src"   # entrypoints main.py/checkpoint_test.py live in src/
echo "[env] $(date) python=$(command -v $PY) env=${CONDA_DEFAULT_ENV:-none}"
$PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC

BUDGET=${BUDGET:-25200}     # 7h launch cutoff (leaves ~1h headroom in an 8h slot)
CAP=${CAP:-50m}            # per-training-fold hard cap (controls are light)

PROTO="$PROTO --cv_mode leave_song_out"   # base $PROTO from sweep_common.sh

# --- done-set from ALL existing runs (resume across sessions/tags) ---
$PY - <<'PYC' > /tmp/lso_done.txt
import glob, os
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_held_out_id':'?'}
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
    if c['cv_held_out_id'] in ('-1','?'): continue
    seen.add(f"{c['objective']}|{c['audio_repr']}|{c['cv_held_out_id']}")
print('\n'.join(sorted(x for x in seen if x)))
PYC
echo "[resume] folds already done: $(wc -l < /tmp/lso_done.txt)"
is_done(){ grep -qxF "$1|$2|$3" /tmp/lso_done.txt; }

run_fold(){   # $1 objective  $2 repr  $3 song  $4 tag
  local obj="$1" repr="$2" song="$3" tag="$4" key="$1|$2|$3"
  if is_done "$obj" "$repr" "$song"; then echo "  skip(done): $key"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  ($key)  elapsed $((SECONDS/60))m ====="
  echo "  TRAIN"
  timeout $CAP $PY -u main.py $PROTO --audio_repr "$repr" --objective "$obj" \
        --cv_held_out_id "$song" --training_date "$tag"
  if compgen -G "../results/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO --audio_repr "$repr" --objective "$obj" \
        --cv_held_out_id "$song" --training_date "$tag" --test_breakdown 1
    echo "$key" >> /tmp/lso_done.txt
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}

# 7 songs per class, all heard by every subject (-> 104-window test sets)
declare -A SONGS=( [vocal]="2 3 5 33 45 38 39" [drum]="7 18 59 112 145 36 115" \
                   [bass]="44 50 55 68 140 86 141" [others]="8 16 43 62 120 63 135" )
CLASSES="vocal drum bass others"
NROUND=7

# Phase 1: the two decisive raw controls (cheap) -> tighten error bars first.
# Phase 2: audio-only CLAP (heavier) -> shows even the 99%-within model collapses.
for phase in "classify_audio:raw classify_eeg:raw" "classify_audio:clap"; do
  for ((r=0;r<NROUND;r++)); do
    for cls in $CLASSES; do
      arr=(${SONGS[$cls]}); s=${arr[$r]:-}; [ -z "$s" ] && continue
      for m in $phase; do
        [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop"; break 4; }
        obj="${m%%:*}"; repr="${m#*:}"
        case "$obj" in
          classify_eeg)   tag="clf_eeg_lso_song$s" ;;
          classify_audio) tag="clf_audio_${repr}_lso_song$s" ;;
        esac
        run_fold "$obj" "$repr" "$s" "$tag"
      done
    done
  done
done

echo; echo "######## CONTROLS AGGREGATE (all LSO control folds on disk) ########"
$PY - <<'PYC'
import glob, os, pandas as pd
TASK={0:'vocal',1:'drum',2:'bass',3:'others'}
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_held_out_id':'?'}
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
    if c['cv_held_out_id'] in ('-1','?'): continue
    if not c['objective'].startswith('classify_'): continue
    df=pd.read_csv(f)
    if 'correct' not in df or 'task' not in df: continue
    byfold[(c['objective'],c['audio_repr'],c['cv_held_out_id'])]=df
groups={}
for (obj,rep,song),df in byfold.items(): groups.setdefault((obj,rep),[]).append(df)
for key in sorted(groups):
    dfs=groups[key]; alld=pd.concat(dfs); per=alld.groupby('task')['correct'].mean()
    name=f"{key[0]}/{key[1]}"
    print(f"\n===== {name}  ({len(dfs)} folds, {len(alld)} windows) =====")
    for t,v in per.items(): print(f"   held-out {TASK[int(t)]:7}: acc={v:.3f}")
    print(f"   MACRO={per.mean():.3f}  GLOBAL={alld['correct'].mean():.3f}  (chance 0.25)")
    dist={TASK[int(k)]:round(v,3) for k,v in alld['pred'].value_counts(normalize=True).items()}
    print(f"   predicts: {dist}   classes covered: {[TASK[int(t)] for t in per.index]}")
PYC
echo "######## DONE ########"
