#!/usr/bin/env bash
# Comprehensive, RESUMABLE leave-song-out sweep across 4 model types.
#   models : contrastive(raw), contrastive(clap), audio-only(classify_audio raw),
#            eeg-only(classify_eeg)
#   songs  : 5 per class (vocal/drum/bass/others), all heard by every subject
# Resume  : at start it scans results/ and SKIPS any (objective,audio_repr,song)
#           already tested (any tag) -> re-running continues where it stopped.
# Budget  : stops LAUNCHING new folds after BUDGET seconds; each train capped by CAP.
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"   # conda env + $PY + $PROTO
cd "$(dirname "$0")/../src"   # entrypoints main.py/checkpoint_test.py live in src/
echo "[env] $(date) python=$(command -v $PY) env=${CONDA_DEFAULT_ENV:-none}"
$PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC

BUDGET=${BUDGET:-23400}     # 6.5h launch cutoff (override: BUDGET=... )
CAP=${CAP:-80m}            # per-training-fold hard cap
NSHARD=${NSHARD:-1}        # in quante istanze parallele dividere i fold
SHARD=${SHARD:-0}         # quale shard e questa istanza (0..NSHARD-1)
DONE_FILE=${DONE_FILE:-/tmp/lso_done.txt}   # resume file (unico per istanza quando shardi)

PROTO="$PROTO --cv_mode leave_song_out"   # base $PROTO from sweep_common.sh

# --- build done-set from ALL existing runs (resume across sessions and tags) ---
$PY - <<'PYC' > "$DONE_FILE"
import glob, os
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_held_out_id':'?','cv_mode':'?'}
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
    if c['cv_mode']!='leave_song_out': continue   # subject-out/within share ids with songs -> don't skip a real LSO fold
    seen.add(f"{c['objective']}|{c['audio_repr']}|{c['cv_held_out_id']}")
print('\n'.join(sorted(x for x in seen if x)))
PYC
echo "[resume] folds already done: $(wc -l < "$DONE_FILE")"
is_done(){ grep -qxF "$1|$2|$3" "$DONE_FILE"; }

run_fold(){   # $1 objective  $2 repr  $3 song  $4 tag
  local obj="$1" repr="$2" song="$3" tag="$4" key
  key="$obj|$repr|$song"
  if is_done "$obj" "$repr" "$song"; then echo "  skip(done): $key"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  ($key)  elapsed $((SECONDS/60))m ====="
  local objflag=""; [ "$obj" != "contrastive" ] && objflag="--objective $obj"
  echo "  TRAIN"
  timeout $CAP $PY -u main.py $PROTO --audio_repr "$repr" $objflag \
        --cv_held_out_id "$song" --training_date "$tag"
  if compgen -G "../results/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO --audio_repr "$repr" $objflag \
        --cv_held_out_id "$song" --training_date "$tag" --test_breakdown 1
    echo "$key" >> "$DONE_FILE"
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}

# 5 songs per class, all heard by every subject
declare -A SONGS=( [vocal]="2 3 5 33 45" [drum]="7 18 59 112 145" \
                   [bass]="44 50 55 68 140" [others]="8 16 43 62 120" )
CLASSES="vocal drum bass others"
# model priority (highest marginal value first): contrastive raw, contrastive clap,
# audio-only, eeg-only.  fields = objective:repr:tagprefix
MODELS="contrastive:raw:ct_raw contrastive:clap:ct_clap classify_audio:raw:clf_audio_raw classify_eeg:raw:clf_eeg"
NROUND=5

declare -A MC=()   # contatore fold per-modello, per il bilanciamento shard
for ((r=0;r<NROUND;r++)); do
  for cls in $CLASSES; do
    arr=(${SONGS[$cls]}); s=${arr[$r]:-}; [ -z "$s" ] && continue
    for m in $MODELS; do
      [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop launching"; break 3; }
      _mi=${MC[$m]:-0}; MC[$m]=$((_mi+1)); (( _mi % NSHARD != SHARD )) && continue   # solo questo shard
      obj="${m%%:*}"; rest="${m#*:}"; repr="${rest%%:*}"
      case "$obj" in
        classify_eeg)   tag="clf_eeg_lso_song$s" ;;
        classify_audio) tag="clf_audio_raw_lso_song$s" ;;
        *)              tag="${rest#*:}_lso_song$s" ;;
      esac
      run_fold "$obj" "$repr" "$s" "$tag"
    done
  done
done

echo; echo "######## AGGREGATE (all leave-song-out folds on disk) ########"
$PY - <<'PYC'
import glob, os, pandas as pd
TASK={0:'vocal',1:'drum',2:'bass',3:'others'}
def hp(d):
    c={'objective':'contrastive','audio_repr':'?','cv_held_out_id':'?','cv_mode':'?'}
    try:
        for l in open(os.path.join(d,'hparams.yaml')):
            s=l.strip()
            for k in c:
                if s.startswith(k+':'): c[k]=s.split(':',1)[1].strip().strip('\'"')
    except Exception: pass
    return c
byfold={}   # dedup by (obj,repr,song) -> df
for f in glob.glob('../results/*/nmed-CL-*/version_*/test_records.csv'):
    c=hp(os.path.dirname(f))
    if c['cv_held_out_id'] in ('-1','?'): continue
    if c['cv_mode']!='leave_song_out': continue   # keep this table leave-song-out only (matches aggregate_lso.py)
    df=pd.read_csv(f)
    if 'correct' not in df or 'task' not in df: continue
    byfold[(c['objective'],c['audio_repr'],c['cv_held_out_id'])]=df
groups={}
for (obj,repr_,song),df in byfold.items():
    groups.setdefault((obj,repr_),[]).append(df)
print(f"{'model':26} {'folds':>5} {'classes':>7} {'MACRO':>7} {'GLOBAL':>7}")
for key in sorted(groups):
    dfs=groups[key]; alld=pd.concat(dfs); per=alld.groupby('task')['correct'].mean()
    print(f"{key[0]+'/'+key[1]:26} {len(dfs):>5} {len(per):>7} {per.mean():>7.3f} {alld['correct'].mean():>7.3f}")
print("chance=0.25 ; MACRO is the honest metric (needs songs from several classes to be valid)")
PYC
echo "######## DONE ########"
