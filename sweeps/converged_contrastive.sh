#!/usr/bin/env bash
# Option B: a FEW contrastive folds trained to CONVERGENCE (not the 80m-truncated sweep).
# Writes into clearly separated conv_* tags so it never collides with the earlier
# under-trained runs; the aggregate at the end reads only conv_* dirs.
# 2 GPUs: launch twice with SHARD=0 and SHARD=1 (NSHARD=2). Resumable.
set -uo pipefail
source "$(dirname "$0")/sweep_common.sh"   # conda env + $PY + $PROTO
cd "$(dirname "$0")/../src"
NSHARD=${NSHARD:-1}; SHARD=${SHARD:-0}
CAP=${CAP:-360m}                 # 6h cap: room for >=50 epochs (the protocol min) to converge
BUDGET=${BUDGET:-86400}          # keep launching folds back-to-back until the allocation ends.
                                 # A fold cut mid-training (no test yet) is simply re-done next
                                 # session -- resumable, no corruption, just some wasted compute.
echo "[env] $(date) env=${CONDA_DEFAULT_ENV:-none} CUDA=${CUDA_VISIBLE_DEVICES:-unset} SHARD=$SHARD/$NSHARD CAP=$CAP"
$PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC

# $PROTO comes from sweep_common.sh (no --cv_mode; this sweep passes --cv_mode per fold)

done_test(){ compgen -G "../results/$1/nmed-CL-*/version_*/test_breakdown_summary.txt" >/dev/null; }

run_fold(){   # obj audio eeg cvmode held tag
  local obj="$1" arepr="$2" erepr="$3" cvm="$4" held="$5" tag="$6"
  if done_test "$tag"; then echo "  skip(done): $tag"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  ($obj|$arepr|$erepr|$cvm|$held)  elapsed $((SECONDS/60))m ====="
  local objflag=""; [ "$obj" != "contrastive" ] && objflag="--objective $obj"
  local eegflag=""; [ "$erepr" != "raw" ] && eegflag="--eeg_repr $erepr"
  echo "  TRAIN (CAP=$CAP, to convergence)"
  timeout $CAP $PY -u main.py $PROTO --cv_mode "$cvm" --cv_held_out_id "$held" \
        --audio_repr "$arepr" $eegflag $objflag --training_date "$tag"
  if compgen -G "../results/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO --cv_mode "$cvm" --cv_held_out_id "$held" \
        --audio_repr "$arepr" $eegflag $objflag --training_date "$tag" --test_breakdown 1
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}

# Ordered by value: paired raw-vs-spectra on the hint class (others) first, then the
# clap+spectra within (overall comparison), then the other classes paired.
FOLDS=(
  "contrastive:clap:spectra:within:-1:conv_ct_clap_spectra_within"
  "contrastive:raw:raw:leave_song_out:16:conv_ct_raw_lso_song16"
  "contrastive:raw:spectra:leave_song_out:16:conv_spectra_ct_raw_lso_song16"
  "contrastive:raw:raw:leave_song_out:44:conv_ct_raw_lso_song44"
  "contrastive:raw:spectra:leave_song_out:44:conv_spectra_ct_raw_lso_song44"
  "contrastive:raw:raw:leave_song_out:7:conv_ct_raw_lso_song7"
  "contrastive:raw:spectra:leave_song_out:7:conv_spectra_ct_raw_lso_song7"
  "contrastive:raw:raw:leave_song_out:2:conv_ct_raw_lso_song2"
  "contrastive:raw:spectra:leave_song_out:2:conv_spectra_ct_raw_lso_song2"
)

idx=0
for spec in "${FOLDS[@]}"; do
  if [ $((idx % NSHARD)) -eq "$SHARD" ]; then
    [ $SECONDS -ge $BUDGET ] && { echo "[budget] $((SECONDS/60))m reached -> stop"; break; }
    IFS=: read -r o a e cv h tag <<< "$spec"
    run_fold "$o" "$a" "$e" "$cv" "$h" "$tag"
  fi
  idx=$((idx+1))
done

echo; echo "######## CONVERGED AGGREGATE (conv_* only) ########"
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
for f in glob.glob('../results/conv_*/nmed-CL-*/version_*/test_records.csv'):
    c=hp(os.path.dirname(f))
    try: df=pd.read_csv(f)
    except Exception: continue
    if 'correct' not in df or 'task' not in df or len(df)==0: continue
    byfold[(c['audio_repr'],c['eeg_repr'],c['cv_mode'],c['cv_held_out_id'])]=df
groups={}
for (a,e,cv,h),df in byfold.items(): groups.setdefault((a,e,cv),[]).append(df)
print("{:34}{:>6}{:>8}{:>8}   {}".format("audio/eeg/eval","folds","MACRO","GLOBAL","per-class[v d b o]"))
for k in sorted(groups):
    dfs=groups[k]; alld=pd.concat(dfs); per=alld.groupby('task')['correct'].mean()
    pc=" ".join(("%.2f"%per[t]) if t in per.index else " -  " for t in range(4))
    print("{:34}{:>6}{:>8.3f}{:>8.3f}   [{}]".format("/".join(k),len(dfs),per.mean(),alld['correct'].mean(),pc))
print("converged contrastive (CAP high). Compare eeg raw vs spectra at eval=leave_song_out.")
PYC
echo "######## DONE ########"
