#!/usr/bin/env bash
# Shared core of the two cross-validation sweeps (sweep_song_out.sh, sweep_subject_out.sh).
# Holds everything the two must agree on, in exactly one place: the conda env, the fixed
# Akama protocol, where run artifacts go, and the resume logic.
#
# Source it BEFORE the per-script `cd` (it has no cwd dependency, and `cd`-then-source
# breaks when $0 is a relative path).
#
# Provides:
#   $PY       - python interpreter (override with PY=... in the environment)
#   $PROTO    - the fixed protocol flags every sweep passes verbatim. Does NOT include
#               --cv_mode: each sweep appends its own.
#   $RESULTS  - run-output dir (mirrors src/utils/paths.py: EEG_LOG_DIR > runs/results,
#               with a fallback to the legacy root-level results/).
#   echo_env / build_done_set / is_done / run_fold  - the resumable train+test loop.
# Side effect: activates the eeg_attention conda env (no-op if already active).

if [ "${CONDA_DEFAULT_ENV:-}" != "eeg_attention" ]; then
  for c in /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
           "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && . "$c" && break
  done
  conda activate eeg_attention 2>/dev/null || true
fi
# `conda activate` is a shell function: in a non-interactive shell (nohup, sbatch, cron)
# the block above is a no-op, and a bare `python` then resolves to a system interpreter
# with no torch -- which surfaces as ModuleNotFoundError several minutes into a run, not
# at launch. Prefer the env's interpreter by path: it needs no activation. An explicit
# PY=... in the environment still wins.
if [ -z "${PY:-}" ]; then
  _env_py="$HOME/.conda/envs/eeg_attention/bin/python"
  [ -x "$_env_py" ] && PY="$_env_py" || PY=python
fi

PROTO="--dataset preprocessing_eegmusic --test_dataset preprocessing_eegmusic_test \
--max_epochs 1000 --batch_size 8 --eeg_length 768 --loss_function clip_loss \
--eeg_normalization MetaAI --clamp_value 20 --learning_rate 0.003 --supervised 1 \
--dim_reduction 1 --split_seed 42 --detach_z_c 0 --window_size 1280 --stride 256 \
--test_window_size 768 --test_stride 256 --start_position 0 --key all \
--attention_values 4 5 --devices 1 --shifting_time 0 --seed 42"

# --- where run artifacts live -------------------------------------------------
# Same three-way resolution as src/utils/paths.py, so shell and python never disagree
# about which directory holds the folds. The legacy branch matters: the cluster
# checkout has ~150 finished folds under the old path, and pointing a resumable sweep
# at an empty new one would silently re-run all of them.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${EEG_LOG_DIR:-}" ]; then
  RESULTS="$EEG_LOG_DIR"
elif [ ! -d "$_ROOT/runs/results" ] && [ -d "$_ROOT/results" ]; then
  RESULTS="$_ROOT/results"
  echo "[paths] legacy layout: using $RESULTS. Migrate with: mkdir -p $_ROOT/runs && mv $_ROOT/results $_ROOT/runs/results"
else
  RESULTS="$_ROOT/runs/results"
fi
mkdir -p "$RESULTS"

echo_env(){                       # one banner, identical in both sweeps
  echo "[env] $(date) python=$(command -v $PY) env=${CONDA_DEFAULT_ENV:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[env] results=$RESULTS"
  $PY - <<'PYC'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PYC
}

# --- resume -------------------------------------------------------------------
# The done-set key is the FULL identity of a fold: objective, audio_repr, eeg_repr and
# held-out id, within one cv_mode. Dropping a field makes two different runs collide,
# which has bitten this project twice: cv_mode (fixed in 10cdc49 -- a leave-subject-out
# fold shadowed a leave-song-out one with the same numeric id) and eeg_repr (found
# 2026-07-26 -- a `spectra` fold shadowed its `raw` twin). Keep every field.
# sweeps/report.py builds the same key, so "done" and "reported" cannot drift apart.
build_done_set(){                 # $1 cv_mode  $2 done-file
  RESULTS="$RESULTS" CV="$1" $PY - > "$2" <<'PYC'
import glob, os
def hp(d):
    c = {'objective': 'contrastive', 'audio_repr': '?', 'eeg_repr': 'raw',
         'cv_held_out_id': '?', 'cv_mode': '?'}
    try:
        for l in open(os.path.join(d, 'hparams.yaml')):
            s = l.strip()
            for k in c:
                if s.startswith(k + ':'):
                    c[k] = s.split(':', 1)[1].strip().strip('\'"')
    except Exception:
        pass
    return c
seen = set()
for f in glob.glob(os.path.join(os.environ['RESULTS'], '*/nmed-CL-*/version_*/test_records.csv')):
    c = hp(os.path.dirname(f))
    if c['cv_held_out_id'] in ('-1', '?') or c['cv_mode'] != os.environ['CV']:
        continue
    seen.add("{objective}|{audio_repr}|{eeg_repr}|{cv_held_out_id}".format(**c))
print('\n'.join(sorted(x for x in seen if x)))
PYC
  echo "[resume] folds already done: $(wc -l < "$2")"
}

is_done(){ grep -qxF "$1|$2|$3|$4" "$DONE_FILE"; }   # objective audio_repr eeg_repr held

run_fold(){   # $1 objective  $2 audio_repr  $3 eeg_repr  $4 held-out id  $5 tag
  local obj="$1" repr="$2" eeg="$3" held="$4" tag="$5" key
  key="$obj|$repr|$eeg|$held"
  if is_done "$obj" "$repr" "$eeg" "$held"; then echo "  skip(done): $key"; return 0; fi
  echo; echo "===== [$(date +%H:%M)] $tag  ($key)  elapsed $((SECONDS/60))m ====="
  local objflag=""; [ "$obj" != "contrastive" ] && objflag="--objective $obj"
  local eegflag=""; [ "$eeg" != "raw" ] && eegflag="--eeg_repr $eeg"
  echo "  TRAIN"
  timeout $CAP $PY -u main.py $PROTO $CV_FLAG --audio_repr "$repr" $objflag $eegflag \
        --cv_held_out_id "$held" --training_date "$tag"
  if compgen -G "$RESULTS/$tag/nmed-CL-*/version_*/checkpoints/best-checkpoint.ckpt" >/dev/null; then
    echo "  TEST"
    $PY -u checkpoint_test.py $PROTO $CV_FLAG --audio_repr "$repr" $objflag $eegflag \
        --cv_held_out_id "$held" --training_date "$tag" --test_breakdown 1
    echo "$key" >> "$DONE_FILE"
  else echo "  no checkpoint (timeout/fail) -> skip test"; fi
}
