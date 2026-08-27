#!/usr/bin/env bash
# =============================================================================
# Chapter 1 - LEAVE-SONG-OUT: the collapse on unseen songs, and the three
#             baselines that sit next to it.
#
# WHAT IT MEASURES: whether the contrastive model holds up when the test song was
#   never seen in training. 20 folds (one song held out at a time) x 4 models.
#   This is THE Chapter 1 number: 0.865 within against 0.268 leave-song-out.
# DATE       : closed 20/20 on 2026-07-26.
# QUESTION   : "does the model hold up on songs it has never seen?"
# NULL       : 0.25 (4 fixed instrument slots, assumed from the design).
# PRE-REGISTERED THRESHOLD: none. There was no test, there was a reproduction, and
#              the 20-fold design was fixed by the paper. The verdict is descriptive.
# VERDICT    : closed 20/20 folds - and the number is 0.268, i.e. AT CHANCE.
# COST       : GPU, ~80 min per fold x 20 folds x 4 models. The sweep is
#              RESUMABLE: relaunching it skips the folds already done.
#              On a local machine this script does NOT train: it verifies the pins
#              and prints the commands.
# BUDGET     : n/a (held-out-look accounting belongs to Chapter 2, not this dataset).
# GPU        : YES to redo the folds. NO to re-read the numbers already pinned.
#
# EXPECTED REFERENCE NUMBERS (RESULTS.md song_out section, 20 folds each):
#   contrastive_clap  MACRO 0.268   per-class [v d b o] 0.57 0.07 0.35 0.08
#   eeg_only          MACRO 0.247
#   audio_only_raw    MACRO 0.181
#   contrastive_raw   MACRO 0.142
#
# THREE MANDATORY CAVEATS, all three to be written next to the number:
#   (a) MIXED VINTAGE on the contrastive_clap row: 6 folds come from 2026-05 code
#       and 14 from 2026-07 code. That is TWO CODE ERAS inside one mean. The
#       homogeneous 14-fold table gives 0.258. Which of the two goes into the
#       write-up has been an OPEN decision since 2026-07-26.
#   (b) BELOW CHANCE IS NOT A MEASUREMENT ERROR: 0.142 and 0.181 are
#       prior-following on single-class folds. Say it that way, not "the model
#       gets it wrong".
#   (c) `contrastive_raw` went 0.142 -> 0.154 -> 0.142 because of an arbitrary
#       dedup, before the pins existed. That is exactly the incident that produced
#       `results_manifest.tsv`. Today's number is 0.142.
#
# Song 36 (drum) is NOT pinned: outside the 5x4 design, May vintage (manifest,
#    row "Deliberately NOT pinned"). 20 folds means 20, not 21.
#
# CANARY     : (1) `src/modules/clip_loss.py` (the InfoNCE arithmetic);
#              (2) `src/run.py --selftest` (23 flags: the sweep and replicate.sh
#                  must pass the SAME $PROTO);
#              (3) `report.py` checks every pin against the run's hparams.yaml -
#                  in particular the fold key includes `cv_mode` and `eeg_repr`,
#                  because twice a fold shadowed another one (10cdc49: a
#                  leave_subject_out shadowed a leave_song_out with the same id;
#                  2026-07-26: a `spectra` fold shadowed its `raw` twin).
# PRE-REGISTRATION: none (reproduction).
# PROVENANCE : RESULTS.md song_out section <- results_manifest.tsv (80 pins:
#              4 models x 20 songs)
# =============================================================================
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/opt/miniconda3/bin/python}
PY_TORCH=${PY_TORCH:-}

cd "$REPO"

echo "=== CANARY 1 - src/modules/clip_loss.py (synthetic, no data) ==="
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
  echo "  NOT CROSSED (no torch). PY_TORCH=... bash \$0"
fi

echo
echo "=== CANARY 2 - src/run.py --selftest ==="
"$PY" src/run.py --selftest

echo
echo "=== THE PINNED NUMBERS (no training: the 80 folds are re-read) ==="
# report.py --check exits 1 if even ONE pin is unverifiable. With `set -e` + pipefail
# a pipe would kill the script BEFORE the warning below - i.e. exactly in the case the
# warning exists to explain. Capture the exit code and declare it.
RC=0; CHECK_OUT=$("$PY" sweeps/report.py --check 2>&1) || RC=$?
echo "$CHECK_OUT" | sed -n '/^## song_out/,/^## subject_out/p'
echo "if a row is missing above, its run directory is not on this machine:"
echo "   runs/ is gitignored and the folds live only where they were run."
if [ "$RC" != 0 ]; then
  echo "report.py --check exited $RC - unverifiable pins. The [BAD PIN] lines:"
  echo "$CHECK_OUT" | grep "^\[BAD PIN\]" || true
fi

cat <<'EOF'

###############################################################################
# REDOING THE FOLDS REQUIRES A GPU
###############################################################################
# The sweep is RESUMABLE: the "already done" key is the FULL identity of the fold
# (objective | audio_repr | eeg_repr | held-out id, within a cv_mode).
# Relaunching after an interruption resumes where it stopped, not from scratch.

cd $REPO
python src/run.py sweep --cv song                 # 4 models x 20 songs

# explicit equivalent, when the per-fold cap or the GPU has to be controlled:
CUDA_VISIBLE_DEVICES=0 CAP=80m bash sweeps/sweep_song_out.sh

# `conda activate` does NOT work in non-interactive shells (nohup/sbatch/cron):
#    it becomes a no-op and bare `python` is the system one, WITHOUT torch - and the
#    error surfaces minutes after launch, not immediately. sweep_common.sh resolves
#    the interpreter BY PATH for exactly this reason. An explicit PY=... always wins.
#
# On the cluster there is no tmux: use nohup, so the web session can drop without
# taking the run with it.
cd $REPO && nohup python src/run.py sweep --cv song > runs/logs/lso_song.log 2>&1 &

# When the folds are done: pin and re-render. report.py refuses to write the table
# if even one pin disagrees with its hparams.yaml.
python src/run.py report
EOF
