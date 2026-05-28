#!/bin/bash
#SBATCH --job-name=eeg_audit_sweep
#SBATCH --partition=edu-thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --open-mode=append
#
# Robust audit sweep: runs the missing CLAP folds in priority order so that
# if the 8h grant is exhausted mid-way, the scientifically most important
# folds are already done. Submit with:
#
#     cd ~/EEG-Attention-decoding-with-CLAP
#     sbatch scripts/sweep_audit.sbatch
#
# Monitor with:
#     squeue -u $USER
#     tail -f logs/sweep_<jobid>.out
#
# Each fold trains from scratch (~40 min on one L40S for the CLAP variant)
# and writes to results/<tag>_<cv_mode>_<id>/. Folds are independent: a
# failure in one does not stop the others (each call is guarded with || true).

set +e   # do NOT abort the whole sweep if a single fold fails

cd "${SLURM_SUBMIT_DIR:-$HOME/EEG-Attention-decoding-with-CLAP}"
echo "[sweep] start $(date)  cwd=$(pwd)  job=$SLURM_JOB_ID"

run () {
    echo ""
    echo "============================================================"
    echo "[sweep $(date +%H:%M:%S)] $*"
    echo "============================================================"
    AUDIO_REPR=clap bash scripts/train_cv.sh "$@" || echo "[sweep] FOLD FAILED: $*"
}

# ---- Priority 1: leave-song-out across all four target tasks (CLAP) ----
# This is the gap that matters most: so far LSO was only tested on drum
# songs (36, 145). We now cover vocal / bass / others to see whether the
# collapse is task-dependent, and in particular whether a vocal-target
# held-out song "succeeds" purely because the model's OOD default is vocal.
run leave_song_out 3   lso_sweep    # vocal  (CRITICAL: target == model default?)
run leave_song_out 44  lso_sweep    # bass
run leave_song_out 16  lso_sweep    # others
run leave_song_out 5   lso_sweep    # vocal  (variance on the critical case)
run leave_song_out 59  lso_sweep    # drum   (third drum point for variance)

# ---- Priority 2: leave-one-subject-out variance (CLAP) ----
# We only had subject 0 (0.9363). Add subjects 1-5 to constrain the
# cross-subject claim with a proper spread (the paper used 3 subjects).
run leave_subject_out 1 cv_sweep
run leave_subject_out 2 cv_sweep
run leave_subject_out 3 cv_sweep
run leave_subject_out 4 cv_sweep
run leave_subject_out 5 cv_sweep

echo ""
echo "[sweep] all done $(date)"
echo "[sweep] aggregate with:"
echo "  grep -H 'global accuracy (records, all)' results/lso_sweep_leave_song_out_*/nmed-CL-*/version_*/test_breakdown_summary.txt"
echo "  grep -H 'global accuracy (records, all)' results/cv_sweep_leave_subject_out_*/nmed-CL-*/version_*/test_breakdown_summary.txt"
