#!/usr/bin/env bash
# =============================================================================
# Arm A - protocol parity with the paper (Cantisani et al., WASPAA 2019).
#
# WHAT IT MEASURES: how close the COMPLETE package declared by the paper gets to
#   its 79 F1 on duets - per-subject PER-INSTRUMENT decoders, shrinkage 0.1,
#   24 mel bands at 256 Hz, lags 0-250 ms, training on the solos, decision on the
#   whole trial (~24 s), F1 metric - run over three analysis bands.
# COST       : TO BE CONFIRMED. This is the heaviest family in the repo (3 runs at
#              --target_fs 256 with --filters per_instrument, ~6 GB of design matrix
#              each); the 2026-08-11 timestamps are identical and give no duration.
#              Expect tens of minutes and 16 GB of RAM.
# BUDGET     : YES - it decides on the STEREO duos. All already spent: EXPLORATORY
#              BY CONSTRUCTION, declared as such in the summary. TRIOS untouched.
# GPU        : no.
#
# EXPECTED REFERENCE NUMBERS (if these do not come out, it did not replicate):
#   band 1-8 Hz    (armA_paper_package)   F1 micro = accuracy 0.5267   n = 150
#   band 1-40 Hz   (armA_paper_band40)    F1 micro 0.4800
#   band 0.2-40 Hz (armA_paper_band02_40) F1 micro 0.5400
#   null 0.500 on the duo; r(attended) 0.016-0.025 against the paper's median 0.119
#   OUTCOME: the 79 DOES NOT REPRODUCE from any package faithful to the declared
#   methods. n = 150 and not 154: 4 trials drop under --filters per_instrument
#   because subject 0007 has no Bo/Fh solos - known and declared, not a bug.
#
# CANARY     : the arm A criterion declares none (no gate, no pre-registered claim:
#              it is a protocol probe). The control is the reproduction of the three
#              F1 values above. Do NOT add a --self_test "to have a control": it has
#              no pinned expected value and would produce a new number.
# PRE-REGISTRATION: none (protocol probe).
# PROVENANCE : runs/results/armA_paper_{package,band40,band02_40}/ and
#              docs/provenance/2026-08-17_armA_paper_*_RESULT_*
#
# SAME/DIFF-MELODY SPLIT (docs/provenance/2026-08-11_armA_same_diff_melody.txt,
#    same 46/89 vs diff 33/61 at 1-8 Hz; same 51/89 vs diff 30/61 at 0.2-40 Hz):
#    NOT reproducible from here.
# TO BE CONFIRMED: no file in the repo produces that split - it was computed ad hoc
#    from madeeg_records.csv + madeeg_sequences_raw.yaml and the script was not saved
#    (verified by grepping src/, scripts/, sweeps/ on 2026-08-12).
#    It has to be rewritten, not guessed.
# =============================================================================
set -euo pipefail

PY=${PY:-/opt/miniconda3/bin/python}          # environment trap #1: NEVER bare `python`
MADEEG_DIR=${MADEEG_DIR:-$HOME/madeeg}        # the dataset is NOT in the repo
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
TAG=${TAG:-repl_}                             # pinned records are never overwritten

cd "$REPO"

# The paper package. --eeg_clean none is imposed by the preprocessed release used
# for the solos; --test_eeg raw and --spatial stereo are the paper protocol.
PKG=(--madeeg_dir "$MADEEG_DIR" --train_on raw_solos --test_eeg raw --spatial stereo
     --ensemble duo --target mel --n_mels 24 --target_fs 256 --lags_ms 250
     --estimator shrinkage --shrinkage_lambda 0.1 --filters per_instrument
     --eeg_clean none --seed 42 --cv_folds 5)

run() {  # run <name> <band_low> <band_high>
  echo; echo "--- $1 : band $2-$3 Hz ---"
  "$PY" src/madeeg_reconstruction.py "${PKG[@]}" --band_low "$2" --band_high "$3" \
        --training_date "${TAG}$1"
}

run armA_paper_package  1   8
run armA_paper_band40   1   40
run armA_paper_band02_40 0.2 40

echo
echo "=== REPLICATION OUTCOME (compare with the REFERENCE NUMBERS in the header) ==="
for R in armA_paper_package armA_paper_band40 armA_paper_band02_40; do
  printf "%-22s " "$R"
  grep -h "^F1 over the attended" "$REPO/runs/results/${TAG}$R/madeeg_summary.txt"
done
echo
echo "pinned records (2026-08-11): runs/results/armA_paper_*"
echo "the paper (WASPAA 2019 Tab. 1, duets): AE 58 / MAG 74 / MEL 79"
