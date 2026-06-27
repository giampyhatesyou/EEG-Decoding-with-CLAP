#!/usr/bin/env bash
# CHANGED(baseline): NEW FILE — fetch the MAD-EEG dataset (Cantisani et al., SMM 2019;
#   Zenodo 4537751, CC-BY-SA) for the *positive* thesis arm (identifiable AAD: the same
#   mixture is attended with different target instruments -> within-stimulus contrast).
#   We only need the preprocessed EEG+sources HDF5 and its YAML; the isolated sources
#   ('soli') are inside the HDF5, so stimuli.zip is NOT required for reconstruction.
#
# Usage:  MADEEG_DIR=/path/to/madeeg bash madeeg_setup.sh
set -euo pipefail
DIR=${MADEEG_DIR:-"$(cd "$(dirname "$0")" && pwd)/madeeg"}
BASE="https://zenodo.org/records/4537751/files"
mkdir -p "$DIR"; cd "$DIR"
echo "[madeeg] downloading into $DIR"
# 3.7 GB EEG+sources + 154 kB metadata. Optional extras commented out.
for f in madeeg_preprocessed.hdf5 madeeg_preprocessed.yaml; do
  if [ -f "$f" ]; then echo "  have $f"; else echo "  get  $f"; wget -q --show-progress "$BASE/$f?download=1" -O "$f"; fi
done
# optional: behavioural ratings + raw + stimuli wavs (not needed for reconstruction)
# wget -q "$BASE/behavioural_data.xlsx?download=1" -O behavioural_data.xlsx
echo "[madeeg] done. Verify schema with:"
echo "  cd src && python madeeg_reconstruction.py --madeeg_dir \"$DIR\" --inspect 3"
