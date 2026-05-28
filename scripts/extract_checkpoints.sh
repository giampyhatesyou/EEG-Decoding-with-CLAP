#!/bin/bash
# Extract paper checkpoints from archive/proposed_model_checkpoints.7z into checkpoints/.
# Required before running scripts/test.sh on a fresh clone.

set -e
cd "$(dirname "$0")/.."

if [ -f checkpoints/model-all0.ckpt ]; then
    echo "Checkpoints already extracted. Skipping."
    exit 0
fi

if ! python -c "import py7zr" 2>/dev/null; then
    echo "Installing py7zr..."
    pip install py7zr
fi

python -c "
import py7zr
with py7zr.SevenZipFile('archive/proposed_model_checkpoints.7z', 'r') as z:
    z.extractall('archive_tmp/')
import shutil, os
src = 'archive_tmp/proposed_model_checkpoints'
for f in os.listdir(src):
    if f.endswith('.ckpt'):
        shutil.move(os.path.join(src, f), os.path.join('checkpoints', f))
shutil.rmtree('archive_tmp')
print('Extracted:', os.listdir('checkpoints'))
"
