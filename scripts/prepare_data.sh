#!/bin/bash
# Download + prepare RSNA Knee data
# Usage: bash scripts/prepare_data.sh
set -e
ROOT="data/raw"
mkdir -p $ROOT
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Downloading RSNA Knee via kaggle API..."
  kaggle competitions download -c rsna-knee-abnormality-detection -p $ROOT --unzip
else
  echo "No kaggle.json found at ~/.kaggle/kaggle.json"
  echo "Manual: kaggle competitions download -c rsna-knee-abnormality-detection -p data/raw --unzip"
  echo "Or on Kaggle: /kaggle/input/rsna-knee-abnormality-detection auto-mounted"
fi
echo "Done. Check $ROOT"
ls -lh $ROOT | head -n 50
