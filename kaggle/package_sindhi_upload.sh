#!/bin/bash
# Build the folder to upload to Kaggle as a Dataset for the Sindhi training run.
#
# Only the tokenized ids and the tokenizer are needed — not the 3.3 GB of source text.
# The repository itself is cloned by the notebook from GitHub, so it is not included here.
set -e
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/sindhi-tokens}"
mkdir -p "$OUT"

cp sindhi/data/tokenized/train.bin "$OUT/"
cp sindhi/data/tokenized/val.bin   "$OUT/"
cp sindhi/data/tokenized/test.bin  "$OUT/"
cp sindhi/tokenizer/sindhi_unigram_16000.model "$OUT/"
cp sindhi/tokenizer/sindhi_unigram_16000.vocab "$OUT/"
cp sindhi/data/tokenized/tokenization_stats.json "$OUT/" 2>/dev/null || true

echo "Upload bundle ready at: $OUT"
du -sh "$OUT"
ls -la "$OUT"
echo
echo "Next: create a Kaggle Dataset from this folder (New Dataset -> upload),"
echo "name it so the mount path is /kaggle/input/sindhi-tokens"
