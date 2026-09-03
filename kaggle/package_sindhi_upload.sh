#!/bin/bash
# Build the Kaggle upload archive for the Sindhi (Model L) training run.
#
# Only the tokenized ids and the tokenizer are needed. The 3.3 GB of source text stays
# local, and the repository itself is cloned from GitHub by the notebook, so neither is
# included here.
#
# The output is a .zip because Kaggle extracts archives automatically on upload, and the
# token stream compresses to about 60% of its raw size — 1.3 GB becomes ~730 MB. Files are
# added at the archive root so they extract flat to /kaggle/input/<dataset>/train.bin
# rather than into a nested directory the training script would not find.
#
#   bash kaggle/package_sindhi_upload.sh [output_dir]
set -e
cd "$(dirname "$0")/.."

OUT="${1:-$HOME/sindhi-kaggle-upload}"
STAGE="$OUT/files"
ARCHIVE="$OUT/sindhi-tokens.zip"

mkdir -p "$STAGE"
cp sindhi/data/tokenized/train.bin "$STAGE/"
cp sindhi/data/tokenized/val.bin   "$STAGE/"
cp sindhi/data/tokenized/test.bin  "$STAGE/"
cp sindhi/tokenizer/sindhi_unigram_16000.model "$STAGE/"
cp sindhi/tokenizer/sindhi_unigram_16000.vocab "$STAGE/"
cp sindhi/data/tokenized/tokenization_stats.json "$STAGE/" 2>/dev/null || true

# -1 (fastest): measured compression is identical at higher levels for uint16 token ids,
# so extra CPU time buys nothing.
rm -f "$ARCHIVE"
( cd "$STAGE" && zip -1 -q "$ARCHIVE" \
    train.bin val.bin test.bin \
    sindhi_unigram_16000.model sindhi_unigram_16000.vocab tokenization_stats.json )

unzip -t "$ARCHIVE" > /dev/null || { echo "ERROR: archive failed its CRC check"; exit 1; }

echo "Archive ready: $ARCHIVE"
du -h "$ARCHIVE" | awk '{print "  size: "$1}'
echo "  contents (must be flat, no nested directory):"
unzip -l "$ARCHIVE" | sed -n '4,10p' | sed 's/^/    /'
echo
echo "Next: Kaggle -> Datasets -> New Dataset -> upload this .zip"
echo "      Title it 'sindhi-tokens' so it mounts at /kaggle/input/sindhi-tokens"
echo "      Kaggle extracts the archive automatically; the .bin files land at the root."
