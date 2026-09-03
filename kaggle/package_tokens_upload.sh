#!/bin/bash
# Build the Kaggle token-data archive for one language.
#
#   bash kaggle/package_tokens_upload.sh hindi   [output_dir]
#   bash kaggle/package_tokens_upload.sh sindhi  [output_dir]
#
# Only the tokenized ids and that language's tokenizer are included. The source text stays
# local and the code ships separately (package_code_upload.sh), so a code change costs a
# 56 KB re-upload rather than a gigabyte.
#
# Output is a .zip: Kaggle extracts archives automatically, and uint16 token ids compress
# to about 60% of raw size. Files are added at the archive root so they extract flat.
set -e
cd "$(dirname "$0")/.."

LANG="${1:?usage: package_tokens_upload.sh <hindi|sindhi> [output_dir]}"
case "$LANG" in hindi|sindhi) ;; *) echo "language must be hindi or sindhi"; exit 1 ;; esac

OUT="${2:-$HOME/kaggle-upload}"
STAGE="$OUT/${LANG}-files"
ARCHIVE="$OUT/${LANG}-tokens.zip"

mkdir -p "$STAGE"
for f in train.bin val.bin test.bin tokenization_stats.json; do
  cp "$LANG/data/tokenized/$f" "$STAGE/" 2>/dev/null || true
done
cp "$LANG/tokenizer/${LANG}_unigram_16000.model" "$STAGE/"
cp "$LANG/tokenizer/${LANG}_unigram_16000.vocab" "$STAGE/"

rm -f "$ARCHIVE"
( cd "$STAGE" && zip -1 -q "$ARCHIVE" ./* )
unzip -t "$ARCHIVE" > /dev/null || { echo "ERROR: archive failed its CRC check"; exit 1; }
rm -rf "$STAGE"

echo "Archive ready: $ARCHIVE"
du -h "$ARCHIVE" | awk '{print "  size: "$1}'
echo "  contents (must be flat):"
unzip -l "$ARCHIVE" | sed -n '4,11p' | sed 's/^/    /'
echo
echo "Next: Kaggle -> Datasets -> New Dataset -> upload this .zip"
echo "      Title it '${LANG}-tokens'"
