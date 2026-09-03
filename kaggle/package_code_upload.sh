#!/bin/bash
# Build the code archive to upload to Kaggle as a second, small Dataset.
#
# The repository is private (institute-owned), so the Kaggle notebook cannot git clone it.
# The training code is only ~190 KB, so it ships as its own Dataset instead.
#
# Keeping code and data in SEPARATE datasets is deliberate: the 734 MB token archive never
# changes, while the code will change as later phases add evaluation and analysis. Bundling
# them together would mean re-uploading 734 MB every time a line of Python changes.
#
#   bash kaggle/package_code_upload.sh [output_dir]
set -e
cd "$(dirname "$0")/.."

OUT="${1:-$HOME/sindhi-kaggle-upload}"
STAGE="$OUT/code"
ARCHIVE="$OUT/lma-code.zip"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# Only what the trainer imports. Tests are included so the model can be verified on Kaggle
# before committing hours of GPU time to it.
for d in common/model common/data common/train; do
  mkdir -p "$STAGE/$d"
  rsync -a --exclude='__pycache__' "$d/" "$STAGE/$d/"
done
cp common/__init__.py "$STAGE/common/"
# Both configs: one code dataset serves the Hindi and the Sindhi run.
for lang in hindi sindhi; do
  mkdir -p "$STAGE/$lang/configs"
  cp "$lang/configs/model_config.json" "$STAGE/$lang/configs/"
done

rm -f "$ARCHIVE"
( cd "$STAGE" && zip -r -q "$ARCHIVE" . -x '*__pycache__*' )
unzip -t "$ARCHIVE" > /dev/null || { echo "ERROR: archive failed its CRC check"; exit 1; }

echo "Code archive ready: $ARCHIVE"
du -h "$ARCHIVE" | awk '{print "  size: "$1}'
echo "  python files: $(unzip -l "$ARCHIVE" | grep -c '\.py$')"
echo
echo "Next: Kaggle -> Datasets -> New Dataset -> upload this .zip"
echo "      Title it 'lma-code' so it mounts at /kaggle/input/lma-code"
echo "      Re-upload only this (190 KB) when the code changes; the token dataset stays put."
