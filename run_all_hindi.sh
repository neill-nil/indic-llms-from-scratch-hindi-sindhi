#!/bin/bash
set -e

echo "=== 1. Downloading/Streaming Hindi data ==="
.venv/bin/python -m hindi.data.collection.download_public_data

echo "=== 2. Preprocessing Hindi data ==="
.venv/bin/python -m common.preprocessing.pipeline --language hindi

echo "=== 3. Fuzzy Deduplication (MinHash LSH) ==="
.venv/bin/python common/preprocessing/minhash_dedup.py --language hindi
mv hindi/data/preprocessed/train_minhash.txt hindi/data/preprocessed/train.txt
.venv/bin/python common/preprocessing/minhash_eval_splits.py --language hindi --split val
mv hindi/data/preprocessed/val_minhash.txt hindi/data/preprocessed/val.txt
.venv/bin/python common/preprocessing/minhash_eval_splits.py --language hindi --split test
mv hindi/data/preprocessed/test_minhash.txt hindi/data/preprocessed/test.txt

echo "=== 4. Counting Final Tokens ==="
.venv/bin/python count_tokens.py hindi

echo "=== DONE ==="
