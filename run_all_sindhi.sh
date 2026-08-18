#!/bin/bash
set -e

echo "=== 1. Extracting cloned datasets ==="
.venv/bin/python sindhi/data/collection/extract_arnizamani.py

echo "=== 2. Running scrapers for remaining data ==="
.venv/bin/python run_scrapers.py 300

echo "=== 3. Preprocessing Sindhi data ==="
.venv/bin/python -m common.preprocessing.pipeline --language sindhi

echo "=== 4. Fuzzy Deduplication (MinHash LSH) ==="
.venv/bin/python common/preprocessing/minhash_dedup.py --language sindhi
mv sindhi/data/preprocessed/train_minhash.txt sindhi/data/preprocessed/train.txt
.venv/bin/python common/preprocessing/minhash_eval_splits.py --language sindhi --split val
mv sindhi/data/preprocessed/val_minhash.txt sindhi/data/preprocessed/val.txt
.venv/bin/python common/preprocessing/minhash_eval_splits.py --language sindhi --split test
mv sindhi/data/preprocessed/test_minhash.txt sindhi/data/preprocessed/test.txt

echo "=== 5. Counting Final Tokens ==="
.venv/bin/python count_tokens.py sindhi

echo "=== DONE ==="
