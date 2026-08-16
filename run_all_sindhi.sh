#!/bin/bash
set -e

echo "=== 1. Extracting cloned datasets ==="
.venv/bin/python extract_arnizamani.py

echo "=== 2. Running scrapers for remaining data ==="
.venv/bin/python run_scrapers.py 300

echo "=== 3. Preprocessing Sindhi data ==="
.venv/bin/python -m common.preprocessing.pipeline --language sindhi

echo "=== DONE ==="
