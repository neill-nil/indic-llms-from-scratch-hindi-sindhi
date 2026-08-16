#!/bin/bash
set -e

echo "=== 1. Downloading/Streaming Hindi data ==="
.venv/bin/python -m hindi.data.collection.download_public_data

echo "=== 2. Preprocessing Hindi data ==="
.venv/bin/python -m common.preprocessing.pipeline --language hindi

echo "=== DONE ==="
