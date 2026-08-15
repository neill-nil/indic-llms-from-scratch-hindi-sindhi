#!/bin/bash
# Train all tokenizer models for Sindhi (BPE + Unigram, 32k vocab)
# This script is designed to run completely detached from any terminal.
set -e
cd /home/neill-rawani/Documents/Assignments/LMA/Indv_Project/individual-project-neill-nil
PYTHON=.venv/bin/python

echo "$(date) — Starting Sindhi BPE tokenizer training (32k vocab)..."
$PYTHON -m common.tokenizer.train_tokenizer --language sindhi --model_type bpe --vocab_size 32000 2>&1
echo "$(date) — Sindhi BPE training complete."

echo ""
echo "$(date) — Starting Sindhi Unigram tokenizer training (32k vocab)..."
$PYTHON -m common.tokenizer.train_tokenizer --language sindhi --model_type unigram --vocab_size 32000 2>&1
echo "$(date) — Sindhi Unigram training complete."

echo ""
echo "$(date) — Running Sindhi tokenizer evaluation..."
$PYTHON -m common.tokenizer.evaluate_tokenizer --language sindhi --vocab_size 32000 2>&1
echo "$(date) — Sindhi tokenizer pipeline DONE."
