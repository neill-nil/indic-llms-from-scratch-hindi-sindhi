#!/bin/bash
# Train all tokenizer models for Hindi (BPE + Unigram, 32k vocab)
# This script is designed to run completely detached from any terminal.
set -e
cd /home/neill-rawani/Documents/Assignments/LMA/Indv_Project/individual-project-neill-nil
PYTHON=.venv/bin/python

echo "$(date) — Starting Hindi BPE tokenizer training (32k vocab)..."
$PYTHON -m common.tokenizer.train_tokenizer --language hindi --model_type bpe --vocab_size 16000 2>&1
echo "$(date) — Hindi BPE training complete."

echo ""
echo "$(date) — Starting Hindi Unigram tokenizer training (32k vocab)..."
$PYTHON -m common.tokenizer.train_tokenizer --language hindi --model_type unigram --vocab_size 16000 2>&1
echo "$(date) — Hindi Unigram training complete."

echo ""
echo "$(date) — Running Hindi tokenizer evaluation..."
$PYTHON -m common.tokenizer.evaluate_tokenizer --language hindi --vocab_size 16000 2>&1
echo "$(date) — Hindi tokenizer pipeline DONE."
