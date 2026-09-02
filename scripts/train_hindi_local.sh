#!/bin/bash
# Pretrain Model H (Hindi) locally on the RTX 3050.
#
# Safe to re-run: if a run already exists it resumes from the last checkpoint rather than
# starting over or overwriting. Detached with nohup so closing the terminal does not kill
# a multi-hour run.
set -e
cd "$(dirname "$0")/.."

OUT_DIR="hindi/train/run"
PYTHON=.venv/bin/python

RESUME=""
if [ -f "$OUT_DIR/latest.pt" ]; then
  echo "Found $OUT_DIR/latest.pt — resuming."
  RESUME="--resume"
else
  echo "Starting a fresh run in $OUT_DIR."
fi

mkdir -p "$OUT_DIR"
nohup $PYTHON -m common.train.train \
  --language hindi \
  --out_dir "$OUT_DIR" \
  --micro_batch 8 \
  --grad_accum 6 \
  --checkpoint_every 500 \
  $RESUME \
  > "$OUT_DIR/nohup.out" 2>&1 &

echo "PID $! — training detached."
echo
echo "  watch progress : tail -f $OUT_DIR/train.log"
echo "  stop           : kill $!"
echo "  resume later   : bash scripts/train_hindi_local.sh"
