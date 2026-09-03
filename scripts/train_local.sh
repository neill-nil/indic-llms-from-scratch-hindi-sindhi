#!/bin/bash
# Pretrain one model locally, safely, on a laptop.
#
#   bash scripts/train_local.sh sindhi
#   bash scripts/train_local.sh hindi
#
# Re-running is always safe: if a run already exists it RESUMES from the last checkpoint
# rather than restarting or overwriting.
#
# Three protections, because a multi-hour run on a laptop will be interrupted:
#
#   1. systemd-inhibit  stops the machine suspending while training. A CUDA context does
#                       not reliably survive suspend/resume — the process usually wakes to
#                       an invalid GPU state and dies — so preventing the suspend is far
#                       better than trying to recover from it. Covers closing the lid.
#   2. nohup + SIGHUP   closing the terminal does not kill the run.
#   3. signal handlers  SIGTERM/SIGINT/SIGHUP save a checkpoint before exiting, so a
#                       deliberate stop loses nothing.
#
# What is NOT covered: SIGKILL, a power cut, or a forced suspend. The periodic checkpoint
# is the backstop there, which is why the interval below is short.
set -e
cd "$(dirname "$0")/.."

LANG="${1:?usage: train_local.sh <hindi|sindhi>}"
case "$LANG" in hindi|sindhi) ;; *) echo "language must be hindi or sindhi"; exit 1 ;; esac

OUT_DIR="$LANG/train/run"
PYTHON=.venv/bin/python
# ~100 steps is about 3 minutes at the measured local throughput, so an unexpected kill
# costs at most that much work. Each checkpoint is ~280 MB and writes in a second or two
# on NVMe, well under 1% overhead, so there is no reason to be stingy here.
CKPT_EVERY=100

# Both models must take the SAME optimizer step — 24,576 tokens — or the H-vs-L comparison
# is confounded by one having seen larger batches. micro_batch is whatever fits in 4 GB of
# VRAM at that language's context length (measured: ctx 512 peaks at 2.2 GB with 8
# sequences, ctx 1024 at 2.9 GB with 4), and grad_accum is then DERIVED to hit the target.
# Setting either by hand is how a mismatched batch or an OOM sneaks in.
TOKENS_PER_STEP=24576
CTX=$($PYTHON -c "import json;print(json.load(open('$LANG/configs/model_config.json'))['max_seq_len'])")
case "$CTX" in
  512)  MICRO=8 ;;
  1024) MICRO=4 ;;
  *)    echo "No measured micro_batch for ctx=$CTX; add one after checking VRAM."; exit 1 ;;
esac
if [ $(( TOKENS_PER_STEP % (MICRO * CTX) )) -ne 0 ]; then
  echo "TOKENS_PER_STEP $TOKENS_PER_STEP is not divisible by micro $MICRO x ctx $CTX"; exit 1
fi
ACCUM=$(( TOKENS_PER_STEP / (MICRO * CTX) ))
echo "Batch: micro $MICRO x accum $ACCUM x ctx $CTX = $(( MICRO * ACCUM * CTX )) tokens/step"

# ── Refuse to run on battery ────────────────────────────────────────────────────
# A sustained GPU load draws more than a laptop battery can supply on its own. On this
# machine that caused an instant firmware power-cut mid-run: no shutdown sequence, no
# kernel log, the last minutes of journal lost. The battery gauge here reports
# energy-full = 0 Wh, so the OS cannot even warn about a low charge. Requiring AC is the
# only reliable guard.
ON_AC=0
for ac in /sys/class/power_supply/A*/online; do
  [ -f "$ac" ] && [ "$(cat "$ac")" = "1" ] && ON_AC=1
done
if [ "$ON_AC" != "1" ]; then
  echo "REFUSING TO START: the laptop is not on AC power."
  echo "  Sustained GPU training draws more than the battery can supply, which has"
  echo "  already caused one instant power-cut on this machine. Plug in and re-run."
  echo "  To override anyway:  ALLOW_BATTERY=1 bash scripts/train_local.sh $LANG"
  [ "${ALLOW_BATTERY:-0}" = "1" ] || exit 1
  echo "  ALLOW_BATTERY=1 set — proceeding against advice."
fi

RESUME=""
if [ -f "$OUT_DIR/latest.pt" ]; then
  STEP=$($PYTHON -c "import torch;print(torch.load('$OUT_DIR/latest.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo "?")
  echo "Found $OUT_DIR/latest.pt at step $STEP — resuming."
  RESUME="--resume"
else
  echo "Starting a fresh $LANG run in $OUT_DIR."
fi

mkdir -p "$OUT_DIR"

INHIBIT=""
if command -v systemd-inhibit >/dev/null; then
  INHIBIT="systemd-inhibit --what=handle-lid-switch:sleep:idle --who=lma-training --why=pretraining --mode=block"
  echo "Sleep and lid-close suspend are inhibited for the duration of the run."
else
  echo "WARNING: systemd-inhibit not found. Closing the lid may suspend the machine and"
  echo "         kill the run. Progress is still checkpointed every $CKPT_EVERY steps."
fi

# expandable_segments reduces allocator fragmentation, which matters on a 4 GB card
# where the attention matrices are large relative to total memory.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $INHIBIT $PYTHON -m common.train.train \
  --language "$LANG" \
  --out_dir "$OUT_DIR" \
  --micro_batch "$MICRO" \
  --grad_accum "$ACCUM" \
  --checkpoint_every "$CKPT_EVERY" \
  $RESUME \
  > "$OUT_DIR/nohup.out" 2>&1 &

PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo
echo "  PID            : $PID  (saved to $OUT_DIR/train.pid)"
echo "  watch progress : tail -f $OUT_DIR/train.log"
echo "  stop safely    : bash scripts/stop_training.sh $LANG"
echo "  resume later   : bash scripts/train_local.sh $LANG"
