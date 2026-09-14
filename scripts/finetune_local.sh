#!/bin/bash
# Finetune one model on its reasoning corpus, locally and safely.
#
#   bash scripts/finetune_local.sh hindi
#   bash scripts/finetune_local.sh sindhi
#
# Re-running is always safe: an existing run RESUMES from its last checkpoint rather than
# restarting or overwriting, exactly as in pretraining.
#
# The same three protections as scripts/train_local.sh — sleep inhibit, nohup, and signal
# handlers that checkpoint before exiting. Finetuning is far shorter than pretraining
# (minutes, not hours), so these matter less here, but a run that dies without a
# checkpoint is still a run that has to start again, and the cost of keeping them is zero.
set -e
cd "$(dirname "$0")/.."

LANG="${1:?usage: finetune_local.sh <hindi|sindhi>}"
case "$LANG" in hindi|sindhi) ;; *) echo "language must be hindi or sindhi"; exit 1 ;; esac

OUT_DIR="$LANG/finetune/run"
PYTHON=.venv/bin/python
PRETRAINED="$LANG/train/run/best.pt"

if [ ! -f "$PRETRAINED" ]; then
  echo "Pretrained checkpoint not found: $PRETRAINED"
  echo "  Finetuning starts from that language's own pretrained weights. Download it from"
  echo "  the Drive link in README.md, or re-run: bash scripts/train_local.sh $LANG"
  exit 1
fi

if [ ! -f "$LANG/finetune/data/train.jsonl" ]; then
  echo "Reasoning data not found. Build it first:  $PYTHON -m common.reasoning.generate"
  exit 1
fi

# Same AC-power guard as pretraining. This machine has had one firmware power-cut
# mid-run under sustained GPU load on battery, and its battery gauge reports
# energy-full = 0 Wh so the OS cannot warn about it.
ON_AC=0
for ac in /sys/class/power_supply/A*/online; do
  [ -f "$ac" ] && [ "$(cat "$ac")" = "1" ] && ON_AC=1
done
if [ "$ON_AC" != "1" ]; then
  echo "REFUSING TO START: the laptop is not on AC power."
  echo "  Override with:  ALLOW_BATTERY=1 bash scripts/finetune_local.sh $LANG"
  [ "${ALLOW_BATTERY:-0}" = "1" ] || exit 1
  echo "  ALLOW_BATTERY=1 set — proceeding against advice."
fi

RESUME=""
if [ -f "$OUT_DIR/latest.pt" ]; then
  STEP=$($PYTHON -c "import torch;print(torch.load('$OUT_DIR/latest.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo "?")
  echo "Found $OUT_DIR/latest.pt at step $STEP — resuming."
  RESUME="--resume"
else
  echo "Starting a fresh $LANG finetune from $PRETRAINED."
fi

mkdir -p "$OUT_DIR"

INHIBIT=""
if command -v systemd-inhibit >/dev/null; then
  INHIBIT="systemd-inhibit --what=handle-lid-switch:sleep:idle --who=lma-finetune --why=finetuning --mode=block"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $INHIBIT $PYTHON -m common.finetune.run \
  --language "$LANG" \
  --out_dir "$OUT_DIR" \
  $RESUME \
  "${@:2}" \
  > "$OUT_DIR/nohup.out" 2>&1 &

PID=$!
echo "$PID" > "$OUT_DIR/finetune.pid"
echo
echo "  PID            : $PID"
echo "  watch progress : tail -f $OUT_DIR/finetune.log"
echo "  resume later   : bash scripts/finetune_local.sh $LANG"
