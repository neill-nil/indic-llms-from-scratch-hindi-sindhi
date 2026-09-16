#!/bin/bash
# Bonus ablation: pretrain Model H (Hindi) with positional embeddings removed.
#
#   bash scripts/train_nopos.sh                 # start, or resume after any interruption
#   systemctl --user status lma-nopos           # is it running?
#   tail -f hindi/train/run_nopos/train.log     # progress
#   systemctl --user stop lma-nopos             # stop safely — saves a checkpoint first
#
# Identical to Model H's pretraining in every setting except the missing position table:
# same corpus, tokenizer, architecture, 500M-token budget, 24,576 tokens per optimizer step
# (8 x 6 x 512), learning-rate schedule and seed. The one difference that could not be held
# fixed is hardware: Model H trained on a Kaggle T4 in fp16, this runs on the local GPU in
# bf16. That is recorded as a limitation of the comparison.
#
# Runs as its own systemd user service rather than under nohup or setsid, so closing the
# editor or terminal cannot kill it — an earlier evaluation died exactly that way. It does
# NOT survive logging out, a reboot, or a power cut; checkpoints every 100 steps (about
# 2-3 minutes) bound what any of those can cost, and re-running this script resumes.
set -e
cd "$(dirname "$0")/.."

OUT=hindi/train/run_nopos
CONFIG=hindi/configs/model_config_nopos.json
PY=.venv/bin/python
UNIT=lma-nopos

ON_AC=0
for ac in /sys/class/power_supply/A*/online; do
  [ -f "$ac" ] && [ "$(cat "$ac")" = "1" ] && ON_AC=1
done
if [ "$ON_AC" != "1" ]; then
  echo "REFUSING TO START: not on AC power. This laptop has power-cut under sustained GPU"
  echo "load on battery. Plug in and re-run:  bash scripts/train_nopos.sh"
  exit 1
fi

if systemctl --user is-active --quiet "$UNIT"; then
  echo "Already running. Progress:  tail -f $OUT/train.log"
  exit 0
fi

RESUME=""
if [ -f "$OUT/latest.pt" ]; then
  STEP=$($PY -c "import torch;print(torch.load('$OUT/latest.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo "?")
  echo "Found $OUT/latest.pt at step $STEP — resuming."
  RESUME="--resume"
else
  echo "Starting a fresh no-positional-embeddings run in $OUT."
fi
mkdir -p "$OUT"

systemd-run --user --unit="$UNIT" --collect --working-directory="$PWD" \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  systemd-inhibit --what=handle-lid-switch:sleep:idle --who="$UNIT" \
    --why="no-positional-embeddings pretraining" --mode=block \
  "$PY" -m common.train.train --language hindi --config "$CONFIG" --out_dir "$OUT" \
    --micro_batch 8 --grad_accum 6 --checkpoint_every 100 $RESUME

echo
echo "  running as     : systemd user service '$UNIT' (lid may be closed; stay plugged in)"
echo "  progress       : tail -f $OUT/train.log"
echo "  stop safely    : systemctl --user stop $UNIT"
echo "  resume later   : bash scripts/train_nopos.sh"
