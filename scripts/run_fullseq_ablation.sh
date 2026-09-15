#!/bin/bash
# Ablation: finetune with the loss on the WHOLE sequence instead of the answer only.
#
#   systemd-run --user --unit=lma-fullseq --collect --working-directory="$PWD" \
#       bash scripts/run_fullseq_ablation.sh
#
# Tests the explanation offered for two failures of the main finetuned models: roughly
# doubled bits per byte on ordinary text, and the inability to produce entity names never
# seen in finetuning. Both were attributed to answer-only supervision on a tied embedding.
# If that is right, supervising every token should reduce both.
#
# Exactly ONE variable changes. Learning rate, epochs, batch, seed and data are those of the
# main runs; the rate is read from the committed probe result rather than re-probed, so the
# comparison is not also a comparison of learning rates. Validation is still scored on the
# answer only, so the two runs' curves are directly comparable.
#
# Idempotent: completed finetunes resume to their end and are skipped in effect.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=fullseq_ablation.log
stage() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
trap 'stage "ABLATION FAILED at line $LINENO (exit $?)"' ERR

ON_AC=0
for ac in /sys/class/power_supply/A*/online; do
  [ -f "$ac" ] && [ "$(cat "$ac")" = "1" ] && ON_AC=1
done
if [ "$ON_AC" != "1" ] && [ "${ALLOW_BATTERY:-0}" != "1" ]; then
  stage "REFUSING TO START: not on AC power"; exit 1
fi

LR=$($PY -c "import json;print(json.load(open('report/finetune_lr_probe.json'))['chosen_learning_rate'])")
stage "ablation start — loss on full sequence, lr=$LR (from the probe)"

for lang in hindi sindhi; do
  OUT="$lang/finetune/run_fullseq"
  RESUME=""; [ -f "$OUT/latest.pt" ] && RESUME="--resume"
  stage "finetune $lang ${RESUME:-(fresh)}"
  $PY -m common.finetune.run --language "$lang" --loss_on full --learning_rate "$LR" \
      --out_dir "$OUT" $RESUME >> "$LOG" 2>&1
done

stage "greedy evaluation"
$PY -m common.finetune.evaluate --checkpoint finetuned_fullseq --shots 0 >> "$LOG" 2>&1
stage "forced choice"
$PY -m common.finetune.rank_candidates --checkpoint finetuned_fullseq >> "$LOG" 2>&1
stage "language-modelling check (validation split)"
$PY -m common.eval.intrinsic --all --splits val --checkpoint finetuned_fullseq \
    --batch_size 16 --out report/intrinsic_results_finetuned_fullseq.json >> "$LOG" 2>&1
stage "ABLATION COMPLETE"
