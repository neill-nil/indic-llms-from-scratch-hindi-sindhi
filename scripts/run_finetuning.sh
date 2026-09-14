#!/bin/bash
# The complete finetuning experiment for both models, unattended.
#
#   bash scripts/run_finetuning.sh             # foreground
#   setsid nohup bash scripts/run_finetuning.sh > /dev/null 2>&1 &   # detached
#
# Stages, in order of how much the result matters:
#   1. learning-rate probe     3 rates x 2 languages, 1 epoch each, validation only
#   2. final finetuning        both languages, 3 epochs, at the chosen rate
#   3. finetuned evaluation    greedy exact match on all four test splits
#   4. pretrained 4-shot       baseline, skipped if its results already exist
#
# The 4-shot baseline runs LAST even though it is logically a baseline: it is the slowest
# stage and the least consequential, so an interruption should cost it rather than the
# finetuning results.
#
# Idempotent. Every stage either skips completed work or resumes it, so re-running after
# any interruption continues rather than restarts.
#
# Run it DETACHED (setsid) for anything unattended. An earlier baseline run was started
# as a child of an interactive session and was killed, unfinished, when that session
# ended — having written no results file. setsid puts the pipeline in its own session so
# it survives the terminal or tool that launched it.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=finetune_pipeline.log

stage() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
trap 'stage "PIPELINE FAILED at line $LINENO (exit $?) — see $LOG"' ERR

ON_AC=0
for ac in /sys/class/power_supply/A*/online; do
  [ -f "$ac" ] && [ "$(cat "$ac")" = "1" ] && ON_AC=1
done
if [ "$ON_AC" != "1" ] && [ "${ALLOW_BATTERY:-0}" != "1" ]; then
  stage "REFUSING TO START: not on AC power (override with ALLOW_BATTERY=1)"
  exit 1
fi

# Hold off sleep and lid-close suspend for the whole pipeline. A CUDA context does not
# reliably survive a suspend.
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  exec env INHIBITED=1 systemd-inhibit --what=handle-lid-switch:sleep:idle \
    --who=lma-finetune --why="reasoning finetuning" --mode=block "$0" "$@"
fi

stage "pipeline start (pid $$)"

stage "1/4 learning-rate probe"
$PY -m common.finetune.lr_probe >> "$LOG" 2>&1
LR=$($PY -c "import json;print(json.load(open('report/finetune_lr_probe.json'))['chosen_learning_rate'])")
stage "    chosen learning rate: $LR"

stage "2/4 final finetuning at lr=$LR"
for lang in hindi sindhi; do
  OUT="$lang/finetune/run"
  RESUME=""
  [ -f "$OUT/latest.pt" ] && RESUME="--resume"
  stage "    $lang ${RESUME:-(fresh)}"
  $PY -m common.finetune.run --language "$lang" --learning_rate "$LR" \
      --out_dir "$OUT" $RESUME >> "$LOG" 2>&1
done

stage "3/4 finetuned evaluation (greedy, exact match)"
$PY -m common.finetune.evaluate --checkpoint finetuned --shots 0 >> "$LOG" 2>&1

stage "4/4 pretrained 4-shot baseline"
if [ -f report/reasoning_pretrained_4shot.json ]; then
  stage "    results exist — skipping"
else
  $PY -m common.finetune.evaluate --checkpoint pretrained --shots 4 >> "$LOG" 2>&1
fi

stage "PIPELINE COMPLETE"
