#!/bin/bash
# Bonus ablation: run the full Phase 2 evaluation suite on the no-positional-embeddings model.
#
#   systemd-run --user --unit=lma-nopos-eval --collect --working-directory="$PWD" \
#       bash scripts/run_nopos_eval.sh
#
# The brief asks for the complete Phase 2 suite on the ablated model — perplexity and bits
# per byte, generation metrics with diversity diagnostics, and attention heatmaps with
# entropy and distance — compared against the standard model for that language.
#
# Every command uses the same settings Phase 2 used, so the two are directly comparable:
# the same splits, the same 200 prompts of 64 tokens continued for 128, the same 64 windows
# of 256 tokens at seed 0. Only --checkpoint and --out differ, so the Phase 2 result files
# are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=nopos_eval.log
stage() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
trap 'stage "NOPOS EVAL FAILED at line $LINENO (exit $?)"' ERR

if [ ! -f hindi/train/run_nopos/best.pt ]; then
  stage "no checkpoint at hindi/train/run_nopos/best.pt — run scripts/train_nopos.sh first"
  exit 1
fi

# Each stage is skipped if its output already exists, so a re-run after a failure resumes
# rather than repeating work that succeeded. Delete the file to force a stage to re-run.
run_stage() {                       # run_stage <label> <output file> <command...>
  local label="$1" out="$2"; shift 2
  if [ -s "$out" ]; then
    stage "$label — $out already exists, skipping"
  else
    stage "$label"
    "$@" >> "$LOG" 2>&1
  fi
}

run_stage "1/3 intrinsic: perplexity and bits per byte (val + test)" \
  report/intrinsic_results_nopos.json \
  $PY -m common.eval.intrinsic --language hindi --splits val test --checkpoint nopos \
      --out report/intrinsic_results_nopos.json

run_stage "2/3 generation: BLEU, chrF++, ROUGE-L, diversity and repetition" \
  report/generation_results_nopos.json \
  $PY -m common.eval.generation --language hindi --split val --n_prompts 200 \
      --checkpoint nopos --out report/generation_results_nopos.json

run_stage "3/3 attention: entropy, distance, heatmaps" \
  report/attention_results_nopos.json \
  $PY -m common.eval.attention --language hindi --split val --checkpoint nopos \
      --out report/attention_results_nopos.json

stage "tables and figures"
$PY -m common.eval.write_nopos_report >> "$LOG" 2>&1
$PY -m common.eval.plot_nopos >> "$LOG" 2>&1
stage "NOPOS EVAL COMPLETE"
