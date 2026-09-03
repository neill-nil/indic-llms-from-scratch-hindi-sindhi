#!/bin/bash
# Stop a local training run *gracefully*, so it writes a checkpoint before exiting.
#
#   bash scripts/stop_training.sh sindhi
#
# Sends SIGTERM, which train.py catches and turns into a checkpoint save. Never use
# `kill -9` — that gives the process no chance to save, and you fall back to the last
# periodic checkpoint instead.
set -e
cd "$(dirname "$0")/.."

LANG="${1:?usage: stop_training.sh <hindi|sindhi>}"
PIDFILE="$LANG/train/run/train.pid"
[ -f "$PIDFILE" ] || { echo "No $PIDFILE — is a run in progress?"; exit 1; }
PID=$(cat "$PIDFILE")

kill -0 "$PID" 2>/dev/null || { echo "PID $PID is not running."; exit 1; }
echo "Sending SIGTERM to $PID and waiting for the checkpoint to be written..."
kill -TERM "$PID"

for _ in $(seq 1 60); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done

if kill -0 "$PID" 2>/dev/null; then
  echo "Still running after 60s. Check $LANG/train/run/train.log before forcing."
  exit 1
fi
echo "Stopped cleanly. Resume with:  bash scripts/train_local.sh $LANG"
tail -3 "$LANG/train/run/train.log"
