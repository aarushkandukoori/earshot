#!/bin/bash
# Crash-restart wrapper for long training runs (10-ml 10.4 and 10.7, contract 8.4).
#
#   ml/tools/run_forever.sh ml/configs/main.yaml
#
# Runs `python -m ml.train --config CONFIG --resume last` under `caffeinate -dimsu` (the Mac does not sleep), again
# and again, and decides by the exit code of ml.train:
#   0      the run finished (last step or stop_at) or stopped on a STOP file  -> stop here
#   2      a gate failed                                                      -> stop (a human looks)
#   3      NaN storm (10 non-finite steps within 1,000)                       -> stop (a human looks)
#   4      refused to start (a configuration problem a restart cannot fix)    -> stop
#   other  a crash (1, or killed by a signal, e.g. 137)  -> wait 30 s and restart from the newest checkpoint
# It gives up after 5 crashes in a row that each lasted under 10 minutes (then something is broken, not unlucky),
# and after 5 crashes in a row that all resumed from the same checkpoint, however long each lasted (exit 5). The data
# is exactly reproducible, so a crash caused by one batch repeats at the same step on every restart; when the replay
# from the checkpoint takes over 10 minutes, the quick-crash rule alone would restart it forever (checkpoints are
# about 19 minutes apart). Quick crashes hit both limits at the 5th crash and keep exit 1, as before.
# Every example is seeded by (seed, example index), so a restarted run sees exactly the data it would have seen.
#
# Stopping cleanly: `touch ml/runs/<run>/STOP`. ml.train notices it within 50 steps, saves a checkpoint and exits 0;
# this script also checks for it before every start. Delete the file to continue later with the same command.
# This script never deletes files. Its own log is ml/runs/<run>/run_forever.log.
set -u

if [ $# -ne 1 ]; then
  echo "usage: ml/tools/run_forever.sh CONFIG.yaml" >&2
  exit 64
fi
CONFIG="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"     # the repository root (this file is ROOT/ml/tools/run_forever.sh)
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8

# The run folder ml/runs/<run>/ comes from the config's `run` key (with base: merging, as ml.train reads it).
RUN="$("$PY" -c 'import sys; from ml.earshot.config import load_yaml; print(load_yaml(sys.argv[1])["run"])' "$CONFIG")"
if [ -z "$RUN" ]; then
  echo "cannot read the run name from $CONFIG" >&2
  exit 1
fi
RUN_DIR="$ROOT/ml/runs/$RUN"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run_forever.log"

say() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

trap 'say "interrupted; stopping"; exit 130' INT TERM

# Step number of the newest step_<n>.pt in the run folder (what --resume last loads), or "none".
newest_ckpt() {
  local n
  n=$(ls "$RUN_DIR" 2>/dev/null | sed -n 's/^step_\([0-9][0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
  echo "${n:-none}"
}

WAIT_S="${RUN_FOREVER_WAIT_S:-30}"              # seconds between a crash and the restart (tests set it lower)
QUICK_S="${RUN_FOREVER_QUICK_S:-600}"           # a crash before this many seconds is "quick" (tests change it)
quick_crashes=0                                 # crashes in a row that each lasted under 10 minutes
same_resume=0                                   # crashes in a row that all resumed from the same checkpoint
prev_resume=""
while true; do
  if [ -e "$RUN_DIR/STOP" ]; then
    say "STOP file found ($RUN_DIR/STOP); not starting. Delete it to continue."
    exit 0
  fi
  resume_from=$(newest_ckpt)
  say "starting: python -m ml.train --config $CONFIG --resume last (newest checkpoint: step $resume_from)"
  start=$(date +%s)
  caffeinate -dimsu "$PY" -m ml.train --config "$CONFIG" --resume last
  code=$?
  lasted=$(( $(date +%s) - start ))
  say "ml.train exited with code $code after $lasted s"
  case "$code" in
    0) say "run finished or stopped cleanly"; exit 0 ;;
    2) say "gate failed; not restarting (see $RUN_DIR/events.log)"; exit 2 ;;
    3) say "NaN storm; not restarting (see $RUN_DIR/events.log)"; exit 3 ;;
    4) say "ml.train refused to start; not restarting"; exit 4 ;;
  esac
  if [ "$lasted" -lt "$QUICK_S" ]; then
    quick_crashes=$((quick_crashes + 1))
  else
    quick_crashes=0                             # a crash after 10+ minutes of training does not count
  fi
  if [ "$quick_crashes" -ge 5 ]; then
    say "5 crashes in a row, each under 10 minutes; giving up (see $RUN_DIR/events.log)"
    exit 1
  fi
  if [ "$resume_from" = "$prev_resume" ]; then
    same_resume=$((same_resume + 1))
  else
    same_resume=1                               # this checkpoint crashed once so far
  fi
  prev_resume="$resume_from"
  if [ "$same_resume" -ge 5 ]; then
    say "5 crashes in a row, all resumed from checkpoint step $resume_from (no new checkpoint in between);" \
        "probably the same batch crashes every time. Giving up (see $RUN_DIR/events.log)"
    exit 5
  fi
  say "crash; restarting in $WAIT_S s (quick crashes in a row: $quick_crashes," \
      "from checkpoint step $resume_from: $same_resume)"
  sleep "$WAIT_S"
done
