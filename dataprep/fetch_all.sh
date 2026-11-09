#!/usr/bin/env bash
# Fetch the whole Earshot data pack v1 (speech, noise, RIRs, eval), then verify it.
#
#   bash dataprep/fetch_all.sh                                  # from the repo root
#   nohup bash dataprep/fetch_all.sh > /dev/null 2>&1 &    # overnight
#
# Every step is resumable, so re-running the script after a crash, a reboot or a disk-floor stop continues where it
# left off (finished files are skipped). The steps run one after another because they share about 11 MB/s of
# download. A failing step does not stop the later ones; the script's exit code is the number of failed steps.
# Logs - data/logs/fetch_all.log (start/end of each step) and data/logs/fetch_all.<step>.out (full output), plus each
# fetcher's own log in data/logs/.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # the repo root, wherever it was cloned
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/data/logs"
MAIN_LOG="$LOGS/fetch_all.log"

mkdir -p "$LOGS"
cd "$ROOT" || exit 1

failed=0

run_step() {
  local name="$1"
  shift
  echo "$(date '+%Y-%m-%d %H:%M:%S') START $name - $PY $*" | tee -a "$MAIN_LOG"
  "$PY" "$@" >> "$LOGS/fetch_all.$name.out" 2>&1
  local rc=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') END   $name - exit code $rc" | tee -a "$MAIN_LOG"
  if [ "$rc" -ne 0 ]; then
    failed=$((failed + 1))
  fi
}

echo "$(date '+%Y-%m-%d %H:%M:%S') fetch_all start, $(df -h "$ROOT" | awk 'NR==2 {print $4}') free" | tee -a "$MAIN_LOG"

run_step speech -m dataprep.speech_fetch --source all
run_step noise  -m dataprep.noise_fetch --source all
run_step rir    -m dataprep.rir_build --sim --real
run_step eval   -m dataprep.eval_fetch
run_step verify -m dataprep.verify

echo "$(date '+%Y-%m-%d %H:%M:%S') fetch_all end, $failed step(s) failed, summary in data/manifests/SUMMARY.md" \
  | tee -a "$MAIN_LOG"
exit "$failed"
