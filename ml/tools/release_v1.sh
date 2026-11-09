#!/bin/bash
# Earshot v1 release (10-ml 7.3, 12, 13 and 15 step 3): turn the finished main run into ml/exports/v1/ and the
# evaluation report, in the documented order, checking every step before the next one starts.
#
#   ml/tools/release_v1.sh             run it (from anywhere; it changes to the repo root)
#   ml/tools/release_v1.sh --dry-run   print the checks and the commands without running anything
#   ml/tools/release_v1.sh --force     allowed to overwrite an existing ml/exports/v1/
#   Environment - CKPT (default ml/runs/main/best_1.pt; e.g. a fine-tune checkpoint), CALIB_DEVICE (cpu, or mps
#   once no training runs), CALIB_THREADS (default 8), CALIB_EXTRA (extra calibrate options for a rehearsal, never
#   for the release itself), EVAL_WORKERS (render processes for the evaluation, default 4).
#
# Steps:
#   1. export      python -m ml.export --ckpt $CKPT --out ml/exports/v1/
#                  (EMA weights of the best eligible checkpoint, test vectors from the rendered scene)
#   2. calibrate   python -m ml.tools.calibrate_thresholds --esw ml/exports/v1/earshot.esw
#                  --out ml/runs/main/thresholds.json   (500 captures per set, 1,000 impostor pairs, seed 2030)
#   3. thresholds  python -m ml.export --thresholds ml/runs/main/thresholds.json --out ml/exports/v1/
#                  (copied only if its model_hash equals the earshot.esw data_sha256)
#   4. testvec     python -m ml.export --testvec --scene render --out ml/exports/v1/
#                  (the parity files re-made from the exported .esw; then tell WP-ENGINE to re-run parity)
#   5. evaluate    python -m ml.evaluate --esw ml/exports/v1/earshot.esw --suites all --out docs/eval/report-v1.md
#                  --workers 4 --eval-dir ml/runs/release_v1/eval   (its own result CSVs, so no row of a
#                  development run is ever reused; a rerun after a crash resumes from them)
#   6. report      docs/eval/report-v1.md exists and names the model_hash of ml/exports/v1/earshot.esw
#   7. gate        the report's header says "Release gate (contract 9 model rows) - **MET**"; otherwise the
#                  script stops with exit code 3 and says not to hand the export to WP-ENGINE and WP-IOS
#
# Refuses to start while ml.train or run_forever.sh is running (the GPU and the run folder belong to training),
# when an input is missing, when ml/exports/v1/ already exists (unless --force), or when the free disk space would
# drop under 10 GB. It never deletes anything. Each step's output is also saved in ml/runs/release_v1/<step>.log.
#
# Time (2026-11-08 measurements on this Mac while training ran, so a quiet machine is faster): export about 40 s;
# calibration 104 to 137 s for 20 captures per set and 20 pairs with 2 CPU threads, which scales to about 50 to 65
# min for the full run (fewer minutes with CALIB_THREADS=8 on a quiet machine, an estimate); thresholds copy 2 s;
# test vectors about 35 s; evaluation not run in full yet (ML-2a's estimate from its measured per-mixture rates:
# about 1 h 03 min on MPS, 2 h 40 min on CPU, of which about 35 min is the 60 min stability stream on the CPU).
# Whole release: about 1.5 to 4 h (estimate).
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
CKPT="${CKPT:-ml/runs/main/best_1.pt}"   # the checkpoint to release (override for a fine-tune)
OUT="ml/exports/v1"
THRESHOLDS="ml/runs/main/thresholds.json"
REPORT="docs/eval/report-v1.md"
LOGS="ml/runs/release_v1"
EVAL_DIR="$LOGS/eval"        # the release's own result CSVs (never the development ml/runs/eval)
EVAL_WORKERS="${EVAL_WORKERS:-4}"
NEED_MB=200                  # new files: v1 export about 21 MB, calibration CSVs under 2 MB, eval CSVs and the
                             # DiPCo listening WAVs about 30 MB (estimate), plus margin
FLOOR_MB=10240               # never let the free space drop under 10 GB
CALIB_DEVICE="${CALIB_DEVICE:-cpu}"
CALIB_THREADS="${CALIB_THREADS:-8}"
CALIB_EXTRA="${CALIB_EXTRA:-}"   # extra calibrate options for a rehearsal only, e.g. "--n 20 --pairs 20"

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (use --dry-run, --force or --help)" >&2; exit 2 ;;
    esac
done

fail() {
    echo "release_v1: $*" >&2
    exit 1
}

# ---- checks before anything runs ----------------------------------------------------------------------------
if pgrep -f 'ml\.train( |$)|run_forever\.sh' > /dev/null; then
    fail "refused: ml.train (or run_forever.sh) is running; wait for the main run to finish (pgrep -fl ml.train)"
fi
[ -x "$PY" ] || fail "missing $PY (the project venv)"
[ -f "$CKPT" ] || fail "missing $CKPT (the best eligible checkpoint of the main run)"
[ -f "ml/cache/val_v1.npz" ] || fail "missing ml/cache/val_v1.npz"
[ -f "ml/evaluate.py" ] || fail "missing ml/evaluate.py (WP-ML package ML-2a); a release needs every step"
[ -f "ml/exports/wespeaker_selftest_inputs.npz" ] || fail "missing ml/exports/wespeaker_selftest_inputs.npz"
if [ -e "$OUT" ] && [ "$FORCE" -ne 1 ]; then
    fail "$OUT already exists; pass --force to overwrite its six files (nothing is deleted)"
fi

FREE_MB=$(( $(df -k / | awk 'NR==2 {print $4}') / 1024 ))
echo "release_v1: disk - this release writes about ${NEED_MB} MB; free now ${FREE_MB} MB; must stay above ${FLOOR_MB} MB"
if [ $(( FREE_MB - NEED_MB )) -lt "$FLOOR_MB" ]; then
    fail "refused: not enough disk space (free ${FREE_MB} MB - ${NEED_MB} MB needed < ${FLOOR_MB} MB floor)"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8

# ---- one step: print, run with its log, time it, stop on failure ----------------------------------------------
STEP_NO=0
step() {
    local name="$1"
    shift
    STEP_NO=$(( STEP_NO + 1 ))
    echo
    echo "=== step ${STEP_NO} ${name}: $*"
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi
    local t0=$SECONDS
    if ! caffeinate -dimsu "$@" 2>&1 | tee "$LOGS/${STEP_NO}_${name}.log"; then
        fail "step ${STEP_NO} ${name} failed after $(( SECONDS - t0 )) s; see $LOGS/${STEP_NO}_${name}.log"
    fi
    echo "=== step ${STEP_NO} ${name} done in $(( SECONDS - t0 )) s"
}

# A check that runs after a step (skipped in a dry run).
check() {
    local what="$1"
    shift
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    check: ${what}"
        return 0
    fi
    "$@" || fail "check failed: ${what}"
    echo "    ok: ${what}"
}

esw_hash() {
    "$PY" -c 'import sys; from ml.earshot.fileio import model_hash; print(model_hash(sys.argv[1]))' "$OUT/earshot.esw"
}

json_hash() {
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model_hash"])' "$1"
}

six_files() {
    for f in earshot.esw testvec_short.est testvec_long.est thresholds.json wespk_selftest.est quality_vectors.json; do
        [ -s "$OUT/$f" ] || return 1
    done
}

verify() {
    "$PY" -c 'import sys; from ml.export import verify_folder; print("\n".join(verify_folder(sys.argv[1])))' "$OUT"
}

same_hash() {
    [ "$(json_hash "$1")" = "$(esw_hash)" ]
}

report_names_hash() {
    [ -s "$REPORT" ] && grep -q "$(esw_hash)" "$REPORT"
}

# The exact header line evalreport.release_gate writes when every gated contract 9 row is PASS on a full run
# ("**NOT MET**" never contains this text).
report_gate_met() {
    grep -qF -- "- Release gate (contract 9 model rows) - **MET**" "$REPORT"
}

[ "$DRY_RUN" -eq 1 ] || mkdir -p "$LOGS"
START=$SECONDS

step export "$PY" -m ml.export --ckpt "$CKPT" --out "$OUT/"
check "$OUT holds the six release files and reads back" six_files
check "$OUT passes verify_folder" verify

step calibrate "$PY" -m ml.tools.calibrate_thresholds --esw "$OUT/earshot.esw" --out "$THRESHOLDS" \
    --device "$CALIB_DEVICE" --threads "$CALIB_THREADS" $CALIB_EXTRA
check "$THRESHOLDS exists" test -s "$THRESHOLDS"
check "$THRESHOLDS model_hash equals the earshot.esw data_sha256" same_hash "$THRESHOLDS"

step thresholds "$PY" -m ml.export --thresholds "$THRESHOLDS" --out "$OUT/"
check "$OUT/thresholds.json is the calibrated file" cmp -s "$THRESHOLDS" "$OUT/thresholds.json"

step testvec "$PY" -m ml.export --testvec --scene render --out "$OUT/"
check "$OUT passes verify_folder after the test vectors" verify

step evaluate "$PY" -m ml.evaluate --esw "$OUT/earshot.esw" --suites all --out "$REPORT" \
    --workers "$EVAL_WORKERS" --eval-dir "$EVAL_DIR"
check "$REPORT exists and names the model_hash" report_names_hash

# Gate: the files exist either way, but a release that misses contract 9 must not be handed off.
if [ "$DRY_RUN" -eq 1 ]; then
    echo "    check: $REPORT says the release gate is MET (else exit 3, do not hand off)"
elif ! report_gate_met; then
    echo >&2
    grep -F -- "- Release gate" "$REPORT" >&2 || true
    echo "release_v1: contract 9 NOT met - do not hand ml/exports/v1/ to WP-ENGINE or WP-IOS; read the Overview" \
         "of $REPORT (the export, thresholds and test vectors stay in $OUT for inspection)" >&2
    exit 3
else
    echo "    ok: $REPORT says the release gate is MET"
fi

echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo "release_v1: dry run only, nothing was run"
else
    echo "release_v1: done in $(( SECONDS - START )) s; model_hash $(esw_hash)"
    echo "release_v1: next - tell WP-ENGINE and WP-IOS that ml/exports/v1/ changed (re-run parity), and read $REPORT"
    echo "release_v1: still open - the owner listens to the 10 DiPCo clips in $EVAL_DIR/dipco_listen/"
fi
