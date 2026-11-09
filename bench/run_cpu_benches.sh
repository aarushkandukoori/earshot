#!/bin/zsh
# CPU benchmarks, each started only when the shared Mac is quiet (see wait_quiet.sh).
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
PY=$ROOT/.venv/bin/python
# 1. PyTorch streaming, 1 thread, default QoS (P-cores)
./wait_quiet.sh 3.5 1800
$PY bench_stream.py --tag p-core 2>/dev/null | grep '^{' > results/stream_cpu_1thread.jsonl
# 2. same, clamped to background QoS = efficiency cores only
./wait_quiet.sh 3.5 1800
taskpolicy -c background $PY bench_stream.py --modes eager --tasnet-impls tm --seconds 5 --tag e-core 2>/dev/null \
  | grep '^{' > results/stream_cpu_1thread_ecore.jsonl
# 3. Accelerate skeleton again (stability check of the first run)
./wait_quiet.sh 3.5 1800
VECLIB_MAXIMUM_THREADS=1 ./chunk_skeleton > results/skeleton_1thread_run2.jsonl
# 4. decode throughput (python -I: reads downloaded audio)
./wait_quiet.sh 3.5 1800
$PY -I bench_decode.py --root $ROOT/data/LibriSpeech/dev-clean --out $ROOT/data/bench_formats > results/decode.jsonl
# 5. RIR simulation + convolution cost
./wait_quiet.sh 3.5 1800
$PY -I bench_augment.py > results/augment.jsonl
# 6. Core ML export check + predict timing
./wait_quiet.sh 3.5 1800
$PY check_coreml.py --models tasnet64 tasnet32 stftlstm stftgru 2>/dev/null | grep '^{' > results/coreml.jsonl
echo done
