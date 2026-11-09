#!/bin/zsh
# Training-step sweep: every candidate on MPS (fp32 / fp16 / bf16 autocast) and CPU (fp32).
# One process per config (clean memory numbers). Waits for a quiet machine before each run.
# Results append to bench/results/train.jsonl
cd "$(dirname "$0")"
PY=$(cd .. && pwd)/.venv/bin/python
OUT=results/train.jsonl
run() {
  ./wait_quiet.sh 4 900
  local load=$(sysctl -n vm.loadavg | awk '{print $2}')
  $PY bench_train.py "$@" 2>/dev/null | grep '^{' | tail -1 | sed "s/}\$/, \"loadavg_1m_start\": $load}/" | tee -a $OUT
}
# fast models: 5 warmup + 20 timed steps
for m in tasnet32 tasnet64 stftlstm bsrnnlstm; do
  for p in fp32 fp16 bf16; do
    run --model $m --device mps --precision $p --warmup 5 --steps 20
  done
done
# GRU models are slow on MPS (no native GRU kernel): 2 warmup + 5 timed steps
for m in stftgru bsrnn; do
  for p in fp32 fp16 bf16; do
    run --model $m --device mps --precision $p --warmup 2 --steps 5
  done
done
# CPU comparison (torch default threads = 4 performance cores)
for m in tasnet32 tasnet64 stftgru stftlstm bsrnn bsrnnlstm; do
  run --model $m --device cpu --precision fp32 --warmup 1 --steps 3
done
