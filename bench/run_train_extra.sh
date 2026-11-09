#!/bin/zsh
# Follow-up training runs: re-runs, fp16 with static loss scaling, activation checkpointing,
# sustained (thermal) runs, and the same configs on the newest torch (temporary venv) for comparison.
# Results append to bench/results/train_extra.jsonl
cd "$(dirname "$0")"
PY=$(cd .. && pwd)/.venv/bin/python
PY_NEW=$(cd .. && pwd)/.venv-torch215/bin/python
OUT=results/train_extra.jsonl
run() {
  local py=$1; shift
  ./wait_quiet.sh 4 900
  local load=$(sysctl -n vm.loadavg | awk '{print $2}')
  $py bench_train.py "$@" 2>/dev/null | grep '^{' | tail -1 | sed "s/}\$/, \"loadavg_1m_start\": $load}/" | tee -a $OUT
}
# re-run under a quiet machine + fp16 with static loss scale
run $PY --model tasnet32 --device mps --precision fp32 --warmup 5 --steps 20
run $PY --model tasnet32 --device mps --precision fp16 --warmup 5 --steps 20
run $PY --model tasnet64 --device mps --precision fp16 --warmup 5 --steps 20
# activation checkpointing per TCN block (memory vs time)
run $PY --model tasnet32 --device mps --precision fp32 --warmup 3 --steps 10 --ckpt
run $PY --model tasnet32 --device mps --precision bf16 --warmup 3 --steps 10 --ckpt
# newest torch, same code
if [ -x $PY_NEW ]; then
  for m in tasnet32 tasnet64 stftgru stftlstm bsrnn bsrnnlstm; do
    run $PY_NEW --model $m --device mps --precision fp32 --warmup 3 --steps 10
  done
  run $PY_NEW --model tasnet64 --device mps --precision bf16 --warmup 3 --steps 10
  run $PY_NEW --model tasnet64 --device mps --precision fp16 --warmup 3 --steps 10 --grad-scaler
  run $PY_NEW --model stftgru --device cpu --precision fp32 --warmup 1 --steps 3
fi
# sustained runs (thermal check): a few minutes each
run $PY --model stftlstm --device mps --precision fp32 --warmup 5 --steps 500
run $PY --model tasnet64 --device mps --precision bf16 --warmup 5 --steps 300
run $PY --model stftgru --device cpu --precision fp32 --warmup 2 --steps 200
