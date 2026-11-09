#!/bin/bash
# wait for a quiet machine (1-min load < 6), then time each config twice, interleaved
for i in $(seq 1 360); do
  L=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$L" 'BEGIN{exit !(l<6)}'; then break; fi
  sleep 10
done
echo "start load $(sysctl -n vm.loadavg)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTORCH_ENABLE_MPS_FALLBACK=0
PY="$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python"
for c in replica_product_esb16 merged_full merged_no_wes merged_full_4x8s merged_full replica_product_esb16; do
  $PY -I bench_merged.py $c 2>/dev/null
done
echo "end load $(sysctl -n vm.loadavg)"
