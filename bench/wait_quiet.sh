#!/bin/zsh
# Wait until this shared Mac is quiet before a benchmark: 1-minute load average below a threshold
# AND GPU "Device Utilization %" (ioreg, no root needed) below a threshold.
# Usage: wait_quiet.sh [load_threshold=4] [max_wait_s=900] [gpu_util_threshold=15]
thr=${1:-4}
max=${2:-900}
gthr=${3:-15}
waited=0
gpu_util() {
  ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o -E '"Device Utilization %"=[0-9]+' | head -1 | cut -d= -f2
}
while true; do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  g=$(gpu_util); g=${g:-0}
  if (( load < thr && g < gthr )); then
    echo "quiet: load1=$load gpu=${g}% after ${waited}s" >&2
    exit 0
  fi
  if (( waited >= max )); then
    echo "NOT quiet: load1=$load gpu=${g}% after ${waited}s, continuing anyway" >&2
    exit 0
  fi
  sleep 15
  waited=$((waited + 15))
done
