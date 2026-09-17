#!/usr/bin/env bash
set -Eeuo pipefail

OUTPUT_FILE="${1:?usage: monitor_gpu_load.sh OUTPUT_FILE PROCESS_OUTPUT_FILE INTERVAL_SECONDS NODE_RANK}"
PROCESS_OUTPUT_FILE="${2:?usage: monitor_gpu_load.sh OUTPUT_FILE PROCESS_OUTPUT_FILE INTERVAL_SECONDS NODE_RANK}"
INTERVAL_SECONDS="${3:?usage: monitor_gpu_load.sh OUTPUT_FILE PROCESS_OUTPUT_FILE INTERVAL_SECONDS NODE_RANK}"
NODE_RANK="${4:?usage: monitor_gpu_load.sh OUTPUT_FILE PROCESS_OUTPUT_FILE INTERVAL_SECONDS NODE_RANK}"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "error: nvidia-smi is required for GPU monitoring" >&2
  exit 2
}
[[ "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: GPU monitor interval must be a positive integer, got '$INTERVAL_SECONDS'" >&2
  exit 2
}

mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$PROCESS_OUTPUT_FILE")"
if [[ ! -s "$OUTPUT_FILE" ]]; then
  echo "sample_time,node_rank,index,uuid,name,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,power_draw_w,power_limit_w,temperature_c" > "$OUTPUT_FILE"
fi
if [[ ! -s "$PROCESS_OUTPUT_FILE" ]]; then
  echo "sample_time,node_rank,gpu_uuid,pid,process_name,used_gpu_memory_mib" > "$PROCESS_OUTPUT_FILE"
fi

stop_requested=0
trap 'stop_requested=1' INT TERM

while (( ! stop_requested )); do
  sample_time="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  while IFS= read -r row; do
    [[ -n "$row" ]] && printf '%s,%s,%s\n' "$sample_time" "$NODE_RANK" "$row"
  done < <(
    nvidia-smi \
      --query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true
  ) >> "$OUTPUT_FILE"

  while IFS= read -r row; do
    [[ -n "$row" ]] && printf '%s,%s,%s\n' "$sample_time" "$NODE_RANK" "$row"
  done < <(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  ) >> "$PROCESS_OUTPUT_FILE"

  sleep "$INTERVAL_SECONDS" &
  wait "$!" || true
done
