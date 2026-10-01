#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NNODES="${NNODES:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-336}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-112}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-336}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-448}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-36000}"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_n16_8node_${JOB_TOKEN}}"

# Keep the tool protocol and training defaults shared with the 24-GPU run.
exec bash "$SCRIPT_DIR/run_visual_agent_multitool_vlocr_3node_24gpu.sh" "$@"
