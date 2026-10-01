#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NNODES="${NNODES:-2}"
# Preserve the global update batch and dataloader cursor from the 24-GPU run.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-126}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-42}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-126}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export ALLOW_FSDP_WORLD_SIZE_CHANGE=True
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-36000}"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_n16_2node_${JOB_TOKEN}}"

exec bash "$SCRIPT_DIR/run_visual_agent_multitool_vlocr_3node_24gpu.sh" "$@"
