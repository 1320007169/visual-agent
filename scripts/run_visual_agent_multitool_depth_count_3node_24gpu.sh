#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same launcher on three 8-GPU nodes: 21 RL GPUs and 3 tool GPUs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export NNODES="${NNODES:-3}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"

# 126 x 16 = 2016 trajectories: 96 per RL GPU, with PPO minibatches of 32.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-126}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-42}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-126}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-168}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export TEST_FREQ="${TEST_FREQ:-40}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_depth_tallyqa5k_multitool_pairdepth_n16_3node_${JOB_TOKEN}}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
  for key in NNODES RL_CUDA_VISIBLE_DEVICES TOOL_GPU VISUAL_TOOL_SERVERS_PER_NODE \
    TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE VAL_BATCH_SIZE ROLLOUT_N MAX_CONCURRENT_REQUESTS \
    TOTAL_TRAINING_STEPS TOTAL_EPOCHS RESUME_MODE TEST_FREQ SAVE_FREQ VAL_BEFORE_TRAIN RUN_ID; do
    printf '%s=%s\n' "$key" "${!key}"
  done
  exit 0
fi

exec bash "$SCRIPT_DIR/run_visual_agent_multitool_depth_count_2node_16gpu.sh" "$@"
