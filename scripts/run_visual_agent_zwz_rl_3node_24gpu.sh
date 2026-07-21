#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts three-node / twenty-four-GPU ZWZ original-image relation GRPO.
# Run this same entrypoint on all three 8-GPU nodes. Per node, GPUs 0-6 run
# VERL/vLLM and GPU 7 runs four SAM3 plus two GroundingDINO replicas.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NNODES="${NNODES:-3}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export SAM3_REPLICAS="${SAM3_REPLICAS:-4}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"

# 126 prompts x 16 samples = 2,016 trajectories per update. Across 21 RL
# GPUs this is 96 trajectories per GPU. A scaled PPO mini batch of 42 x 16
# keeps 32 trajectories per GPU, matching the established 16-GPU run.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-126}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-21}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-42}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-42}"

# floor(18,518 / 126) = 146 updates, or 294,336 sampled trajectories. This is
# within 0.5% of the 16-GPU run's 295,680 trajectories.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-146}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-zwz_original_relation_qwen3_3node_toolpool_crop_v3_${JOB_TOKEN}}"

exec bash "$SCRIPT_DIR/run_visual_agent_zwz_rl_2node_16gpu.sh"
