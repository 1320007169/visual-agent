#!/usr/bin/env bash
set -Eeuo pipefail

# Four-node / 32-GPU SLIME entrypoint for the DeepEyesV2 mixed n=8 run.
# Run this same script on all four 8-GPU ModelArts nodes. Per node, GPUs 0-6
# are colocated SLIME actor/rollout workers and GPU 7 hosts GroundingDINO.

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"

export NNODES=4
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-$TOOL_GPU}"

# This keeps tool capacity proportional to the 28 rollout engines. For a
# utilization-first trial, override these with 1 and inspect W&B/CSV GPU load.
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-3}"

export RUN_ID="${RUN_ID:-zwz_deepeyesv2_3k_nocount_v1_n8_4node_slime}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_UPLOAD_MODE="${WANDB_UPLOAD_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-visual-agent-deepeyesv2-rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-$BASE/secrets/wandb_api_key.txt}"
export WANDB_SYSTEM_MONITOR_INTERVAL="${WANDB_SYSTEM_MONITOR_INTERVAL:-5}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_deepeyesv2_n8_slime_2node_16gpu.sh"
