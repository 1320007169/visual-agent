#!/usr/bin/env bash
set -Eeuo pipefail

# Train the original Qwen3-VL-8B-Instruct with raw depth and TallyQA counting QA.
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924}"
export BASE REPO_ROOT PIPELINE_ROOT
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA_DIR/val.parquet}"
export DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-messages}"
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$REPO_ROOT/reinforcement_learning/verl/utils/dataset/zwz_deepeyesv2_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-ZwzDeepEyesV2Dataset}"
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_config.yaml}"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_multitool_depth_count.txt}"
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-1}"
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"

export NNODES="${NNODES:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-276}"
export MAX_TURNS="${MAX_TURNS:-8}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export SAVE_FREQ="${SAVE_FREQ:-40}"
export RUN_ID="${RUN_ID:-qwen3base_depth_tallyqa5k_multitool_n16_$(date +%Y%m%d_%H%M%S)}"

# Both isolated VTS services must be reachable on every training node.
export VTS_DEPTH_ENDPOINT="${VTS_DEPTH_ENDPOINT:-http://127.0.0.1:9003}"
export VTS_COUNT_ENDPOINT="${VTS_COUNT_ENDPOINT:-http://127.0.0.1:9004}"
export VTS_TOOL_BRIDGE_ROOT="${VTS_TOOL_BRIDGE_ROOT:-$BASE/outputs/visual-agent-vts-bridge}"
export COUNT_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
if [[ ! -f "$COUNT_SERVICE_CONFIG" ]]; then
  echo "error: default CountGD++ service config is missing: $COUNT_SERVICE_CONFIG" >&2
  exit 2
fi
token_env="${VTS_TOOL_SERVICE_TOKEN_ENV:-VTS_TOOL_SERVICE_TOKEN}"
if [[ -z "${!token_env:-}" ]]; then
  echo "error: set $token_env for the VTS depth and count services" >&2
  exit 2
fi
for endpoint in "$VTS_DEPTH_ENDPOINT" "$VTS_COUNT_ENDPOINT"; do
  if ! curl --fail --silent --max-time 5 "$endpoint/health" >/dev/null; then
    echo "error: VTS service is not healthy at $endpoint; start depth and default CountGD++ ($COUNT_SERVICE_CONFIG) on this node before training" >&2
    exit 2
  fi
done

exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_2node_16gpu.sh"
