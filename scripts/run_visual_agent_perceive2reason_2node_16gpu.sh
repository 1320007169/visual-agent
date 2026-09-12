#!/usr/bin/env bash
set -Eeuo pipefail

# Perceive2Reason training for visual agent RL.
# Extends the standard GroundingDINO RL training with fixed perception prefixes.
# For samples where detection succeeded but reasoning failed, we:
# 1. Fix the tool-calling sequence (perception prefix)
# 2. Only train the final answer generation (reasoning suffix)
# 3. Prohibit tool calls during reasoning via logit bias

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"

# Keep the ModelArts bootstrap identical to the existing two-node entrypoint.
mkdir -p /opt/huawei/explorer-env /home/ma-user/work/algorithm /home/ma-user/work/model

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}

ensure_symlink /opt/huawei/dataset /opt/huawei/explorer-env/dataset
ensure_symlink /opt/huawei/dataset /home/ma-user/work/dataset
ensure_symlink /opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage

export BASE REPO_ROOT RL_ROOT
export RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"
export NNODES="${NNODES:-2}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"

# Perceive2Reason specific naming
export RUN_ID="${RUN_ID:-perceive2reason_qwen3_base_2node_groundingdino_kl_${JOB_TOKEN}}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_perceive2reason/qwen3/$RUN_ID}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-p2r/$RUN_ID}"
export RL_LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-p2r}"

# Use the same prompts and tool config as standard GroundingDINO training
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_groundingdino_config.yaml}"
export VISUAL_TOOL_BACKEND="${VISUAL_TOOL_BACKEND:-groundingdino}"
export SAM3_REPLICAS=0
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-3}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export GROUNDING_DINO_MAX_TEXT_TOKENS="${GROUNDING_DINO_MAX_TEXT_TOKENS:-256}"
export VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT="${VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT:-60}"

# Keep the same hyperparameters as successful groundingdino_kl_safe training
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
export ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
export ACTOR_ENTROPY_COEFF="${ACTOR_ENTROPY_COEFF:-0}"

# Perceive2Reason specific configuration
export PERCEIVE2REASON_MODE="${PERCEIVE2REASON_MODE:-true}"
export PERCEIVE2REASON_DATA_PATH="${PERCEIVE2REASON_DATA_PATH:-$REPO_ROOT/data/perceive2reason/p2r_train.jsonl}"
export PERCEIVE2REASON_MIX_RATIO="${PERCEIVE2REASON_MIX_RATIO:-0.3}"  # 30% p2r samples, 70% standard
export REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-256}"

# Standard training data path (for mixing with P2R samples)
export STANDARD_TRAIN_DATA_PATH="${STANDARD_TRAIN_DATA_PATH:-/opt/huawei/dataset/Common_wl/zwz_rl_vqa/original_images/train.parquet}"
export STANDARD_VAL_DATA_PATH="${STANDARD_VAL_DATA_PATH:-/opt/huawei/dataset/Common_wl/zwz_rl_vqa/original_images/val.parquet}"
export TRAIN_FILES="${TRAIN_FILES:-$STANDARD_TRAIN_DATA_PATH}"
export VAL_FILES="${VAL_FILES:-$STANDARD_VAL_DATA_PATH}"

# Custom dataset configuration (tell trainer to use Perceive2ReasonDataset)
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$RL_ROOT/verl/utils/dataset/perceive2reason_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-Perceive2ReasonDataset}"

# Use Perceive2ReasonRollout instead of the standard vLLM rollout.
export ROLLOUT_NAME="${ROLLOUT_NAME:-perceive2reason}"
export ROLLOUT_MODE="${ROLLOUT_MODE:-sync}"
export ROLLOUT_WORKER_TYPE="${ROLLOUT_WORKER_TYPE:-perceive2reason}"

# Match the existing ModelArts entrypoint's credential contract
export DEEPSEEK_API_KEY_FILE="${DEEPSEEK_API_KEY_FILE:-$BASE/secrets/deepseek_api_key.txt}"
if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${LLM_AS_A_JUDGE_KEY:-}" ]]; then
  [[ -r "$DEEPSEEK_API_KEY_FILE" ]] || {
    echo "error: DeepSeek API key file is not readable: $DEEPSEEK_API_KEY_FILE" >&2
    exit 2
  }
  DEEPSEEK_API_KEY="$(head -n 1 "$DEEPSEEK_API_KEY_FILE")"
  DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY%$'\r'}"
  [[ -n "$DEEPSEEK_API_KEY" ]] || {
    echo "error: DeepSeek API key file is empty: $DEEPSEEK_API_KEY_FILE" >&2
    exit 2
  }
  export DEEPSEEK_API_KEY
fi

# Training control
export RESUME_MODE="${RESUME_MODE:-disable}"
export SAVE_FREQ="${SAVE_FREQ:-5}"  # Save more frequently for P2R experiments
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"

# Print configuration
echo "==================== Perceive2Reason Training Configuration ===================="
echo "RUN_ID: $RUN_ID"
echo "MODEL_PATH: $MODEL_PATH"
echo "RL_OUTPUT_DIR: $RL_OUTPUT_DIR"
echo "PERCEIVE2REASON_MODE: $PERCEIVE2REASON_MODE"
echo "PERCEIVE2REASON_DATA_PATH: $PERCEIVE2REASON_DATA_PATH"
echo "PERCEIVE2REASON_MIX_RATIO: $PERCEIVE2REASON_MIX_RATIO"
echo "ROLLOUT_WORKER_TYPE: $ROLLOUT_WORKER_TYPE"
echo "================================================================================"

# Validate P2R data exists
if [[ "$PERCEIVE2REASON_MODE" == "true" ]]; then
  if [[ ! -f "$PERCEIVE2REASON_DATA_PATH" ]]; then
    echo "error: Perceive2Reason data file not found: $PERCEIVE2REASON_DATA_PATH" >&2
    echo "Please run prepare_perceive2reason_data_from_train.py first" >&2
    exit 2
  fi
  echo "Found P2R data: $(wc -l < "$PERCEIVE2REASON_DATA_PATH") samples"
fi

exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_modelarts.sh"
