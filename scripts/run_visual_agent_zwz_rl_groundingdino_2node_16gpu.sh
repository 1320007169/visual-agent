#!/usr/bin/env bash
set -Eeuo pipefail

# GroundingDINO-only detector ablation for ZWZ original-relation GRPO.
# The model can call grounding_detect and deterministic crop_zoom; SAM3 is not
# loaded and is absent from both the prompt and the rollout tool schema.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"

# Keep the ModelArts bootstrap identical to the existing two-node entrypoint.
# The wrapper is executed on every node, while the repository launcher handles
# Ray, workers, checkpointing, and the visual-tool process pool.
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
# Unconditionally pin the toolchain to the portable conda CUDA env. The
# ModelArts base image presets CUDA_HOME=/usr/local/cuda (which does not exist
# on these nodes); keeping it via `:-` produces a nonexistent conda gcc path
# and crashes every vLLM EngineCore with FileNotFoundError.
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
export RUN_ID="${RUN_ID:-zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_${JOB_TOKEN}}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-zwz-rl/$RUN_ID}"
export RL_LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}"

export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_groundingdino_config.yaml}"
export VISUAL_TOOL_BACKEND="${VISUAL_TOOL_BACKEND:-groundingdino}"
export SAM3_REPLICAS=0
# Two ingress servers per node with three replicas each preserve the previous
# six-detector concurrency while using only GroundingDINO on the tool GPU.
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-3}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export GROUNDING_DINO_MAX_TEXT_TOKENS="${GROUNDING_DINO_MAX_TEXT_TOKENS:-256}"
export VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT="${VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT:-60}"

# Stabilize GRPO against policy drift. Keep the original actor learning rate so
# the actor KL loss below is the isolated optimization change for this run.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
export ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
export ACTOR_ENTROPY_COEFF="${ACTOR_ENTROPY_COEFF:-0}"

# Match the existing ModelArts entrypoint's credential contract without
# putting the secret in this script or in the submitted repository.
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

# Start a clean run by default so this remains a controlled detector ablation.
export RESUME_MODE="${RESUME_MODE:-disable}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_modelarts.sh"
