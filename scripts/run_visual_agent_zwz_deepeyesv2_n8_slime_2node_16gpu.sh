#!/usr/bin/env bash
set -Eeuo pipefail

# SLIME counterpart of run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh.
# Run this entrypoint on both 8-GPU ModelArts nodes.

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
export ZWZ_RL_DIR="${ZWZ_RL_DIR:-$REPO_ROOT/data/zwz_deepeyesv2_3k_nocount_hr4k_v1}"
export TRAIN_FILES="${TRAIN_FILES:-$ZWZ_RL_DIR/train.parquet}"
export SOURCE_DATA="${SOURCE_DATA:-$TRAIN_FILES}"

# Match the VERL run: 112 prompts x 8 samples, split into four optimizer
# steps of 28 prompts x 8 samples. Global batch 224 is divisible by 14 GPUs.
export TRAIN_BATCH_SIZE=112
export PPO_MINI_BATCH_SIZE=28
export ROLLOUT_N=8
export NUM_ROLLOUT="${NUM_ROLLOUT:-192}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

export RUN_ID="${RUN_ID:-zwz_deepeyesv2_3k_nocount_v1_n8_2node_slime}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"

export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://models.sjtu.edu.cn/api/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-chat}"
export LLM_AS_A_JUDGE_BACKUP_BASE="${LLM_AS_A_JUDGE_BACKUP_BASE:-https://api-cn.hi-code.cc/v1}"
export LLM_AS_A_JUDGE_BACKUP_MODEL="${LLM_AS_A_JUDGE_BACKUP_MODEL:-deepseek-v4.1-flash}"
export LLM_AS_A_JUDGE_RETRIES="${LLM_AS_A_JUDGE_RETRIES:-2}"
export JUDGE_ENABLED=1

load_key() {
  local variable_name="$1"
  local key_file="$2"
  local value="${!variable_name:-}"
  if [[ -z "$value" ]]; then
    [[ -r "$key_file" ]] || { echo "error: judge key file is not readable: $key_file" >&2; exit 2; }
    IFS= read -r value < "$key_file" || true
    value="${value%$'\r'}"
    [[ -n "$value" ]] || { echo "error: judge key file is empty: $key_file" >&2; exit 2; }
    printf -v "$variable_name" '%s' "$value"
    export "$variable_name"
  fi
}

if [[ "${MIXED_CONFIG_ONLY:-0}" != "1" ]]; then
  load_key LLM_AS_A_JUDGE_KEY "${JUDGE_KEY_FILE:-$BASE/secrets/sjtu_judge_api_key.txt}"
  load_key LLM_AS_A_JUDGE_BACKUP_KEY "${BACKUP_JUDGE_KEY_FILE:-$BASE/secrets/hicode_judge_api_key.txt}"
fi

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.modelarts.com:80}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

if [[ "${MIXED_CONFIG_ONLY:-0}" == "1" ]]; then
  for key in RUN_ID RL_OUTPUT_DIR NNODES RL_CUDA_VISIBLE_DEVICES \
    TOOL_CUDA_VISIBLE_DEVICES VISUAL_TOOL_SERVERS_PER_NODE GROUNDING_DINO_REPLICAS \
    TRAIN_FILES TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N NUM_ROLLOUT \
    SAVE_INTERVAL LLM_AS_A_JUDGE_BASE LLM_AS_A_JUDGE_MODEL WANDB_ENABLE \
    WANDB_UPLOAD_MODE WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP WANDB_API_KEY_FILE; do
    printf '%s=%s\n' "$key" "${!key:-}"
  done
  exit 0
fi

echo "note: this SLIME launcher trains the mixed set but does not yet run periodic HRBench4K validation" >&2
exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_groundingdino_slime_2node_16gpu.sh"
