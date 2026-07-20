#!/usr/bin/env bash
set -Eeuo pipefail

# Visual-Agent GRPO on Kwai-Keye/Thyme-RL. Run this same command on both nodes.
# Per node: GPUs 0-6 = VERL, GPU 7 = local SAM3 + GroundingDINO.
# The judge is an already-running OpenAI-compatible service.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
THYME_DATA_DIR="${THYME_DATA_DIR:-$REPO_ROOT/data/thyme_rl_snapshot/data}"
QWEN3_RL_ENV="${QWEN3_RL_ENV:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
QWEN3_CUDA_HOME="${QWEN3_CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
QWEN3_CUDA_LIBRARY_DIR="${QWEN3_CUDA_LIBRARY_DIR:-$QWEN3_CUDA_HOME/targets/x86_64-linux/lib}"
QWEN3_CUDA_HOST_CC="${QWEN3_CUDA_HOST_CC:-$QWEN3_CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
QWEN3_CUDA_HOST_CXX="${QWEN3_CUDA_HOST_CXX:-$QWEN3_CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system.txt}"

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_agent_combined_4679/qwen3/full/sft_visual_sft_20260719_001713}"
export RL_ENV_DIR="${RL_ENV_DIR:-$QWEN3_RL_ENV}"
export CUDA_HOME="$QWEN3_CUDA_HOME"
export CUDA_LIBRARY_DIR="$QWEN3_CUDA_LIBRARY_DIR"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$RL_SYSTEM_PROMPT_FILE"
export PREPARE_SMOKE_DATA=0
export CC="$QWEN3_CUDA_HOST_CC"
export CXX="$QWEN3_CUDA_HOST_CXX"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

export DATA_PROMPT_KEY=messages
export CUSTOM_DATASET_PATH="$REPO_ROOT/reinforcement_learning/verl/utils/dataset/thyme_visual_agent_dataset.py"
export CUSTOM_DATASET_NAME=ThymeVisualAgentDataset
export TOOL_CALL_FORMAT="${TOOL_CALL_FORMAT:-hermes}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-14}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export MAX_TURNS="${MAX_TURNS:-4}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-8}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# Saving every GRPO update would create thousands of large FSDP checkpoints in
# a full Thyme epoch. Keep periodic recovery points without dominating I/O.
export SAVE_FREQ="${SAVE_FREQ:-200}"
export TEST_FREQ="${TEST_FREQ:--1}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-visual-agent-thyme-rl}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-visual_agent_thyme_qwen3_${JOB_TOKEN}}"
export OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_thyme_rl/qwen3/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-thyme-rl}"

[[ -d "$THYME_DATA_DIR" ]] || {
  echo "error: Thyme data directory not found: $THYME_DATA_DIR" >&2
  exit 2
}

train_files=()
for ((index=0; index<97; index++)); do
  file="$THYME_DATA_DIR/train-$(printf '%05d' "$index")-of-00098.parquet"
  [[ -f "$file" ]] || {
    echo "error: missing Thyme shard: $file" >&2
    exit 2
  }
  train_files+=("$file")
done
val_file="$THYME_DATA_DIR/train-00097-of-00098.parquet"
[[ -f "$val_file" ]] || {
  echo "error: missing Thyme validation shard: $val_file" >&2
  exit 2
}

IFS=,
export TRAIN_FILES="${train_files[*]}"
unset IFS
export VAL_FILES="$val_file"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -n "${LLM_AS_A_JUDGE_BASE:-}" ]] || {
    echo "error: LLM_AS_A_JUDGE_BASE is required, for example http://judge-host:18901/v1" >&2
    exit 2
  }
  "$RL_ENV_DIR/bin/python" - "${LLM_AS_A_JUDGE_BASE}" <<'PYJUDGE'
import json
import sys
import urllib.request
base = sys.argv[1].rstrip("/")
if not base.endswith("/v1"):
    base += "/v1"
with urllib.request.urlopen(base + "/models", timeout=15) as response:
    payload = json.load(response)
models = payload.get("data") or []
if not models:
    raise SystemExit("judge server returned no models")
print("Judge service ready:", models[0].get("id"))
PYJUDGE
fi

echo "Thyme train shards: 97"
echo "Thyme validation shard: $val_file"
echo "Initial Visual-Agent checkpoint: $MODEL_PATH"
echo "Qwen3 RL environment: $RL_ENV_DIR"
echo "Qwen3 CUDA toolkit: $CUDA_HOME"
echo "RL system prompt: $RL_SYSTEM_PROMPT_FILE"
echo "Training steps: $TOTAL_TRAINING_STEPS"
echo "Judge: ${LLM_AS_A_JUDGE_BASE:-skipped-for-dry-run}"

exec bash "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"
