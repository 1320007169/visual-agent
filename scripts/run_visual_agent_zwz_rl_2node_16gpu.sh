#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts two-node / sixteen-GPU ZWZ original-image relation GRPO.
# Run the same entrypoint on both 8-GPU nodes. Per node, GPUs 0-6 run
# VERL/vLLM and GPU 7 runs the shared visual-tool replica pool.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
ZWZ_RL_DIR="${ZWZ_RL_DIR:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation}"

export BASE REPO_ROOT RL_ROOT ZWZ_RL_DIR
export RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
export CUDA_LIBRARY_DIR="${CUDA_LIBRARY_DIR:-$CUDA_HOME/targets/x86_64-linux/lib}"
export CC="${CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

export NNODES="${NNODES:-2}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-604800}"
export SAM3_REPLICAS="${SAM3_REPLICAS:-4}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"

# NODE_RANK and MASTER_ADDR are resolved by the common launcher from the
# ModelArts worker host list. They can still be supplied for manual launches.
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_agent_combined_8679/qwen3/full/sft_visual_sft_20260719_154704}"
export TRAIN_FILES="${TRAIN_FILES:-$ZWZ_RL_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ZWZ_RL_DIR/val.parquet}"
export DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-messages}"
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$RL_ROOT/verl/utils/dataset/zwz_original_relation_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-ZwzOriginalRelationDataset}"
export PREPARE_SMOKE_DATA=0

# 112 prompts x 16 samples = 1,792 trajectories per update, processed in
# waves of 28. The PPO mini batch is divisible across 14 RL GPUs.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-14}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-28}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-28}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-165}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export MAX_TURNS="${MAX_TURNS:-9}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
export FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-2}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:--1}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-visual-agent-zwz-original-relation-rl}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-zwz_original_relation_qwen3_2node_toolpool_crop_v2_${JOB_TOKEN}}"
export OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-zwz-rl/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system.txt}"
export VISUAL_AGENT_IMAGE_MAX_PIXELS="${VISUAL_AGENT_IMAGE_MAX_PIXELS:-2359296}"
export VISUAL_AGENT_IMAGE_PATCH_SIZE="${VISUAL_AGENT_IMAGE_PATCH_SIZE:-16}"

# Reward is 0.9 answer accuracy + 0.1 answer format. The API judge is only a
# fallback when rule matching rejects an answer. Inject the key at job launch;
# credentials must never be stored in this repository.
export ENABLE_API_JUDGE="${ENABLE_API_JUDGE:-1}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://api.deepseek.com}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4-flash}"
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-${DEEPSEEK_API_KEY:-}}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.modelarts.com:80}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.modelarts.com:80}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

case "$ENABLE_API_JUDGE" in
  auto)
    [[ -n "$LLM_AS_A_JUDGE_BASE" ]] && JUDGE_ENABLED=1 || JUDGE_ENABLED=0
    ;;
  1|true|TRUE|yes|YES) JUDGE_ENABLED=1 ;;
  0|false|FALSE|no|NO) JUDGE_ENABLED=0 ;;
  *)
    echo "error: ENABLE_API_JUDGE must be auto, 1, or 0; got $ENABLE_API_JUDGE" >&2
    exit 2
    ;;
esac
export JUDGE_ENABLED

if [[ "$JUDGE_ENABLED" == "1" ]]; then
  [[ -n "$LLM_AS_A_JUDGE_BASE" ]] || {
    echo "error: LLM_AS_A_JUDGE_BASE is required when the API judge is enabled" >&2
    exit 2
  }
  [[ -n "$LLM_AS_A_JUDGE_KEY" ]] || {
    echo "error: set DEEPSEEK_API_KEY or LLM_AS_A_JUDGE_KEY before launching" >&2
    exit 2
  }
  export LLM_AS_A_JUDGE_TIMEOUT="${LLM_AS_A_JUDGE_TIMEOUT:-20}"
  export LLM_AS_A_JUDGE_RETRIES="${LLM_AS_A_JUDGE_RETRIES:-2}"
  export LLM_AS_A_JUDGE_PREFLIGHT="${LLM_AS_A_JUDGE_PREFLIGHT:-1}"
else
  unset LLM_AS_A_JUDGE_BASE LLM_AS_A_JUDGE_KEY LLM_AS_A_JUDGE_MODEL \
    LLM_AS_A_JUDGE_TIMEOUT LLM_AS_A_JUDGE_RETRIES LLM_AS_A_JUDGE_PREFLIGHT
fi

for required_path in \
  "$RL_ENV_DIR/bin/python" \
  "$TRAIN_FILES" \
  "$VAL_FILES" \
  "$CUSTOM_DATASET_PATH" \
  "$MODEL_PATH/config.json" \
  "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"; do
  [[ -e "$required_path" ]] || {
    echo "error: required path does not exist: $required_path" >&2
    exit 2
  }
done

"$RL_ENV_DIR/bin/python" - "$TRAIN_FILES" "$VAL_FILES" <<'PYIMAGECHECK'
import sys
from pathlib import Path

import pyarrow.parquet as pq

missing = []
for label, parquet_path in zip(("train", "validation"), sys.argv[1:]):
    table = pq.read_table(parquet_path, columns=["images"])
    print(f"ZWZ {label} rows: {table.num_rows}")
    for values in table.column("images").to_pylist():
        for value in values or []:
            if not Path(value).is_file():
                missing.append(value)
if missing:
    raise SystemExit(f"missing {len(missing)} original-image paths, first: {missing[0]}")
print("ZWZ selected original-image dataset ready")
PYIMAGECHECK

echo "ZWZ train / validation: $TRAIN_FILES / $VAL_FILES"
echo "Initial checkpoint: $MODEL_PATH"
echo "Train batch / rollout n: $TRAIN_BATCH_SIZE / $ROLLOUT_N"
echo "Training steps / save frequency: $TOTAL_TRAINING_STEPS / $SAVE_FREQ"
echo "Tool replicas per node (SAM3 / GroundingDINO): $SAM3_REPLICAS / $GROUNDING_DINO_REPLICAS"
echo "Rollout traces: $ROLLOUT_DATA_DIR"
echo "API judge fallback: $JUDGE_ENABLED"

exec bash "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"
