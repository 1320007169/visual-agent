#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts four-node / thirty-two-GPU ZWZ original-image relation GRPO.
# Run this same entrypoint on all four 8-GPU nodes. Per node, GPUs 0-5 run
# VERL/vLLM and GPUs 6-7 run four independent visual-tool service processes.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
ZWZ_RL_DIR="${ZWZ_RL_DIR:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation}"

export BASE REPO_ROOT RL_ROOT ZWZ_RL_DIR
export RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"

# Match the known-good two-node launcher exactly. ModelArts defines CUDA_HOME
# as /usr/local/cuda on some images, but that tree does not contain the Conda
# host compiler required by Triton during vLLM engine profiling.
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/merged/global_step_40}"
export WARM_START_DATA_PATH="${WARM_START_DATA_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_2node_toolpool_crop_v2_manual/global_step_40/data.pt}"
export WARM_START_GLOBAL_STEP="${WARM_START_GLOBAL_STEP:-40}"

# Replace this placeholder before submitting the ModelArts job.
export DEEPSEEK_API_KEY="REPLACE_WITH_YOUR_DEEPSEEK_API_KEY"

export NNODES="${NNODES:-4}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TOOL_GPU="${TOOL_GPU:-6}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-6,7}"
export WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-604800}"
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-4}"
export SAM3_REPLICAS="${SAM3_REPLICAS:-3}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-1}"

# Step 40 already consumed 4,480 prompts. At batch 144, logical steps 41-137
# consume the remaining training epoch (the final dropped remainder is normal).
# Each update gives 96 trajectories per RL GPU and four PPO mini-batches.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-144}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-24}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-36}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-72}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-137}"
export SAVE_FREQ="${SAVE_FREQ:-16}"
export RESUME_MODE="${RESUME_MODE:-auto}"

# The 14-GPU FSDP optimizer shards cannot load on 24 GPUs. Warm-start the
# portable model plus dataloader cursor at step 40, then use a fresh optimizer.
export RUN_ID="${RUN_ID:-zwz_original_relation_qwen3_4node_6rl2tool_from_2node_step40}"


# The scheduler exposes the model volume under /opt before the bootstrap script
# creates the stable /home/ma-user/work/model symlink used by the training config.
MODELARTS_REPO_ROOT="$REPO_ROOT"
SCHEDULER_REPO_ROOT="/opt/huawei/quoteModel/xiaoyi_tmpstorage/haohang/min/gx/visual-agent"
if [[ -d "$SCHEDULER_REPO_ROOT" ]]; then
  MODELARTS_REPO_ROOT="$SCHEDULER_REPO_ROOT"
fi

exec bash "$MODELARTS_REPO_ROOT/scripts/run_visual_agent_zwz_rl_modelarts.sh"
