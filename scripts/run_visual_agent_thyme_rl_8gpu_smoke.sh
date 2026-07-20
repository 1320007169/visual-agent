#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts one-node / eight-GPU environment smoke test for Thyme RL.
# GPUs 0-6 run VERL/vLLM; GPU 7 runs the local SAM3 + GroundingDINO service.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
THYME_DATA_DIR="${THYME_DATA_DIR:-$REPO_ROOT/data/thyme_rl_snapshot/data}"

export RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
export CUDA_LIBRARY_DIR="${CUDA_LIBRARY_DIR:-$CUDA_HOME/targets/x86_64-linux/lib}"
export CC="${CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"

export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_agent_combined_8679/qwen3/full/sft_visual_sft_20260719_154704}"
export TRAIN_FILES="${TRAIN_FILES:-$THYME_DATA_DIR/train-00000-of-00098.parquet}"
export VAL_FILES="${VAL_FILES:-$THYME_DATA_DIR/train-00001-of-00098.parquet}"
export DATA_PROMPT_KEY=messages
export CUSTOM_DATASET_PATH="$REPO_ROOT/reinforcement_learning/verl/utils/dataset/thyme_visual_agent_dataset.py"
export CUSTOM_DATASET_NAME=ThymeVisualAgentDataset
export PREPARE_SMOKE_DATA=0

export TRAIN_BATCH_SIZE=14
export VAL_BATCH_SIZE=2
export ROLLOUT_N=2
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1
export MAX_PROMPT_LENGTH=8192
export MAX_RESPONSE_LENGTH=4096
export MAX_TURNS=4
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
export FILTER_OVERLONG_PROMPTS=True
export TRAIN_SHUFFLE=True
export SAVE_FREQ=1
export TEST_FREQ=-1
export TRAINER_PROJECT_NAME=visual-agent-thyme-rl-env-smoke
export RUN_ID="${RUN_ID:-thyme_qwen3_8gpu_env_smoke_${MA_JOB_ID:-${VC_JOB_ID:-manual}}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_thyme_rl/qwen3/$RUN_ID}"

# For an infrastructure smoke test exact-match reward is sufficient; do not
# require a separately deployed judge service before Ray/vLLM/tool validation.
unset LLM_AS_A_JUDGE_BASE LLM_AS_A_JUDGE_KEY LLM_AS_A_JUDGE_MODEL

exec bash "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"
