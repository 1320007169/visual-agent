#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="$BASE/visual-agent"

export BASE REPO_ROOT
export SKIP_MODELARTS_BOOTSTRAP=1
export ENV_DIR="${ENV_DIR:-$BASE/conda_envs/visual-agent-eval}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/visual-agent-eval}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-$BASE/conda_envs/visual-tools}"
export MODEL_CUDA_HOME="${MODEL_CUDA_HOME-}"
export MODEL_CC="${MODEL_CC:-$BASE/conda_envs/deepeyes-sft-conda/bin/gcc}"
export MODEL_CXX="${MODEL_CXX:-$BASE/conda_envs/deepeyes-sft-conda/bin/g++}"

export EVAL_MODELS="${EVAL_MODELS:-dino_step120}"
export EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K}"
export VISUAL_TOOL_BACKEND=groundingdino
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=grounding_detect,crop_zoom
export SAM3_REPLICAS=0
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-1}"

export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0}"
export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-1}"
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-1}"
export MODEL_SERVER_BACKEND="${MODEL_SERVER_BACKEND:-transformers}"
export VISUAL_AGENT_MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-12}"
export VISUAL_AGENT_MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-512}"
export RUN_ID="${RUN_ID:-qwen3_dino_only_step120_local_2gpu_$(date +%Y%m%d_%H%M%S)}"
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/qwen3_dino_only_local/$RUN_ID}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh" "$@"
