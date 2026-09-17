#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must point to the completed training run}"
TRAIN_RUN_ID="${RUN_ID:?RUN_ID must identify the completed training run}"
LATEST_FILE="$TRAIN_OUTPUT_DIR/latest_checkpointed_iteration.txt"
[[ -r "$LATEST_FILE" ]] || { echo "error: missing checkpoint pointer: $LATEST_FILE" >&2; exit 2; }
LATEST_STEP="$(tr -d '\r\n' < "$LATEST_FILE")"
[[ "$LATEST_STEP" =~ ^[0-9]+$ ]] || { echo "error: invalid checkpoint step" >&2; exit 2; }
export RL_DINO_LATEST_STEP="$LATEST_STEP"
export RL_DINO_LATEST_MODEL_PATH="$TRAIN_OUTPUT_DIR/global_step_$LATEST_STEP/actor/huggingface"
[[ -f "$RL_DINO_LATEST_MODEL_PATH/config.json" ]] || { echo "error: missing HF checkpoint" >&2; exit 2; }
export EVAL_MODELS=dino_latest
export EVAL_DATASETS=HRBench4K
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0
export GROUNDING_DINO_REPLICAS="${EVAL_GROUNDING_DINO_REPLICAS:-2}"
export TOOL_CUDA_VISIBLE_DEVICES="${EVAL_TOOL_CUDA_VISIBLE_DEVICES:-0}"
export MODEL_CUDA_VISIBLE_DEVICES="${EVAL_MODEL_CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES="grounding_detect,crop_zoom"
export MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-65536}"
export VISUAL_AGENT_MAX_TURNS="${EVAL_MAX_TURNS:-6}"
export VISUAL_AGENT_MAX_TOKENS="${EVAL_MAX_TOKENS:-6144}"
export VLMEVAL_API_NPROC="${EVAL_API_NPROC:-28}"
export EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-0}"
export RUN_ID="${TRAIN_RUN_ID}_final_step${LATEST_STEP}_hrbench4k"
export WORK_ROOT="${EVAL_WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/$RUN_ID}"
export LOG_DIR="${EVAL_LOG_DIR:-$BASE/logs/visual-agent-eval}"
exec bash "$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh"
