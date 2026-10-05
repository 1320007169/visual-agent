#!/usr/bin/env bash
set -Eeuo pipefail

# Start counterfactual GRPO from base weights on two 8-GPU ModelArts nodes.
# Prepare the shared dataset once with prepare_reliance_pairs.py before launch.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/vlocr_reliance_pairs_tallyhalf_fsc3000_20261006}"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet"
export VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_reliance_n16_2node}"

export NNODES=2
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-126}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-42}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-126}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-0}"
export TRAIN_SHUFFLE=True
# Changed training rows require a fresh optimizer and dataloader cursor.
export RESUME_MODE=disable
export RESUME_FROM_PATH=""

export RL_SPLIT_OCR=0
export RL_CHART_PARSE=1
export VTS_VL_OCR=1
export TOOL_CONFIG_PATH="$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" != "1" ]]; then
    [[ -s "$MULTITOOL_DATA_DIR/manifest.json" ]] || {
        echo "error: prepare reliance pairs before launching: $MULTITOOL_DATA_DIR" >&2
        exit 2
    }
fi

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh" "$@"
