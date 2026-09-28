#!/usr/bin/env bash
set -Eeuo pipefail

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_SPLIT_OCR=1
export TOOL_CONFIG_PATH="$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_ocr_config.yaml"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_depth_count_ocr.txt"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_text_det_rec_${JOB_TOKEN}}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh" "$@"
