#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export RL_SPLIT_OCR=0
export RL_CHART_PARSE=1
export VTS_VL_OCR=1
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml}"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-36000}"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_n16_3node_${JOB_TOKEN}}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    for key in RL_SPLIT_OCR RL_CHART_PARSE VTS_VL_OCR TOOL_CONFIG_PATH VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE TRAINER_STOP_AFTER_SECONDS; do
        printf '%s=%s\n' "$key" "${!key}"
    done
fi

exec bash "$SCRIPT_DIR/run_visual_agent_multitool_depth_count_3node_24gpu.sh" "$@"
