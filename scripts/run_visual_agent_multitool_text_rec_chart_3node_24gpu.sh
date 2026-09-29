#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export RL_CHART_PARSE=1
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_text_rec_chart_config.yaml}"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_multitool_text_rec_chart.txt}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_text_rec_chart_n16_3node}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    printf 'RL_CHART_PARSE=%s\n' "$RL_CHART_PARSE"
fi

exec bash "$SCRIPT_DIR/run_visual_agent_multitool_text_rec_3node_24gpu.sh" "$@"
