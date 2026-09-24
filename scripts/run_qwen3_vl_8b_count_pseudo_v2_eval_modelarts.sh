#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated CountGD++ pseudo-exemplar evaluation entrypoint. Reuses the
# five-tool launcher while swapping in the pseudo-exemplar count backend and
# the count-aware v2 prompt, and pins a dedicated run id so no existing
# prediction directory is reused or overwritten.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"

export COUNT_SERVICE_CONFIG="${COUNT_SERVICE_CONFIG:-$PIPELINE_ROOT/configs/services/countgd_plusplus_pseudo_eval.yaml}"
export FIVE_TOOL_PROMPT_FILE="${FIVE_TOOL_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_eval_five_tools_count_v2.txt}"
export FIVE_TOOL_RUN_ID="${FIVE_TOOL_RUN_ID:-qwen3_vl_8b_count_pseudo_v2_$(date +%Y%m%d_%H%M%S)}"
export EVAL_DATASETS="${EVAL_DATASETS:-MME-RealWorld-Lite}"

exec bash "$REPO_ROOT/scripts/run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh" "$@"
