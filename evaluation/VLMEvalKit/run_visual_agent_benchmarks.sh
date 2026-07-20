#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUAL_AGENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_API_BASE="${VISUAL_AGENT_API_BASE:-http://127.0.0.1:8000/v1}"
TOOL_API_BASE="${VISUAL_TOOL_API_BASE:-http://127.0.0.1:9000}"
MODEL_NAME="${VISUAL_AGENT_MODEL:-visual-agent}"
API_NPROC="${VLMEVAL_API_NPROC:-1}"
MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-8}"
MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-4096}"
WORK_DIR="${VLMEVAL_WORK_DIR:-$VISUAL_AGENT_ROOT/outputs/vlmeval}"
export LMUData="${LMUData:-$VISUAL_AGENT_ROOT/data/vlmeval}"

mkdir -p "$LMUData" "$WORK_DIR"

if [[ "$#" -eq 0 ]]; then
  set -- VStarBench HRBench4K HRBench8K
fi

MODEL_CONFIGS="{\"VisualAgent-vllm\":{\"api_base\":\"$MODEL_API_BASE\",\"tool_api_base\":\"$TOOL_API_BASE\",\"model\":\"$MODEL_NAME\",\"max_turns\":$MAX_TURNS,\"max_tokens\":$MAX_TOKENS}}"

cd "$SCRIPT_DIR"
exec python run.py \
  --model VisualAgent-vllm \
  --data "$@" \
  --work-dir "$WORK_DIR" \
  --inference-mode agent \
  --api-nproc "$API_NPROC" \
  --model-configs "$MODEL_CONFIGS" \
  --judge exact_matching \
  --reuse
