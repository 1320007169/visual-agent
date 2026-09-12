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
USE_TOOLS="${VISUAL_AGENT_USE_TOOLS:-true}"
INFERENCE_MODE="${VISUAL_AGENT_INFERENCE_MODE:-agent}"
WORK_DIR="${VLMEVAL_WORK_DIR:-$VISUAL_AGENT_ROOT/outputs/vlmeval}"
PYTHON_BIN="${VLMEVAL_PYTHON:-python}"
SYSTEM_PROMPT_FILE="${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-}"
ALLOWED_TOOL_NAMES="${VISUAL_AGENT_ALLOWED_TOOL_NAMES:-}"
export LMUData="${LMUData:-$VISUAL_AGENT_ROOT/data/vlmeval}"

mkdir -p "$LMUData" "$WORK_DIR"

if [[ "$#" -eq 0 ]]; then
  set -- VStarBench HRBench4K HRBench8K
fi

case "$USE_TOOLS" in
  true|false) ;;
  *) echo "VISUAL_AGENT_USE_TOOLS must be true or false, got: $USE_TOOLS" >&2; exit 2 ;;
esac

MODEL_CONFIGS="$("$PYTHON_BIN" - \
  "$MODEL_API_BASE" "$TOOL_API_BASE" "$MODEL_NAME" \
  "$MAX_TURNS" "$MAX_TOKENS" "$USE_TOOLS" \
  "$SYSTEM_PROMPT_FILE" "$ALLOWED_TOOL_NAMES" <<'PYCONFIG'
import json
import sys

model_api, tool_api, model, turns, tokens, use_tools, prompt_file, allowed = sys.argv[1:]
config = {
    "api_base": model_api,
    "tool_api_base": tool_api,
    "model": model,
    "max_turns": int(turns),
    "max_tokens": int(tokens),
    "use_tools": use_tools == "true",
}
if prompt_file:
    config["system_prompt_file"] = prompt_file
if allowed:
    config["allowed_tool_names"] = [name.strip() for name in allowed.split(",") if name.strip()]
print(json.dumps({"VisualAgent-vllm": config}))
PYCONFIG
)"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" run.py \
  --model VisualAgent-vllm \
  --data "$@" \
  --work-dir "$WORK_DIR" \
  --inference-mode "$INFERENCE_MODE" \
  --api-nproc "$API_NPROC" \
  --model-configs "$MODEL_CONFIGS" \
  --judge exact_matching \
  --reuse
