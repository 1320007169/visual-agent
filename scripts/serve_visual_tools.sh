#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# With two visible GPUs, SAM3 defaults to cuda:0 and GroundingDINO to cuda:1.
# Override the model paths and logical CUDA devices without editing this file.

export VISUAL_TOOL_BACKEND="${VISUAL_TOOL_BACKEND:-all}"
export VISUAL_TOOL_HOST="${VISUAL_TOOL_HOST:-0.0.0.0}"
export VISUAL_TOOL_PORT="${VISUAL_TOOL_PORT:-9000}"
export SAM3_DEVICE="${SAM3_DEVICE:-cuda:0}"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda:1}"
export GROUNDING_DINO_MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-IDEA-Research/grounding-dino-base}"

exec python "$ROOT_DIR/scripts/visual_tool_server.py" \
  --backend "$VISUAL_TOOL_BACKEND" \
  --host "$VISUAL_TOOL_HOST" \
  --port "$VISUAL_TOOL_PORT"
