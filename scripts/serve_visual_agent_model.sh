#!/usr/bin/env bash
set -euo pipefail

# Serve a fine-tuned Qwen2.5-VL checkpoint through an OpenAI-compatible API.
# The visual tools are separate services and are not started by this script.

: "${MODEL_PATH:?Set MODEL_PATH to the merged SFT/RL checkpoint directory}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-visual-agent}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-16}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt "image=$MAX_IMAGES_PER_PROMPT" \
  --trust-remote-code
