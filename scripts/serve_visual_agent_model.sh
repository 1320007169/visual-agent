#!/usr/bin/env bash
set -euo pipefail

# Serve a fine-tuned Qwen2.5-VL checkpoint through an OpenAI-compatible API.
# The visual tools are separate services and are not started by this script.

: "${MODEL_PATH:?Set MODEL_PATH to the merged SFT/RL checkpoint directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$ROOT_DIR/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-visual-agent}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-16}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-image=$MAX_IMAGES_PER_PROMPT}"

# Triton compiles a tiny launcher during the first multimodal profile. Prefer
# the complete compiler from the active conda environment over the host GCC,
# which is often incomplete on managed notebook images.
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/gcc" ]]; then
  export CC="${CC:-$CONDA_PREFIX/bin/gcc}"
  export CXX="${CXX:-$CONDA_PREFIX/bin/g++}"
fi

# vLLM V1 uses Unix-domain sockets under TMPDIR; Linux limits these paths to
# roughly 107 characters, so do not inherit the long source-build directory.
export TMPDIR="${VLLM_TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE_ROOT/cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"

if [[ -n "${VISUAL_AGENT_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$VISUAL_AGENT_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  --trust-remote-code
