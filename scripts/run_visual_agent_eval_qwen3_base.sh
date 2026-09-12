#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_MODELS=qwen3_base
export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RUN_ID="${RUN_ID:-qwen3_base_3bench_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/run_visual_agent_eval_qwen3_compare.sh"
