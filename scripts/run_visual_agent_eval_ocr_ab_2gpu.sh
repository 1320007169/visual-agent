#!/usr/bin/env bash
set -Eeuo pipefail

# Run both full OCRBench arms locally with one model GPU and one tool GPU.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_MODELARTS_BOOTSTRAP=1
export MODEL_CUDA_VISIBLE_DEVICES=0 TOOL_CUDA_VISIBLE_DEVICES=1
export VLMEVAL_API_NPROC=1
export ENV_DIR=/home/ma-user/work/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM
export VLOCR_DEPTH_ENV=/home/ma-user/work/dataset/Common_wl/miniconda3/envs/starVLA_flash_dzw1
export TOOL_CUDA_HOME=/home/ma-user/work/dataset/trellis_ckpt/cuda/cuda118
exec bash "$SCRIPT_DIR/run_visual_agent_eval_ocr_hme_ab_8gpu_modelarts.sh" "$@"
