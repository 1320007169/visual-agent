#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RUN_ID="${RUN_ID:-multitool_ocr_24gpu_step40_$(date +%Y%m%dT%H%M%S%N)_$$}"
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/multitool_ocr_24gpu_step40_8gpu/$RUN_ID}"
export EVAL_CHECKPOINTS=24gpu_step40
export VLMEVAL_FAILED_SAMPLE_RETRIES="${VLMEVAL_FAILED_SAMPLE_RETRIES:-3}"
export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/24gpu_step40/failures"
export VISUAL_AGENT_FAILURE_REPORT=1
export FSC147_DOWNLOAD_IMAGES=0

exec bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_ocr_step20_8gpu.sh" "$@"
