#!/usr/bin/env bash
set -Eeuo pipefail

# Run this entrypoint on both 8-GPU ModelArts nodes.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
export COUNT_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
export RL_ENV_DIR="${MULTITOOL_RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924}"
export TRAIN_FILES="${TRAIN_FILES:-$MULTITOOL_DATA_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$MULTITOOL_DATA_DIR/val.parquet}"
export NNODES="${NNODES:-2}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-28}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-112}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export TEST_FREQ="${TEST_FREQ:-40}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP-2}"
export MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP-1}"
export SAVE_BEST_ONLY=False
export SAVE_BEST_HF_MODEL="${SAVE_BEST_HF_MODEL:-True}"
export BEST_METRIC="${BEST_METRIC:-val-core/visual-agent/acc/macro_mean}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export POST_TRAIN_SCRIPT=""
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
export FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-1}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export GROUNDING_DINO_MAX_QUERY_WORDS="${GROUNDING_DINO_MAX_QUERY_WORDS:-12}"
export GROUNDING_QUERY_MAX_WORDS="${GROUNDING_QUERY_MAX_WORDS:-12}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_depth_tallyqa5k_multitool_pairdepth_n8_${JOB_TOKEN}}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$RL_OUTPUT_DIR/validation}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-zwz-rl/$RUN_ID}"
export RL_LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}"

export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://43.155.134.160:8080/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4-flash}"
export LLM_AS_A_JUDGE_BACKUP_BASE="${LLM_AS_A_JUDGE_BACKUP_BASE:-https://api.deepseek.com/v1}"
export LLM_AS_A_JUDGE_BACKUP_MODEL="${LLM_AS_A_JUDGE_BACKUP_MODEL:-deepseek-v4-flash}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
  source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
  for key in RL_ENV_DIR MODEL_PATH TRAIN_FILES VAL_FILES COUNT_SERVICE_CONFIG NNODES RL_CUDA_VISIBLE_DEVICES \
    TOOL_GPU RUN_ID RL_OUTPUT_DIR RL_LOG_DIR ROLLOUT_DATA_DIR RESUME_MODE TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE VAL_BATCH_SIZE \
    ROLLOUT_N TOTAL_TRAINING_STEPS TEST_FREQ SAVE_FREQ BEST_METRIC VAL_BEFORE_TRAIN \
    VISUAL_AGENT_IMAGE_TRANSPORT MAX_CONCURRENT_REQUESTS RL_SPLIT_OCR TOOL_CONFIG_PATH \
    VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE TRAINER_STOP_AFTER_SECONDS; do
    printf '%s=%s\n' "$key" "${!key:-}"
  done
  exit 0
fi

mkdir -p /opt/huawei/explorer-env /home/ma-user/work/algorithm /home/ma-user/work/model
ensure_symlink() {
  local target="$1" link_path="$2"
  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}
ensure_symlink /opt/huawei/dataset /opt/huawei/explorer-env/dataset
ensure_symlink /opt/huawei/dataset /home/ma-user/work/dataset
ensure_symlink /opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage
source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"

if [[ -z "${MULTITOOL_RL_ENV_DIR:-}" && ! -x "$RL_ENV_DIR/bin/python" ]]; then
  for dataset_root in /home/ma-user/work/dataset /opt/huawei/dataset; do
    candidate="$dataset_root/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl"
    if [[ -x "$candidate/bin/python" ]]; then
      export RL_ENV_DIR="$candidate"
      break
    fi
  done
fi
[[ -x "$RL_ENV_DIR/bin/python" ]] || { echo "error: Qwen3-VL environment is missing: $RL_ENV_DIR" >&2; exit 2; }

export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

load_key() {
  local variable_name="$1" key_file="$2" value="${!1:-}"
  if [[ -z "$value" ]]; then
    [[ -r "$key_file" ]] || { echo "error: judge key file is not readable: $key_file" >&2; exit 2; }
    IFS= read -r value < "$key_file" || true
    value="${value%$'\r'}"
    [[ -n "$value" ]] || { echo "error: judge key file is empty: $key_file" >&2; exit 2; }
    printf -v "$variable_name" '%s' "$value"
    export "$variable_name"
  fi
}
if [[ "${ENABLE_API_JUDGE:-1}" != "0" ]]; then
  load_key LLM_AS_A_JUDGE_BACKUP_KEY "${BACKUP_JUDGE_KEY_FILE:-$BASE/secrets/deepseek_api_key.txt}"
  load_key LLM_AS_A_JUDGE_KEY "${JUDGE_KEY_FILE:-$BASE/secrets/judge_api_43_155_134_160_key.txt}"
fi

OPENCV_HEADLESS_SITE_PACKAGES="${OPENCV_HEADLESS_SITE_PACKAGES:-${RL_ENV_DIR%/*}/fineanno_wdz/lib/python3.10/site-packages}"
OPENCV_HEADLESS_OVERLAY="${OPENCV_HEADLESS_OVERLAY:-/tmp/visual-agent-opencv-python-headless-5.0.0.93}"
for package in cv2 opencv_python_headless.libs; do
  [[ -d "$OPENCV_HEADLESS_SITE_PACKAGES/$package" ]] || {
    echo "error: headless OpenCV package not found: $OPENCV_HEADLESS_SITE_PACKAGES/$package" >&2
    exit 2
  }
done
mkdir -p "$OPENCV_HEADLESS_OVERLAY"
for package in cv2 opencv_python_headless.libs; do
  [[ -e "$OPENCV_HEADLESS_OVERLAY/$package" || -L "$OPENCV_HEADLESS_OVERLAY/$package" ]] || \
    ln -s "$OPENCV_HEADLESS_SITE_PACKAGES/$package" "$OPENCV_HEADLESS_OVERLAY/$package"
done
export PYTHONPATH="$OPENCV_HEADLESS_OVERLAY:${PYTHONPATH:-}"
"$RL_ENV_DIR/bin/python" - "$MODEL_PATH" <<'PYMODEL'
import sys
import cv2
from transformers import AutoProcessor, Qwen3VLProcessor

gui = next(line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
           if line.strip().startswith("GUI:"))
if gui != "NONE":
    raise RuntimeError(f"Headless OpenCV overlay is not active: GUI={gui}")
processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
if not isinstance(processor, Qwen3VLProcessor):
    raise RuntimeError(f"Expected Qwen3VLProcessor, got {type(processor).__name__}")
print(f"ModelArts preflight passed: {type(processor).__name__}, OpenCV GUI={gui}", flush=True)
PYMODEL

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh"
