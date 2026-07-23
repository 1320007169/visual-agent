#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the Qwen3-VL SFT checkpoint and the original Qwen3-VL model in one
# eight-GPU ModelArts job. GPU 0 serves visual tools; GPUs 1-7 serve replicated
# single-GPU vLLM endpoints. Models run sequentially and use separate outputs.

mkdir -p /opt/huawei/explorer-env /home/ma-user/work/model
[[ -e /opt/huawei/explorer-env/dataset ]] || ln -s /opt/huawei/dataset /opt/huawei/explorer-env/dataset
[[ -e /home/ma-user/work/dataset ]] || ln -s /opt/huawei/dataset /home/ma-user/work/dataset
[[ -e /home/ma-user/work/model/xiaoyi_tmpstorage ]] || \
  ln -s /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
ENV_DIR="${ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM}"
TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
CUDA_HOME="${CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-$CUDA_HOME}"
MODEL_CUDA_HOME="${MODEL_CUDA_HOME:-}"
MODEL_CC="${MODEL_CC:-$BASE/conda_envs/deepeyes-sft-conda/bin/gcc}"
MODEL_CXX="${MODEL_CXX:-$BASE/conda_envs/deepeyes-sft-conda/bin/g++}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_combined_8679/qwen3/full/sft_visual_sft_20260719_154704}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
EVAL_MODELS="${EVAL_MODELS:-sft qwen3_base}"
EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K}"

SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$BASE/visual-tools/sam3/sam3.pt}"
GROUNDING_DINO_MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-$BASE/visual-tools/grounding-dino-base-transformers}"
SAM3_REPLICAS="${SAM3_REPLICAS:-4}"
GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"
TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
MODEL_PORT_BASE="${MODEL_PORT_BASE:-8000}"
TOOL_PORT="${TOOL_PORT:-9000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 16}}"
VISUAL_AGENT_MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-8}"
VISUAL_AGENT_MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-4096}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-43200}"
EVAL_EXIT_GRACE_SECONDS="${EVAL_EXIT_GRACE_SECONDS:-120}"

RUN_ID="${RUN_ID:-qwen3_sft_vs_base_3bench_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-eval}"
LOG_FILE="$LOG_DIR/$RUN_ID.log"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/qwen3_comparison/$RUN_ID}"
LMUData="${LMUData:-$REPO_ROOT/data/vlmeval}"
HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-$BASE/cache/torch}"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-$BASE/cache/triton/$RUN_ID}"

export BASE REPO_ROOT ENV_DIR TOOL_ENV_DIR CUDA_HOME TOOL_CUDA_HOME MODEL_CUDA_HOME
export MODEL_CC MODEL_CXX CUDAHOSTCXX="$MODEL_CXX"
export LMUData HF_HOME XDG_CACHE_HOME TORCH_HOME
export TMPDIR="${TMPDIR:-/tmp}" TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export WANDB_DISABLED=true WANDB_MODE=disabled REPORT_TO=none

mkdir -p "$LOG_DIR" "$WORK_ROOT" "$LMUData" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_ROOT"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

required_paths=(
  "$ENV_DIR/bin/python"
  "$TOOL_ENV_DIR/bin/python"
  "$REPO_ROOT/scripts/serve_visual_agent_model.sh"
  "$REPO_ROOT/scripts/visual_tool_server.py"
  "$REPO_ROOT/evaluation/VLMEvalKit/run_visual_agent_benchmarks.sh"
  "$SFT_MODEL_PATH/config.json"
  "$SFT_MODEL_PATH/model.safetensors.index.json"
  "$QWEN3_MODEL_PATH/config.json"
  "$QWEN3_MODEL_PATH/model.safetensors.index.json"
  "$SAM3_MODEL_PATH"
  "$GROUNDING_DINO_MODEL_PATH/config.json"
  "$MODEL_CC"
  "$MODEL_CXX"
)
for required in "${required_paths[@]}"; do
  [[ -e "$required" ]] || { echo "Error: missing required path: $required"; exit 2; }
done
command -v setsid >/dev/null || { echo "Error: setsid is required for bounded process cleanup"; exit 2; }

MINICONDA_PATH=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3
if [[ -f "$MINICONDA_PATH/etc/profile.d/conda.sh" ]]; then
  export PATH="$MINICONDA_PATH/bin:$PATH"
  source "$MINICONDA_PATH/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "Error: conda.sh was not found"
  exit 2
fi

set +u
conda activate "$ENV_DIR"
conda_status=$?
set -u
[[ "$conda_status" -eq 0 ]] || { echo "Error: unable to activate $ENV_DIR"; exit "$conda_status"; }

export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="$ENV_DIR/bin/python"
TOOL_PYTHON="$TOOL_ENV_DIR/bin/python"

IFS=',' read -r -a MODEL_GPUS <<< "$MODEL_CUDA_VISIBLE_DEVICES"
[[ "${#MODEL_GPUS[@]}" -gt 0 ]] || { echo "Error: no model GPUs configured"; exit 2; }
VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-${#MODEL_GPUS[@]}}"
export VLMEVAL_API_NPROC

TOOL_PID=""
EVAL_PID=""
MODEL_PIDS=()

stop_group() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 -- "-$pid" 2>/dev/null && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

stop_models() {
  local pid
  for pid in "${MODEL_PIDS[@]:-}"; do stop_group "$pid"; done
  MODEL_PIDS=()
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$EVAL_PID"
  stop_models
  stop_group "$TOOL_PID"
  echo "Evaluation end time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Evaluation exit code: $status"
  echo "Log file: $LOG_FILE"
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1" name="$2" pid="$3"
  for _ in $(seq 1 300); do
    if "$PYTHON_BIN" - "$url" <<'PY' 2>/dev/null
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
        raise SystemExit(0 if response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "$name is ready: $url"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo "$name exited before becoming ready"; return 1; }
    sleep 2
  done
  echo "Timed out waiting for $name: $url"
  return 1
}

results_ready() {
  local work_dir="$1" dataset
  for dataset in $EVAL_DATASETS; do
    find "$work_dir" -type f -name "VisualAgent-vllm_${dataset}_acc.csv" -size +0c -print -quit | grep -q . || return 1
  done
}

run_benchmarks() {
  local work_dir="$1"
  local started now ready_at=0 status=0
  read -r -a dataset_args <<< "$EVAL_DATASETS"
  export VLMEVAL_WORK_DIR="$work_dir"
  export VISUAL_AGENT_API_BASE
  export VISUAL_TOOL_API_BASE="http://127.0.0.1:$TOOL_PORT"
  export VISUAL_AGENT_MODEL
  export VISUAL_AGENT_MAX_TURNS VISUAL_AGENT_MAX_TOKENS
  export VISUAL_AGENT_USE_TOOLS VISUAL_AGENT_INFERENCE_MODE

  cd "$REPO_ROOT/evaluation/VLMEvalKit"
  setsid bash ./run_visual_agent_benchmarks.sh "${dataset_args[@]}" &
  EVAL_PID=$!
  started=$(date +%s)
  while kill -0 "$EVAL_PID" 2>/dev/null; do
    now=$(date +%s)
    if results_ready "$work_dir"; then
      if [[ "$ready_at" -eq 0 ]]; then
        ready_at=$now
        echo "All benchmark result CSV files are ready; waiting up to ${EVAL_EXIT_GRACE_SECONDS}s for VLMEvalKit to exit"
      elif (( now - ready_at >= EVAL_EXIT_GRACE_SECONDS )); then
        echo "VLMEvalKit did not exit after writing all results; terminating its process group"
        stop_group "$EVAL_PID"
        EVAL_PID=""
        return 0
      fi
    fi
    if (( now - started >= EVAL_TIMEOUT_SECONDS )); then
      echo "Evaluation exceeded ${EVAL_TIMEOUT_SECONDS}s"
      stop_group "$EVAL_PID"
      EVAL_PID=""
      return 1
    fi
    sleep 10
  done
  wait "$EVAL_PID" || status=$?
  EVAL_PID=""
  if [[ "$status" -ne 0 ]] && ! results_ready "$work_dir"; then
    return "$status"
  fi
  results_ready "$work_dir"
}

echo "============================================================"
echo "Qwen3-VL SFT vs base evaluation"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Models: $EVAL_MODELS"
echo "SFT model: $SFT_MODEL_PATH"
echo "Qwen3 base model: $QWEN3_MODEL_PATH"
echo "Datasets: $EVAL_DATASETS"
echo "Model GPUs: $MODEL_CUDA_VISIBLE_DEVICES"
echo "Tool GPU: $TOOL_CUDA_VISIBLE_DEVICES"
echo "Work root: $WORK_ROOT"
echo "============================================================"

setsid env \
  CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" \
  CUDA_HOME="$TOOL_CUDA_HOME" \
  PATH="$TOOL_ENV_DIR/bin:$TOOL_CUDA_HOME/bin:$PATH" \
  LD_LIBRARY_PATH="$TOOL_ENV_DIR/lib:$TOOL_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
  SAM3_MODEL_PATH="$SAM3_MODEL_PATH" SAM3_DEVICE=cuda:0 SAM3_REPLICAS="$SAM3_REPLICAS" \
  GROUNDING_DINO_MODEL_PATH="$GROUNDING_DINO_MODEL_PATH" GROUNDING_DINO_DEVICE=cuda:0 \
  GROUNDING_DINO_REPLICAS="$GROUNDING_DINO_REPLICAS" \
  "$TOOL_PYTHON" "$REPO_ROOT/scripts/visual_tool_server.py" \
    --backend all --host "$HOST" --port "$TOOL_PORT" &
TOOL_PID=$!
wait_http "http://127.0.0.1:$TOOL_PORT/health" "visual tool server" "$TOOL_PID"

for variant in $EVAL_MODELS; do
  case "$variant" in
    sft)
      model_path="$SFT_MODEL_PATH"
      served_model="visual-agent-sft"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    qwen3_base)
      model_path="$QWEN3_MODEL_PATH"
      served_model="qwen3-vl-base"
      VISUAL_AGENT_USE_TOOLS=false
      VISUAL_AGENT_INFERENCE_MODE=non-think
      ;;
    *)
      echo "Error: unsupported EVAL_MODELS entry: $variant"
      exit 2
      ;;
  esac

  variant_work_dir="$WORK_ROOT/$variant"
  mkdir -p "$variant_work_dir"
  MODEL_PIDS=()
  model_api_bases=()
  echo "Starting model variant $variant from $model_path"
  for index in "${!MODEL_GPUS[@]}"; do
    gpu="${MODEL_GPUS[$index]}"
    port=$((MODEL_PORT_BASE + index))
    replica_cache="$TRITON_CACHE_ROOT/${variant}_gpu_${gpu}"
    mkdir -p "$replica_cache"
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpu" MODEL_PATH="$model_path" SERVED_MODEL_NAME="$served_model" \
      HOST="$HOST" PORT="$port" TENSOR_PARALLEL_SIZE=1 \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      LIMIT_MM_PER_PROMPT="$LIMIT_MM_PER_PROMPT" CC="$MODEL_CC" CXX="$MODEL_CXX" \
      CUDAHOSTCXX="$MODEL_CXX" VISUAL_AGENT_CUDA_HOME="$MODEL_CUDA_HOME" \
      TRITON_CACHE_DIR="$replica_cache" \
      bash "$REPO_ROOT/scripts/serve_visual_agent_model.sh" &
    MODEL_PIDS+=("$!")
    model_api_bases+=("http://127.0.0.1:$port/v1")
  done

  for index in "${!model_api_bases[@]}"; do
    wait_http "${model_api_bases[$index]}/models" "$variant vLLM replica $index" "${MODEL_PIDS[$index]}"
  done
  VISUAL_AGENT_API_BASE="$(IFS=,; echo "${model_api_bases[*]}")"
  VISUAL_AGENT_MODEL="$served_model"
  export VISUAL_AGENT_API_BASE VISUAL_AGENT_MODEL VISUAL_AGENT_USE_TOOLS VISUAL_AGENT_INFERENCE_MODE
  echo "Starting $variant benchmarks ($VISUAL_AGENT_INFERENCE_MODE mode): $EVAL_DATASETS"
  run_benchmarks "$variant_work_dir"
  echo "Completed $variant; results: $variant_work_dir"
  stop_models
done

echo "All requested model evaluations completed"
