#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate Qwen3-VL RL checkpoints with a separate visual-tool GPU and one or
# more model GPUs. The defaults target one eight-GPU ModelArts node.

if [[ "${SKIP_MODELARTS_BOOTSTRAP:-0}" != "1" ]]; then
  mkdir -p /opt/huawei/explorer-env /home/ma-user/work/algorithm /home/ma-user/work/model

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}

  ensure_symlink /opt/huawei/dataset /opt/huawei/explorer-env/dataset
  ensure_symlink /opt/huawei/dataset /home/ma-user/work/dataset
  ensure_symlink /opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl /home/ma-user/work/algorithm/synaflow_wl
  ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage
fi

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
ENV_DIR="${ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM}"
VLMEVAL_ENV_DIR="${VLMEVAL_ENV_DIR:-$ENV_DIR}"
VLMEVAL_PYTHON="${VLMEVAL_PYTHON:-$VLMEVAL_ENV_DIR/bin/python}"
VLMEVAL_LD_LIBRARY_PATH="${VLMEVAL_LD_LIBRARY_PATH:-$VLMEVAL_ENV_DIR/lib}"
VLMEVAL_PYTHONPATH="${VLMEVAL_PYTHONPATH-}"
TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
TOOL_CUDA_HOME="${TOOL_CUDA_HOME-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
MODEL_CUDA_HOME="${MODEL_CUDA_HOME-$CUDA_HOME}"
MODEL_CC="${MODEL_CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
MODEL_CXX="${MODEL_CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"

RL_RUN_DIR="${RL_RUN_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/eval_snapshots/zwz_original_relation_qwen3_base_2node_http_pool_v2}"
RL_STEP80_MODEL_PATH="${RL_STEP80_MODEL_PATH:-$RL_RUN_DIR/global_step_80/actor/huggingface}"
RL_STEP90_MODEL_PATH="${RL_STEP90_MODEL_PATH:-$RL_RUN_DIR/global_step_90/actor/huggingface}"
RL_DINO_STEP40_MODEL_PATH="${RL_DINO_STEP40_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_only_manual/global_step_40/actor/huggingface}"
RL_DINO_STEP50_MODEL_PATH="${RL_DINO_STEP50_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_only_manual/global_step_50/actor/huggingface}"
RL_DINO_STEP120_MODEL_PATH="${RL_DINO_STEP120_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_only_manual/global_step_120/actor/huggingface}"
RL_DINO_KL_STEP30_MODEL_PATH="${RL_DINO_KL_STEP30_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual/global_step_30/actor/huggingface}"
RL_DINO_KL_STEP50_MODEL_PATH="${RL_DINO_KL_STEP50_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual/global_step_50/actor/huggingface}"
DINO_RUN_DIR="${DINO_RUN_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_original_relation_qwen3_base_2node_groundingdino_only_manual}"
LATEST_CHECKPOINT_FILE="${LATEST_CHECKPOINT_FILE:-$DINO_RUN_DIR/latest_checkpointed_iteration.txt}"
RL_DINO_LATEST_STEP="${RL_DINO_LATEST_STEP:-}"
RL_DINO_LATEST_MODEL_PATH="${RL_DINO_LATEST_MODEL_PATH:-}"
EVAL_MODELS="${EVAL_MODELS:-rl_step80 rl_step90}"
EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K}"

if [[ " $EVAL_MODELS " == *" dino_latest "* ]]; then
  if [[ -z "$RL_DINO_LATEST_STEP" ]]; then
    [[ -r "$LATEST_CHECKPOINT_FILE" ]] || {
      echo "Error: latest checkpoint pointer is not readable: $LATEST_CHECKPOINT_FILE" >&2
      exit 2
    }
    RL_DINO_LATEST_STEP="$(<"$LATEST_CHECKPOINT_FILE")"
    RL_DINO_LATEST_STEP="${RL_DINO_LATEST_STEP%$'\r'}"
  fi
  [[ "$RL_DINO_LATEST_STEP" =~ ^[0-9]+$ ]] || {
    echo "Error: invalid latest checkpoint step: $RL_DINO_LATEST_STEP" >&2
    exit 2
  }
  RL_DINO_LATEST_MODEL_PATH="${RL_DINO_LATEST_MODEL_PATH:-$DINO_RUN_DIR/global_step_$RL_DINO_LATEST_STEP/actor/huggingface}"
fi

SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$BASE/visual-tools/sam3/sam3.pt}"
GROUNDING_DINO_MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-$BASE/visual-tools/grounding-dino-base-transformers}"
SAM3_REPLICAS="${SAM3_REPLICAS:-4}"
GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"
VISUAL_TOOL_BACKEND="${VISUAL_TOOL_BACKEND:-all}"
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
EVAL_PREFLIGHT_ONLY="${EVAL_PREFLIGHT_ONLY:-0}"
MODEL_SERVER_BACKEND="${MODEL_SERVER_BACKEND:-auto}"
REQUESTED_MODEL_SERVER_BACKEND="$MODEL_SERVER_BACKEND"
SKIP_CONDA_ACTIVATION="${SKIP_CONDA_ACTIVATION:-0}"
VLMEVAL_IMPORT_PREFLIGHT="${VLMEVAL_IMPORT_PREFLIGHT:-0}"

case "$EVAL_PREFLIGHT_ONLY" in
  0|1) ;;
  *) echo "Error: EVAL_PREFLIGHT_ONLY must be 0 or 1, got: $EVAL_PREFLIGHT_ONLY" >&2; exit 2 ;;
esac
case "$SKIP_CONDA_ACTIVATION" in
  0|1) ;;
  *) echo "Error: SKIP_CONDA_ACTIVATION must be 0 or 1, got: $SKIP_CONDA_ACTIVATION" >&2; exit 2 ;;
esac
case "$VLMEVAL_IMPORT_PREFLIGHT" in
  0|1) ;;
  *) echo "Error: VLMEVAL_IMPORT_PREFLIGHT must be 0 or 1, got: $VLMEVAL_IMPORT_PREFLIGHT" >&2; exit 2 ;;
esac

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ " $EVAL_MODELS " == *" dino_latest "* ]]; then
    RUN_ID="qwen3_dino_only_step${RL_DINO_LATEST_STEP}_3bench_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_ID="qwen3_zwz_rl_step80_90_3bench_$(date +%Y%m%d_%H%M%S)"
  fi
fi
LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-eval}"
LOG_FILE="$LOG_DIR/$RUN_ID.log"
if [[ -z "${WORK_ROOT:-}" ]]; then
  if [[ " $EVAL_MODELS " == *" dino_latest "* ]]; then
    WORK_ROOT="$REPO_ROOT/outputs/vlmeval/qwen3_dino_only_latest/$RUN_ID"
  else
    WORK_ROOT="$REPO_ROOT/outputs/vlmeval/qwen3_rl_step80_90/$RUN_ID"
  fi
fi
LMUData="${LMUData:-$REPO_ROOT/data/vlmeval}"
HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-$BASE/cache/torch}"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-$BASE/cache/triton/$RUN_ID}"

export BASE REPO_ROOT ENV_DIR VLMEVAL_ENV_DIR VLMEVAL_PYTHON TOOL_ENV_DIR CUDA_HOME TOOL_CUDA_HOME MODEL_CUDA_HOME
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
  "$VLMEVAL_PYTHON"
  "$TOOL_ENV_DIR/bin/python"
  "$REPO_ROOT/scripts/serve_visual_agent_model.sh"
  "$REPO_ROOT/scripts/serve_visual_agent_transformers.py"
  "$REPO_ROOT/scripts/visual_tool_server.py"
  "$REPO_ROOT/evaluation/VLMEvalKit/run_visual_agent_benchmarks.sh"
  "$GROUNDING_DINO_MODEL_PATH/config.json"
  "$MODEL_CC"
  "$MODEL_CXX"
)
for required in "${required_paths[@]}"; do
  [[ -e "$required" ]] || { echo "Error: missing required path: $required"; exit 2; }
done
if [[ " $EVAL_MODELS " == *" rl_step80 "* ]]; then
  [[ -f "$RL_STEP80_MODEL_PATH/config.json" && -f "$RL_STEP80_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete RL step80 model: $RL_STEP80_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" rl_step90 "* ]]; then
  [[ -f "$RL_STEP90_MODEL_PATH/config.json" && -f "$RL_STEP90_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete RL step90 model: $RL_STEP90_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_step40 "* ]]; then
  [[ -f "$RL_DINO_STEP40_MODEL_PATH/config.json" && -f "$RL_DINO_STEP40_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only step40 model: $RL_DINO_STEP40_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_step50 "* ]]; then
  [[ -f "$RL_DINO_STEP50_MODEL_PATH/config.json" && -f "$RL_DINO_STEP50_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only step50 model: $RL_DINO_STEP50_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_step120 "* ]]; then
  [[ -f "$RL_DINO_STEP120_MODEL_PATH/config.json" && -f "$RL_DINO_STEP120_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only step120 model: $RL_DINO_STEP120_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_kl_step30 "* ]]; then
  [[ -f "$RL_DINO_KL_STEP30_MODEL_PATH/config.json" && -f "$RL_DINO_KL_STEP30_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only KL step30 model: $RL_DINO_KL_STEP30_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_kl_step50 "* ]]; then
  [[ -f "$RL_DINO_KL_STEP50_MODEL_PATH/config.json" && -f "$RL_DINO_KL_STEP50_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only KL step50 model: $RL_DINO_KL_STEP50_MODEL_PATH"
    exit 2
  }
fi
if [[ " $EVAL_MODELS " == *" dino_latest "* ]]; then
  [[ "$RL_DINO_LATEST_STEP" =~ ^[0-9]+$ ]] || {
    echo "Error: invalid DINO-only latest step: $RL_DINO_LATEST_STEP"
    exit 2
  }
  [[ -f "$RL_DINO_LATEST_MODEL_PATH/config.json" && -f "$RL_DINO_LATEST_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "Error: incomplete DINO-only latest model: $RL_DINO_LATEST_MODEL_PATH"
    exit 2
  }
fi
if [[ "$VISUAL_TOOL_BACKEND" != "groundingdino" && ! -e "$SAM3_MODEL_PATH" ]]; then
  echo "Error: missing required path: $SAM3_MODEL_PATH"
  exit 2
fi
if [[ -n "${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-}" && ! -s "$VISUAL_AGENT_SYSTEM_PROMPT_FILE" ]]; then
  echo "Error: missing or empty agent prompt: $VISUAL_AGENT_SYSTEM_PROMPT_FILE"
  exit 2
fi
if [[ "$VISUAL_TOOL_BACKEND" == "groundingdino" ]]; then
  VISUAL_AGENT_ALLOWED_TOOL_NAMES="${VISUAL_AGENT_ALLOWED_TOOL_NAMES:-grounding_detect,crop_zoom}"
  [[ -n "${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-}" ]] || {
    echo "Error: groundingdino evaluation requires VISUAL_AGENT_SYSTEM_PROMPT_FILE"
    exit 2
  }
  if grep -q 'sam3_' "$VISUAL_AGENT_SYSTEM_PROMPT_FILE"; then
    echo "Error: groundingdino-only prompt exposes a SAM3 tool: $VISUAL_AGENT_SYSTEM_PROMPT_FILE"
    exit 2
  fi
fi
export VISUAL_AGENT_ALLOWED_TOOL_NAMES
command -v setsid >/dev/null || { echo "Error: setsid is required for bounded process cleanup"; exit 2; }

MINICONDA_PATH=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3
CONDA_ACTIVATION_AVAILABLE=0
if [[ "$SKIP_CONDA_ACTIVATION" == "1" ]]; then
  [[ -x "$ENV_DIR/bin/python" ]] || {
    echo "Error: direct Conda-prefix mode requires an executable Python: $ENV_DIR/bin/python"
    exit 2
  }
  echo "Using Conda prefix directly without activation: $ENV_DIR"
elif [[ -f "$MINICONDA_PATH/etc/profile.d/conda.sh" ]]; then
  export PATH="$MINICONDA_PATH/bin:$PATH"
  source "$MINICONDA_PATH/etc/profile.d/conda.sh"
  CONDA_ACTIVATION_AVAILABLE=1
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  CONDA_ACTIVATION_AVAILABLE=1
elif [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Error: conda.sh was not found and ENV_DIR is not an executable prefix: $ENV_DIR"
  exit 2
else
  echo "Using Conda prefix directly without activation: $ENV_DIR"
fi

if [[ "$CONDA_ACTIVATION_AVAILABLE" == "1" ]]; then
  set +u
  conda activate "$ENV_DIR"
  conda_status=$?
  set -u
  [[ "$conda_status" -eq 0 ]] || { echo "Error: unable to activate $ENV_DIR"; exit "$conda_status"; }
fi

export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="$ENV_DIR/bin/python"
EVAL_PYTHON="$VLMEVAL_PYTHON"
TOOL_PYTHON="$TOOL_ENV_DIR/bin/python"

case "$MODEL_SERVER_BACKEND" in
  auto)
    if "$PYTHON_BIN" -c 'import vllm' >/dev/null 2>&1; then
      MODEL_SERVER_BACKEND=vllm
    else
      MODEL_SERVER_BACKEND=transformers
    fi
    ;;
  vllm)
    "$PYTHON_BIN" -c 'import vllm' >/dev/null 2>&1 || {
      echo "Error: vLLM is unavailable in $ENV_DIR"
      exit 2
    }
    ;;
  transformers)
    "$PYTHON_BIN" -c 'import fastapi, qwen_vl_utils, torch, transformers, uvicorn' >/dev/null 2>&1 || {
      echo "Error: Transformers server dependencies are unavailable in $ENV_DIR"
      exit 2
    }
    ;;
  *) echo "Error: MODEL_SERVER_BACKEND must be auto, vllm, or transformers"; exit 2 ;;
esac

IFS=',' read -r -a MODEL_GPUS <<< "$MODEL_CUDA_VISIBLE_DEVICES"
[[ "${#MODEL_GPUS[@]}" -gt 0 ]] || { echo "Error: no model GPUs configured"; exit 2; }
VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-${#MODEL_GPUS[@]}}"
export VLMEVAL_API_NPROC

validate_vlmeval_imports() {
  (
    cd "$REPO_ROOT/evaluation/VLMEvalKit"
    env PYTHONPATH="$VLMEVAL_PYTHONPATH" LD_LIBRARY_PATH="$VLMEVAL_LD_LIBRARY_PATH" \
      "$EVAL_PYTHON" -c 'from vlmeval.config import supported_VLM; from vlmeval.dataset import SUPPORTED_DATASETS; print(f"VLMEvalKit imports ready: models={len(supported_VLM)} datasets={len(SUPPORTED_DATASETS)}")'
  )
}

if [[ "$EVAL_PREFLIGHT_ONLY" == "1" || "$VLMEVAL_IMPORT_PREFLIGHT" == "1" ]]; then
  "$PYTHON_BIN" -c 'import pandas, torch, transformers'
  "$TOOL_PYTHON" -c 'import PIL, torch, transformers'
  validate_vlmeval_imports
fi

if [[ "$EVAL_PREFLIGHT_ONLY" == "1" ]]; then
  echo "Evaluation preflight passed"
  echo "Model environment: $ENV_DIR"
  echo "VLMEval environment: $VLMEVAL_ENV_DIR"
  echo "Model CUDA: $MODEL_CUDA_HOME"
  echo "Tool environment: $TOOL_ENV_DIR"
  echo "Tool CUDA: $TOOL_CUDA_HOME"
  echo "Model server backend: $MODEL_SERVER_BACKEND"
  echo "Checkpoint: ${RL_DINO_LATEST_MODEL_PATH:-not selected}"
  exit 0
fi

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

start_model_servers() {
  local backend="$1" index gpu port replica_cache
  MODEL_PIDS=()
  model_api_bases=()
  for index in "${!MODEL_GPUS[@]}"; do
    gpu="${MODEL_GPUS[$index]}"
    port=$((MODEL_PORT_BASE + index))
    replica_cache="$TRITON_CACHE_ROOT/${variant}_gpu_${gpu}"
    mkdir -p "$replica_cache"
    if [[ "$backend" == "vllm" ]]; then
      setsid env \
        CUDA_VISIBLE_DEVICES="$gpu" MODEL_PATH="$model_path" SERVED_MODEL_NAME="$served_model" \
        HOST="$HOST" PORT="$port" TENSOR_PARALLEL_SIZE=1 \
        GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        LIMIT_MM_PER_PROMPT="$LIMIT_MM_PER_PROMPT" CC="$MODEL_CC" CXX="$MODEL_CXX" \
        CUDAHOSTCXX="$MODEL_CXX" VISUAL_AGENT_CUDA_HOME="$MODEL_CUDA_HOME" \
        VLLM_PYTHON="$PYTHON_BIN" \
        TRITON_CACHE_DIR="$replica_cache" \
        bash "$REPO_ROOT/scripts/serve_visual_agent_model.sh" &
    else
      setsid env CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" "$REPO_ROOT/scripts/serve_visual_agent_transformers.py" \
          --model "$model_path" --served-model-name "$served_model" \
          --host "$HOST" --port "$port" &
    fi
    MODEL_PIDS+=("$!")
    model_api_bases+=("http://127.0.0.1:$port/v1")
  done
}

wait_model_servers() {
  local backend="$1" index
  for index in "${!model_api_bases[@]}"; do
    wait_http "${model_api_bases[$index]}/models" \
      "$variant $backend replica $index" "${MODEL_PIDS[$index]}" || return 1
  done
}

results_ready() {
  local work_dir="$1" dataset result_name
  for dataset in $EVAL_DATASETS; do
    case "$dataset" in
      OCRBench*) result_name="VisualAgent-vllm_${dataset}_score.json" ;;
      MME-RealWorld*) result_name="VisualAgent-vllm_${dataset}_rating.json" ;;
      *) result_name="VisualAgent-vllm_${dataset}_acc.csv" ;;
    esac
    find "$work_dir" -type f -name "$result_name" -size +0c -print -quit | grep -q . || return 1
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
  export VISUAL_AGENT_SYSTEM_PROMPT_FILE VISUAL_AGENT_ALLOWED_TOOL_NAMES
  export VLMEVAL_PYTHON="$EVAL_PYTHON"

  cd "$REPO_ROOT/evaluation/VLMEvalKit"
  setsid env PYTHONPATH="$VLMEVAL_PYTHONPATH" LD_LIBRARY_PATH="$VLMEVAL_LD_LIBRARY_PATH" \
    bash ./run_visual_agent_benchmarks.sh "${dataset_args[@]}" &
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
    if (( EVAL_TIMEOUT_SECONDS > 0 && now - started >= EVAL_TIMEOUT_SECONDS )); then
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
echo "Qwen3-VL RL evaluation"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Models: $EVAL_MODELS"
echo "RL step 80 model: $RL_STEP80_MODEL_PATH"
echo "RL step 90 model: $RL_STEP90_MODEL_PATH"
echo "DINO-only step 40 model: $RL_DINO_STEP40_MODEL_PATH"
echo "DINO-only step 50 model: $RL_DINO_STEP50_MODEL_PATH"
echo "DINO-only step 120 model: $RL_DINO_STEP120_MODEL_PATH"
echo "DINO-only KL step 30 model: $RL_DINO_KL_STEP30_MODEL_PATH"
echo "DINO-only KL step 50 model: $RL_DINO_KL_STEP50_MODEL_PATH"
echo "DINO-only latest model: step ${RL_DINO_LATEST_STEP:-unset} at ${RL_DINO_LATEST_MODEL_PATH:-unset}"
echo "Datasets: $EVAL_DATASETS"
echo "Model GPUs: $MODEL_CUDA_VISIBLE_DEVICES"
echo "Tool GPU: $TOOL_CUDA_VISIBLE_DEVICES"
echo "Tool backend: $VISUAL_TOOL_BACKEND"
echo "Agent system prompt: ${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-built-in default}"
echo "Allowed tools: ${VISUAL_AGENT_ALLOWED_TOOL_NAMES:-all}"
echo "Model server backend: $MODEL_SERVER_BACKEND"
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
    --backend "$VISUAL_TOOL_BACKEND" --host "$HOST" --port "$TOOL_PORT" &
TOOL_PID=$!
wait_http "http://127.0.0.1:$TOOL_PORT/health" "visual tool server" "$TOOL_PID"

for variant in $EVAL_MODELS; do
  case "$variant" in
    rl_step80)
      model_path="$RL_STEP80_MODEL_PATH"
      served_model="visual-agent-rl-step80"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    rl_step90)
      model_path="$RL_STEP90_MODEL_PATH"
      served_model="visual-agent-rl-step90"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_step40)
      model_path="$RL_DINO_STEP40_MODEL_PATH"
      served_model="visual-agent-dino-step40"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_step50)
      model_path="$RL_DINO_STEP50_MODEL_PATH"
      served_model="visual-agent-dino-step50"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_step120)
      model_path="$RL_DINO_STEP120_MODEL_PATH"
      served_model="visual-agent-dino-step120"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_kl_step30)
      model_path="$RL_DINO_KL_STEP30_MODEL_PATH"
      served_model="visual-agent-dino-kl-step30"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_kl_step50)
      model_path="$RL_DINO_KL_STEP50_MODEL_PATH"
      served_model="visual-agent-dino-kl-step50"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    dino_latest)
      model_path="$RL_DINO_LATEST_MODEL_PATH"
      served_model="visual-agent-dino-step${RL_DINO_LATEST_STEP}"
      VISUAL_AGENT_USE_TOOLS=true
      VISUAL_AGENT_INFERENCE_MODE=agent
      ;;
    *)
      echo "Error: unsupported EVAL_MODELS entry: $variant"
      exit 2
      ;;
  esac

  variant_work_dir="$WORK_ROOT/$variant"
  mkdir -p "$variant_work_dir"
  echo "Starting model variant $variant from $model_path"
  start_model_servers "$MODEL_SERVER_BACKEND"
  if ! wait_model_servers "$MODEL_SERVER_BACKEND"; then
    if [[ "$REQUESTED_MODEL_SERVER_BACKEND" == "auto" && "$MODEL_SERVER_BACKEND" == "vllm" ]]; then
      echo "vLLM failed to become ready; retrying $variant with Transformers"
      stop_models
      "$PYTHON_BIN" -c 'import fastapi, qwen_vl_utils, torch, transformers, uvicorn' >/dev/null 2>&1 || {
        echo "Error: Transformers fallback dependencies are unavailable in $ENV_DIR"
        exit 2
      }
      MODEL_SERVER_BACKEND=transformers
      start_model_servers "$MODEL_SERVER_BACKEND"
      wait_model_servers "$MODEL_SERVER_BACKEND"
    else
      exit 1
    fi
  fi
  VISUAL_AGENT_API_BASE="$(IFS=,; echo "${model_api_bases[*]}")"
  VISUAL_AGENT_MODEL="$served_model"
  export VISUAL_AGENT_API_BASE VISUAL_AGENT_MODEL VISUAL_AGENT_USE_TOOLS VISUAL_AGENT_INFERENCE_MODE
  echo "Starting $variant benchmarks ($VISUAL_AGENT_INFERENCE_MODE mode): $EVAL_DATASETS"
  run_benchmarks "$variant_work_dir"
  echo "Completed $variant; results: $variant_work_dir"
  stop_models
done

echo "All requested model evaluations completed"
