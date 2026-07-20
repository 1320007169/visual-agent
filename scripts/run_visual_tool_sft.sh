#!/bin/bash
set -o pipefail

# Huawei platform paths. Override any value from the job environment if needed.
PLATFORM_SETUP="${PLATFORM_SETUP:-1}"
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
ROOT_DIR="${ROOT_DIR:-$BASE/visual-agent}"
ENV_DIR="${ENV_DIR-$BASE/conda_envs/deepeyes-sft-conda}"
MODEL_DIR="${MODEL_DIR-$BASE/DeepEyesV2/models/Qwen2.5-VL-7B-Instruct}"
REPO_ROOT="$ROOT_DIR"
PLATFORM_WORK_ROOT="$BASE"
PLATFORM_DATASET_ROOT="${PLATFORM_DATASET_ROOT:-/opt/huawei/dataset}"
PLATFORM_ALGORITHM_ROOT="${PLATFORM_ALGORITHM_ROOT:-/opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl}"
PLATFORM_MODEL_ROOT="${PLATFORM_MODEL_ROOT:-/opt/huawei/quoteModel/xiaoyi_tmpstorage}"
CUDA_HOME="${VISUAL_TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH-$ENV_DIR}"
CONDA_SH="${CONDA_SH:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH-$MODEL_DIR}"

# Current target: one node with eight GPUs.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29611}"

OFFLINE_MODE="${OFFLINE_MODE:-0}"
LOG_DIR="${LOG_DIR:-$PLATFORM_WORK_ROOT/logs/visual-tool-sft}"
PRINT_HARDWARE_INFO="${PRINT_HARDWARE_INFO:-1}"
HF_HOME="${HF_HOME:-$PLATFORM_WORK_ROOT/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PLATFORM_WORK_ROOT/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-$PLATFORM_WORK_ROOT/cache/torch}"
RUN_ID="${RUN_ID:-visual_sft_$(date +%Y%m%d_%H%M%S)}"
VISUAL_TOOL_OUTPUT_DIR="${VISUAL_TOOL_OUTPUT_DIR:-$ROOT_DIR/saves/visual_agent_parquet_sft/full/sft_$RUN_ID}"
CONFIG_PATH="$ROOT_DIR/configs/train_visual_tool_sft_full.yaml"
DATASET_NAME="visual_agent_parquet_sft"
DATASET_JSONL="$ROOT_DIR/data/visual_agent_parquet.jsonl"
DATASET_SAMPLES=3679

die() {
  echo "error: $*" >&2
  exit 2
}

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  if [[ -e "$link_path" || -L "$link_path" ]]; then
    echo "Platform path already exists: $link_path"
    return
  fi
  [[ -e "$target" ]] || die "platform symlink target does not exist: $target"
  mkdir -p "$(dirname "$link_path")"
  ln -s "$target" "$link_path"
  echo "Created platform symlink: $link_path -> $target"
}

if [[ "$PLATFORM_SETUP" == 1 ]]; then
  ensure_symlink "$PLATFORM_DATASET_ROOT" /opt/huawei/explorer-env/dataset
  ensure_symlink "$PLATFORM_DATASET_ROOT" /home/ma-user/work/dataset
  ensure_symlink "$PLATFORM_ALGORITHM_ROOT" /home/ma-user/work/algorithm/synaflow_wl
  ensure_symlink "$PLATFORM_MODEL_ROOT" /home/ma-user/work/model/xiaoyi_tmpstorage
  [[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

[[ -d "$REPO_ROOT" ]] || die "repository root does not exist: $REPO_ROOT"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
[[ -f "$DATASET_JSONL" ]] || die "dataset JSONL does not exist: $DATASET_JSONL"
[[ -f "$REPO_ROOT/scripts/train_visual_tool_sft_full.sh" ]] || {
  die "REPO_ROOT does not contain the training launcher: $REPO_ROOT"
}

if [[ -n "$CONDA_ENV_PATH" ]]; then
  if [[ -f "$CONDA_ENV_PATH/bin/activate" && ! -d "$CONDA_ENV_PATH/conda-meta" ]]; then
    # The prepared SFT environment is a venv that reuses the platform CUDA stack.
    set +u
    source "$CONDA_ENV_PATH/bin/activate"
    set -u
  else
    if [[ -z "$CONDA_SH" ]]; then
      for candidate in \
        "$MINICONDA_ROOT/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
          CONDA_SH="$candidate"
          break
        fi
      done
    fi
    [[ -f "$CONDA_SH" ]] || die "conda.sh was not found; set CONDA_SH"
    # Conda activation can reference unset shell variables.
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_PATH"
    set -u
  fi
fi

if [[ -n "$CONDA_ENV_PATH" ]]; then
  export PATH="$CONDA_ENV_PATH/bin:$CUDA_HOME/bin:$PATH"
  export LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-$CONDA_ENV_PATH/bin/llamafactory-cli}"
  export PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
  export CC="${CC:-$CONDA_ENV_PATH/bin/gcc}"
  export CXX="${CXX:-$CONDA_ENV_PATH/bin/g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
fi

if [[ "$OFFLINE_MODE" == 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

export CUDA_VISIBLE_DEVICES NNODES NPROC_PER_NODE TARGET_GLOBAL_BATCH_SIZE MASTER_PORT
export MODEL_NAME_OR_PATH
export CONFIG_PATH
export HF_HOME XDG_CACHE_HOME TORCH_HOME
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "$LOG_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"
NODE_LOG_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
LOG_FILE="$LOG_DIR/train-node${NODE_LOG_RANK}-$(date +%Y%m%d_%H%M%S).log"

{
  echo "============================================================"
  echo "Visual-agent Qwen2.5-VL SFT launcher"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Repository: $REPO_ROOT"
  echo "BASE: $BASE"
  echo "ROOT_DIR: $ROOT_DIR"
  echo "ENV_DIR: $ENV_DIR"
  echo "MODEL_DIR: $MODEL_DIR"
  echo "CUDA_HOME: $CUDA_HOME"
  echo "Conda environment: ${CONDA_ENV_PATH:-current shell environment}"
  echo "Model: $MODEL_NAME_OR_PATH"
  echo "HF_HOME: $HF_HOME"
  echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"
  echo "TORCH_HOME: $TORCH_HOME"
  echo "Visual-tool output override: ${VISUAL_TOOL_OUTPUT_DIR:-<config default>}"
  echo "Dataset: $DATASET_NAME"
  echo "Dataset JSONL: $DATASET_JSONL"
  echo "Dataset samples: $DATASET_SAMPLES"
  echo "Python: $(command -v python || true)"
  echo "Torchrun: $(command -v torchrun || true)"
  echo "LLaMA-Factory: ${LLAMAFACTORY_CLI:-$(command -v llamafactory-cli || true)}"
  echo "CC: ${CC:-unset}"
  echo "CXX: ${CXX:-unset}"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "NNODES: $NNODES"
  echo "NPROC_PER_NODE: $NPROC_PER_NODE"
  echo "TARGET_GLOBAL_BATCH_SIZE: $TARGET_GLOBAL_BATCH_SIZE"
  echo "MASTER_PORT: $MASTER_PORT"
  echo "Log: $LOG_FILE"
  echo "============================================================"

  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
  fi
  if [[ "$PRINT_HARDWARE_INFO" == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
} 2>&1 | tee -a "$LOG_FILE"

if [[ -n "$VISUAL_TOOL_OUTPUT_DIR" ]]; then
  export OUTPUT_DIR="$VISUAL_TOOL_OUTPUT_DIR"
else
  # ModelArts may define OUTPUT_DIR broadly; keep LLaMA-Factory on the repo config default unless explicitly overridden.
  unset OUTPUT_DIR
fi

bash "$REPO_ROOT/scripts/train_visual_tool_sft_full.sh" 2>&1 | tee -a "$LOG_FILE"
