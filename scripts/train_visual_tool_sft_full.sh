#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/train_visual_tool_sft_full.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-cold_start/examples/deepspeed/ds_z3_config.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MAX_STEPS="${MAX_STEPS:-}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  echo "error: $*" >&2
  exit 2
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got: $value"
}

require_uint NNODES "$NNODES"
require_uint NPROC_PER_NODE "$NPROC_PER_NODE"
require_uint NODE_RANK "$NODE_RANK"
require_uint MASTER_PORT "$MASTER_PORT"
require_uint TARGET_GLOBAL_BATCH_SIZE "$TARGET_GLOBAL_BATCH_SIZE"
require_uint DRY_RUN "$DRY_RUN"
((NNODES > 0)) || die "NNODES must be greater than zero"
((NPROC_PER_NODE > 0)) || die "NPROC_PER_NODE must be greater than zero"
((MASTER_PORT > 0 && MASTER_PORT <= 65535)) || die "MASTER_PORT must be between 1 and 65535"
((TARGET_GLOBAL_BATCH_SIZE > 0)) || die "TARGET_GLOBAL_BATCH_SIZE must be greater than zero"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"

if [[ -z "$MASTER_ADDR" ]]; then
  if ((NNODES == 1)); then
    MASTER_ADDR="127.0.0.1"
  else
    die "MASTER_ADDR is required when NNODES is greater than 1"
  fi
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if ((TARGET_GLOBAL_BATCH_SIZE % WORLD_SIZE != 0)); then
  die "world size $WORLD_SIZE does not divide TARGET_GLOBAL_BATCH_SIZE=$TARGET_GLOBAL_BATCH_SIZE"
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / WORLD_SIZE))

[[ -f "$CONFIG_PATH" ]] || die "training config does not exist: $CONFIG_PATH"
[[ -f "$DEEPSPEED_CONFIG" ]] || die "DeepSpeed config does not exist: $DEEPSPEED_CONFIG"
[[ -f scripts/prepare_visual_tool_sft_config.py ]] || die "runtime config renderer is missing"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable was not found: $PYTHON_BIN"

if [[ -n "$MODEL_NAME_OR_PATH" && ( "$MODEL_NAME_OR_PATH" == /* || "$MODEL_NAME_OR_PATH" == ./* || "$MODEL_NAME_OR_PATH" == ../* ) ]]; then
  [[ -e "$MODEL_NAME_OR_PATH" ]] || die "model path does not exist: $MODEL_NAME_OR_PATH"
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && ( "$RESUME_FROM_CHECKPOINT" == /* || "$RESUME_FROM_CHECKPOINT" == ./* || "$RESUME_FROM_CHECKPOINT" == ../* ) ]]; then
  [[ -e "$RESUME_FROM_CHECKPOINT" ]] || die "checkpoint path does not exist: $RESUME_FROM_CHECKPOINT"
fi
if [[ -n "$MAX_STEPS" ]]; then
  require_uint MAX_STEPS "$MAX_STEPS"
  ((MAX_STEPS > 0)) || die "MAX_STEPS must be greater than zero"
fi

if [[ -z "$LLAMAFACTORY_CLI" ]] && command -v llamafactory-cli >/dev/null 2>&1; then
  LLAMAFACTORY_CLI="$(command -v llamafactory-cli)"
fi
if [[ -z "$LLAMAFACTORY_CLI" && -x .venv-llamafactory/bin/llamafactory-cli ]]; then
  LLAMAFACTORY_CLI="$REPO_ROOT/.venv-llamafactory/bin/llamafactory-cli"
fi
[[ -n "$LLAMAFACTORY_CLI" ]] || die "llamafactory-cli was not found; set LLAMAFACTORY_CLI"
if [[ "$LLAMAFACTORY_CLI" == */* ]]; then
  [[ -x "$LLAMAFACTORY_CLI" ]] || die "LLaMA-Factory CLI is not executable: $LLAMAFACTORY_CLI"
else
  command -v "$LLAMAFACTORY_CLI" >/dev/null 2>&1 || die "LLaMA-Factory CLI was not found: $LLAMAFACTORY_CLI"
fi

RUNTIME_CONFIG="$(mktemp "${TMPDIR:-/tmp}/visual_tool_sft_full.XXXXXX")"
cleanup() {
  rm -f "$RUNTIME_CONFIG"
}
trap cleanup EXIT

render_args=(
  --source "$CONFIG_PATH"
  --destination "$RUNTIME_CONFIG"
  --set-int gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --set deepspeed "$DEEPSPEED_CONFIG"
)
[[ -z "$MODEL_NAME_OR_PATH" ]] || render_args+=(--set model_name_or_path "$MODEL_NAME_OR_PATH")
[[ -z "$OUTPUT_DIR" ]] || render_args+=(--set output_dir "$OUTPUT_DIR")
[[ -z "$RESUME_FROM_CHECKPOINT" ]] || render_args+=(--set resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
[[ -z "$MAX_STEPS" ]] || render_args+=(--set-int max_steps "$MAX_STEPS")
"$PYTHON_BIN" scripts/prepare_visual_tool_sft_config.py "${render_args[@]}"

export FORCE_TORCHRUN=1
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT

echo "Distributed visual-tool SFT topology:"
echo "  nnodes=$NNODES"
echo "  nproc_per_node=$NPROC_PER_NODE"
echo "  node_rank=$NODE_RANK"
echo "  world_size=$WORLD_SIZE"
echo "  target_global_batch_size=$TARGET_GLOBAL_BATCH_SIZE"
echo "  gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "  master=$MASTER_ADDR:$MASTER_PORT"

launch_command=("$LLAMAFACTORY_CLI" train "$RUNTIME_CONFIG")
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Launch command:'
  printf ' %q' "${launch_command[@]}"
  printf '\n\nRendered training config:\n'
  cat "$RUNTIME_CONFIG"
  exit 0
fi

"${launch_command[@]}"
