#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts: one dedicated 8-GPU node, multiple checkpoints evaluated sequentially.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export BASE="${BASE:-$(dirname "$REPO_ROOT")}"
CONFIG_PYTHON="${CONFIG_PYTHON:-python}"
CONFIG_ONLY="${CHECKPOINT_EVAL_CONFIG_ONLY:-0}"
[[ "$CONFIG_ONLY" == 0 || "$CONFIG_ONLY" == 1 ]] || { echo 'Invalid CHECKPOINT_EVAL_CONFIG_ONLY' >&2; exit 2; }
[[ "${NNODES:-1}" == 1 ]] || { echo 'Submit this evaluation as a single-node job.' >&2; exit 2; }

if [[ "${CHECKPOINT_EVAL_INLINE:-0}" != 1 ]]; then
RUN16="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_2node"
RUN64="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_b224_8node_scratch"
LABELS=(16gpu_step120 16gpu_best80 64gpu_step50 64gpu_best40)
STEPS=(120 80 50 40)
MODELS=("$RUN16/global_step_120/actor/huggingface" "$RUN16/best_huggingface"
        "$RUN64/global_step_50/actor/huggingface" "$RUN64/best_huggingface")
if [[ -n "${CHECKPOINT_EVAL_CONFIG:-}" ]]; then
  source "$CHECKPOINT_EVAL_CONFIG"
fi
fi
(( ${#LABELS[@]} > 0 && ${#LABELS[@]} == ${#STEPS[@]} && ${#LABELS[@]} == ${#MODELS[@]} )) || {
  echo 'LABELS, STEPS and MODELS must be nonempty and have equal lengths' >&2; exit 2;
}
DEFAULT_EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K}"
if declare -p DATASETS >/dev/null 2>&1; then
  (( ${#DATASETS[@]} == ${#LABELS[@]} )) || {
    echo 'DATASETS must have the same length as LABELS when provided' >&2; exit 2;
  }
else
  DATASETS=()
  for _ in "${LABELS[@]}"; do
    DATASETS+=("$DEFAULT_EVAL_DATASETS")
  done
fi
declare -A seen_labels=()
for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  [[ "$label" =~ ^[A-Za-z0-9_-]+$ && "${STEPS[$index]}" =~ ^[0-9]+$ && -n "${DATASETS[$index]}" && -z "${seen_labels[$label]:-}" ]] || {
    echo 'Invalid or duplicate checkpoint label/step' >&2; exit 2;
  }
  seen_labels[$label]=1
done

# Validate all inputs before loading the first model. Never copy large weights.
"$CONFIG_PYTHON" - "${#MODELS[@]}" "${STEPS[@]}" "${MODELS[@]}" <<'PY'
import json
from pathlib import Path
import sys
count = int(sys.argv[1])
for expected, value in zip(sys.argv[2:2+count], sys.argv[2+count:]):
    path = Path(value)
    if path.name == 'best_huggingface':
        metadata = json.loads((path.parent / 'best_checkpoint.json').read_text())
        if metadata['step'] != int(expected):
            raise SystemExit(f'Best checkpoint changed: {path}, expected {expected}, got {metadata["step"]}')
    json.loads((path / 'config.json').read_text())
    weights = set(json.loads((path / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if not weights or any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in weights):
        raise SystemExit(f'Missing/empty weight shards: {path}')
    size = sum((path / name).stat().st_size for name in weights)
    print(f'Checkpoint ready: {path}, shards={len(weights)}, GiB={size / 2**30:.2f}')
PY

export EVAL_MODELS=dino_latest
export EVAL_DATASETS="$DEFAULT_EVAL_DATASETS"
export VISUAL_TOOL_BACKEND=groundingdino SAM3_REPLICAS=0 GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0}" MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=grounding_detect,crop_zoom
export VISUAL_AGENT_MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-6}" VISUAL_AGENT_MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-6144}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-28}" VLMEVAL_FAILED_SAMPLE_RETRIES="${VLMEVAL_FAILED_SAMPLE_RETRIES:-5}"
export MODEL_SERVER_BACKEND="${EVAL_MODEL_SERVER_BACKEND:-vllm}"
export EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-0}"
GROUP_ID="${CHECKPOINT_EVAL_RUN_ID:-mixed_four_checkpoints_$(date +%Y%m%d_%H%M%S)}"
[[ "$GROUP_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid CHECKPOINT_EVAL_RUN_ID' >&2; exit 2; }
GROUP_ROOT="${CHECKPOINT_EVAL_OUTPUT_ROOT:-$REPO_ROOT/outputs/vlmeval/mixed_checkpoints}/$GROUP_ID"
export LOG_DIR="$GROUP_ROOT/logs"

for index in "${!LABELS[@]}"; do
  printf '%s: %s; datasets=%s\n' "${LABELS[$index]}" "${MODELS[$index]}" "${DATASETS[$index]}"
done
printf 'Protocol: per-checkpoint datasets; turns=%s tokens=%s backend=%s\n' "$VISUAL_AGENT_MAX_TURNS" "$VISUAL_AGENT_MAX_TOKENS" "$MODEL_SERVER_BACKEND"
echo "Results: $GROUP_ROOT"
if [[ "$CONFIG_ONLY" == 1 ]]; then
  echo 'Configuration only: no evaluation or model server started.'
  exit 0
fi

mkdir -p "$(dirname "$GROUP_ROOT")"
mkdir "$GROUP_ROOT"  # Refuse to reuse old prediction caches accidentally.
mkdir "$LOG_DIR"
if [[ -n "${CHECKPOINT_EVAL_CONFIG:-}" ]]; then
  cp -- "$CHECKPOINT_EVAL_CONFIG" "$GROUP_ROOT/eval_config.sh"
fi
printf 'checkpoint\texit_code\tseconds\tresult_dir\n' > "$GROUP_ROOT/status.tsv"
trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  export RL_DINO_LATEST_STEP="${STEPS[$index]}"
  export RL_DINO_LATEST_MODEL_PATH="${MODELS[$index]}"
  export EVAL_DATASETS="${DATASETS[$index]}"
  export RUN_ID="${GROUP_ID}_${label}"
  export VLMEVAL_EVAL_ID="T_${RUN_ID}"
  export WORK_ROOT="$GROUP_ROOT/$label"
  echo "Starting $label: $EVAL_DATASETS"
  started=$SECONDS
  status=0
  bash "${CHECKPOINT_EVAL_ENTRY:-$SCRIPT_DIR/run_visual_agent_eval_qwen3.sh}" || status=$?
  printf '%s\t%s\t%s\t%s\n' "$label" "$status" "$((SECONDS - started))" "$WORK_ROOT" >> "$GROUP_ROOT/status.tsv"
  if (( status != 0 )); then
    echo "Evaluation failed: $label (exit=$status); continuing with remaining checkpoints" >&2
    failed=1
  fi
done
echo "All requested attempts finished. Check $GROUP_ROOT/status.tsv and per-benchmark result CSVs."
exit "$failed"
