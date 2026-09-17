#!/usr/bin/env bash
set -Eeuo pipefail

# SLIME FSDP launcher for Qwen3-VL and image-returning visual tools. The
# defaults retain the isolated one-update PoC; full experiments override them
# from a thin experiment entrypoint.

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-}"
SLIME_PIN="${SLIME_PIN:-0104a9e922ecd3cea768f7c2520e7d1a122afdc9}"
SLIME_PYTHON="${SLIME_PYTHON:-python3}"
SLIME_RAY_BIN="${SLIME_RAY_BIN:-ray}"

MODEL_PATH="${MODEL_PATH:-}"
SOURCE_DATA="${SOURCE_DATA:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation/smoke_train.parquet}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system.txt}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-$REPO_ROOT/slime_visual_agent/config/rollout.yaml}"
POC_ROOT="${SLIME_POC_ROOT:-$REPO_ROOT/saves/slime_visual_agent_poc}"
RUN_ID="${RUN_ID:-qwen3vl_fixed_update_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$POC_ROOT/$RUN_ID}"
PROMPT_DATA="${PROMPT_DATA:-$OUTPUT_DIR/data/train.jsonl}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$ACTOR_GPUS_PER_NODE}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-$ACTOR_GPUS_PER_NODE}"
COLOCATE="${COLOCATE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-$ACTOR_GPUS_PER_NODE}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
ROLLOUT_SAMPLE_COUNT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
PREPARE_LIMIT="${PREPARE_LIMIT:-$ROLLOUT_BATCH_SIZE}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.55}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
KL_LOSS_ENABLED="${KL_LOSS_ENABLED:-0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
REF_MODEL_PATH="${REF_MODEL_PATH:-$MODEL_PATH}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-0}"
BALANCE_DATA="${BALANCE_DATA:-0}"
CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-1}"
# An explicitly empty value disables the large per-rollout debug dumps.
DUMP_DETAILS="${DUMP_DETAILS-$OUTPUT_DIR/debug}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_UPLOAD_MODE="${WANDB_UPLOAD_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-visual-agent-slime}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"

START_LOCAL_RAY="${START_LOCAL_RAY:-0}"
RAY_MASTER_ADDR="${RAY_MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-17379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-18265}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://$RAY_MASTER_ADDR:$RAY_DASHBOARD_PORT}"
DRY_RUN="${DRY_RUN:-0}"
CONFIRM_DEDICATED_RESOURCES="${CONFIRM_DEDICATED_RESOURCES:-0}"

die() {
  echo "error: $*" >&2
  exit 2
}

[[ -n "$SLIME_ROOT" ]] || die "set SLIME_ROOT to a checkout of THUDM/slime"
[[ -f "$SLIME_ROOT/train.py" ]] || die "invalid SLIME_ROOT: $SLIME_ROOT"
git -C "$SLIME_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "invalid SLIME_ROOT: $SLIME_ROOT"
[[ -n "$MODEL_PATH" ]] || die "set MODEL_PATH to the Qwen3-VL HF/SFT checkpoint"
[[ -f "$MODEL_PATH/config.json" ]] || die "model config not found: $MODEL_PATH/config.json"
[[ -f "$SOURCE_DATA" ]] || die "source dataset not found: $SOURCE_DATA"
[[ -f "$SYSTEM_PROMPT_FILE" ]] || die "system prompt not found: $SYSTEM_PROMPT_FILE"
[[ -f "$CUSTOM_CONFIG_PATH" ]] || die "custom rollout config not found: $CUSTOM_CONFIG_PATH"
[[ "$ACTOR_NUM_NODES" =~ ^[1-9][0-9]*$ ]] || die "ACTOR_NUM_NODES must be positive"
[[ "$ACTOR_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "ACTOR_GPUS_PER_NODE must be positive"
[[ "$ROLLOUT_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUT_NUM_GPUS must be positive"
[[ "$ROLLOUT_GPUS_PER_ENGINE" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUT_GPUS_PER_ENGINE must be positive"
[[ "$NUM_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NUM_GPUS_PER_NODE must be positive"
[[ "$NUM_STEPS_PER_ROLLOUT" =~ ^[1-9][0-9]*$ ]] || die "NUM_STEPS_PER_ROLLOUT must be positive"
[[ "$PREPARE_LIMIT" =~ ^[0-9]+$ ]] || die "PREPARE_LIMIT must be a non-negative integer"
[[ "$COLOCATE" =~ ^[01]$ ]] || die "COLOCATE must be 0 or 1"
[[ "$START_LOCAL_RAY" =~ ^[01]$ ]] || die "START_LOCAL_RAY must be 0 or 1"
[[ "$CONFIRM_DEDICATED_RESOURCES" =~ ^[01]$ ]] || die "CONFIRM_DEDICATED_RESOURCES must be 0 or 1"
[[ "$WANDB_ENABLE" =~ ^[01]$ ]] || die "WANDB_ENABLE must be 0 or 1"
[[ "$WANDB_UPLOAD_MODE" == "online" || "$WANDB_UPLOAD_MODE" == "offline" ]] || \
  die "WANDB_UPLOAD_MODE must be online or offline"
for boolean_name in KL_LOSS_ENABLED ROLLOUT_SHUFFLE BALANCE_DATA CALCULATE_PER_TOKEN_LOSS CHECK_WEIGHT_UPDATE_EQUAL; do
  [[ "${!boolean_name}" =~ ^[01]$ ]] || die "$boolean_name must be 0 or 1"
done
if [[ "$DRY_RUN" != "1" && "$CONFIRM_DEDICATED_RESOURCES" != "1" ]]; then
  die "set CONFIRM_DEDICATED_RESOURCES=1 after confirming the Ray GPUs and visual-tool endpoint are not serving an existing run"
fi
if [[ "$START_LOCAL_RAY" == "1" && ( "$ACTOR_NUM_NODES" != "1" || "$COLOCATE" != "1" ) ]]; then
  die "START_LOCAL_RAY=1 is only for the isolated one-node colocated PoC; use a dedicated external Ray cluster otherwise"
fi
(( ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE == 0 )) || die "rollout GPU count must divide evenly into engines"
ACTOR_WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))
(( ROLLOUT_SAMPLE_COUNT % NUM_STEPS_PER_ROLLOUT == 0 )) || \
  die "rollout sample count must be divisible by NUM_STEPS_PER_ROLLOUT"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_SAMPLE_COUNT / NUM_STEPS_PER_ROLLOUT))}"
[[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "GLOBAL_BATCH_SIZE must be positive"
[[ "$GLOBAL_BATCH_SIZE" -eq "$((ROLLOUT_SAMPLE_COUNT / NUM_STEPS_PER_ROLLOUT))" ]] || \
  die "GLOBAL_BATCH_SIZE must equal rollout_batch * samples_per_prompt / steps_per_rollout"
(( GLOBAL_BATCH_SIZE % ACTOR_WORLD_SIZE == 0 )) || die "GLOBAL_BATCH_SIZE must be divisible by actor world size"
if [[ "$KL_LOSS_ENABLED" == "1" ]]; then
  [[ -d "$REF_MODEL_PATH" ]] || die "reference model path not found: $REF_MODEL_PATH"
fi

actual_slime_commit="$(git -C "$SLIME_ROOT" rev-parse HEAD)"
if [[ "$actual_slime_commit" != "$SLIME_PIN" && "${ALLOW_UNPINNED_SLIME:-0}" != "1" ]]; then
  die "SLIME commit is $actual_slime_commit; expected $SLIME_PIN (set ALLOW_UNPINNED_SLIME=1 only after compatibility testing)"
fi

model_type="$($SLIME_PYTHON - "$MODEL_PATH" <<'PYMODEL'
import json
import sys
from pathlib import Path
config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
print(config.get("model_type", ""))
PYMODEL
)"
[[ "$model_type" == "qwen3_vl" ]] || die "expected model_type=qwen3_vl, got '$model_type'"

IFS=',' read -r -a tool_endpoints <<< "${VISUAL_TOOL_API_BASES:-${VISUAL_TOOL_API_BASE:-}}"
(( ${#tool_endpoints[@]} > 0 )) || die "set VISUAL_TOOL_API_BASES to an existing visual-tool service"
if [[ "$DRY_RUN" != "1" ]]; then
  for endpoint in "${tool_endpoints[@]}"; do
    endpoint="${endpoint%/}"
    "$SLIME_PYTHON" - "$endpoint" <<'PYHEALTH'
import json
import sys
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1] + "/health", timeout=15) as response:
    payload = json.load(response)
print("visual-tool ready:", sys.argv[1], payload)
PYHEALTH
  done
fi

mkdir -p "$OUTPUT_DIR/data" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/ray"
prepare_args=(
  --input "$SOURCE_DATA"
  --output "$PROMPT_DATA"
  --system-prompt "$SYSTEM_PROMPT_FILE"
  --force
)
if (( PREPARE_LIMIT > 0 )); then
  prepare_args+=(--limit "$PREPARE_LIMIT")
fi
if [[ -n "${VISUAL_AGENT_IMAGE_MAX_PIXELS:-}" ]]; then
  prepare_args+=(--image-max-pixels "$VISUAL_AGENT_IMAGE_MAX_PIXELS")
fi
if [[ "${REQUIRE_IMAGES:-1}" == "1" ]]; then
  prepare_args+=(--require-images)
fi
"$SLIME_PYTHON" "$REPO_ROOT/scripts/prepare_visual_agent_slime_poc.py" "${prepare_args[@]}"

train_args=(
  --train-backend fsdp
  --hf-checkpoint "$MODEL_PATH"
  --prompt-data "$PROMPT_DATA"
  --input-key messages
  --label-key solution
  --metadata-key metadata
  --apply-chat-template
  --custom-generate-function-path slime_visual_agent.rollout.generate
  --custom-rm-path slime_visual_agent.reward.compute_reward
  --reward-key score
  --custom-config-path "$CUSTOM_CONFIG_PATH"
  --advantage-estimator grpo
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT"
  --num-rollout "$NUM_ROLLOUT"
  --rollout-max-prompt-len "$MAX_PROMPT_LENGTH"
  --rollout-max-response-len "$MAX_RESPONSE_LENGTH"
  --rollout-max-context-len "$MAX_CONTEXT_LENGTH"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --actor-num-nodes "$ACTOR_NUM_NODES"
  --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE"
  --rollout-num-gpus "$ROLLOUT_NUM_GPUS"
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION"
  --gradient-checkpointing
  --attn-implementation flash_attention_2
  --update-weight-buffer-size 536870912
  --optimizer adam
  --lr "$LEARNING_RATE"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --kl-loss-coef "$KL_LOSS_COEF"
  --kl-loss-type "$KL_LOSS_TYPE"
  --kl-coef 0.0
  --entropy-coef "$ENTROPY_COEF"
  --eps-clip 0.2
  --eps-clip-high 0.28
  --save "$OUTPUT_DIR/checkpoints"
  --save-interval "$SAVE_INTERVAL"
)
if [[ "$COLOCATE" == "1" ]]; then
  train_args+=(--colocate)
fi
if [[ "$KL_LOSS_ENABLED" == "1" ]]; then
  train_args+=(--use-kl-loss --ref-load "$REF_MODEL_PATH")
fi
if [[ "$ROLLOUT_SHUFFLE" == "1" ]]; then
  train_args+=(--rollout-shuffle)
fi
if [[ "$BALANCE_DATA" == "1" ]]; then
  train_args+=(--balance-data)
fi
if [[ "$CALCULATE_PER_TOKEN_LOSS" == "1" ]]; then
  train_args+=(--calculate-per-token-loss)
fi
if [[ "$CHECK_WEIGHT_UPDATE_EQUAL" == "1" ]]; then
  train_args+=(--check-weight-update-equal)
fi
if [[ -n "$DUMP_DETAILS" ]]; then
  train_args+=(--dump-details "$DUMP_DETAILS")
fi
if [[ "$WANDB_ENABLE" == "1" ]]; then
  train_args+=(
    --use-wandb
    --wandb-mode "$WANDB_UPLOAD_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_RUN_GROUP"
  )
fi
if [[ -n "$SGLANG_SERVER_CONCURRENCY" ]]; then
  [[ "$SGLANG_SERVER_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "SGLANG_SERVER_CONCURRENCY must be positive"
  train_args+=(--sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY")
fi
if [[ -n "$SGLANG_MAX_RUNNING_REQUESTS" ]]; then
  [[ "$SGLANG_MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "SGLANG_MAX_RUNNING_REQUESTS must be positive"
  train_args+=(--sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi
if [[ -n "$SGLANG_CHUNKED_PREFILL_SIZE" ]]; then
  [[ "$SGLANG_CHUNKED_PREFILL_SIZE" =~ ^[1-9][0-9]*$ ]] || die "SGLANG_CHUNKED_PREFILL_SIZE must be positive"
  train_args+=(--sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE")
fi

runtime_env="$($SLIME_PYTHON - "$REPO_ROOT" "$SLIME_ROOT" <<'PYENV'
import json
import os
import sys
repo_root, slime_root = sys.argv[1:]
existing = os.environ.get("PYTHONPATH", "")
pythonpath = os.pathsep.join(item for item in (repo_root, slime_root, existing) if item)
forward = {
    key: value
    for key, value in os.environ.items()
    if key.startswith(("VISUAL_TOOL_", "VISUAL_AGENT_", "LLM_AS_A_JUDGE_", "GROUNDING_QUERY_", "WANDB_"))
    or key in {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"}
}
forward["PYTHONPATH"] = pythonpath
forward["PYTHONUNBUFFERED"] = "1"
print(json.dumps({"env_vars": forward}))
PYENV
)"

echo "SLIME Visual-Agent PoC"
echo "  SLIME commit: $actual_slime_commit"
echo "  model: $MODEL_PATH"
echo "  output: $OUTPUT_DIR"
echo "  actor nodes x GPUs: $ACTOR_NUM_NODES x $ACTOR_GPUS_PER_NODE"
echo "  rollout GPUs / per engine: $ROLLOUT_NUM_GPUS / $ROLLOUT_GPUS_PER_ENGINE"
echo "  colocate: $COLOCATE"
echo "  rollout groups x samples: $ROLLOUT_BATCH_SIZE x $N_SAMPLES_PER_PROMPT"
echo "  optimizer steps / global sample batch: $NUM_STEPS_PER_ROLLOUT / $GLOBAL_BATCH_SIZE"
echo "  KL loss / coefficient / type: $KL_LOSS_ENABLED / $KL_LOSS_COEF / $KL_LOSS_TYPE"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'ray job submit --address=%q --runtime-env-json=<redacted> -- python3 slime_visual_agent/train.py' "$RAY_DASHBOARD_ADDRESS"
  printf ' %q' "${train_args[@]}"
  printf '\n'
  exit 0
fi

if [[ "$START_LOCAL_RAY" == "1" ]]; then
  "$SLIME_RAY_BIN" start \
    --head \
    --node-ip-address "$RAY_MASTER_ADDR" \
    --port "$RAY_PORT" \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "$RAY_DASHBOARD_PORT" \
    --num-gpus "$ACTOR_GPUS_PER_NODE" \
    --temp-dir "$OUTPUT_DIR/ray" \
    --disable-usage-stats
fi

cd "$SLIME_ROOT"
exec "$SLIME_RAY_BIN" job submit \
  --address "$RAY_DASHBOARD_ADDRESS" \
  --runtime-env-json "$runtime_env" \
  -- "$SLIME_PYTHON" "$REPO_ROOT/slime_visual_agent/train.py" "${train_args[@]}"
