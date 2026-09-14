#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated one-update SLIME PoC for Qwen3-VL and image-returning visual tools.
# This launcher never stops or kills Ray/SGLang processes owned by another job.

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-}"
SLIME_PIN="${SLIME_PIN:-0104a9e922ecd3cea768f7c2520e7d1a122afdc9}"
SLIME_PYTHON="${SLIME_PYTHON:-python3}"
SLIME_RAY_BIN="${SLIME_RAY_BIN:-ray}"

MODEL_PATH="${MODEL_PATH:-}"
SOURCE_DATA="${SOURCE_DATA:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation/smoke_train.parquet}"
POC_ROOT="${SLIME_POC_ROOT:-$REPO_ROOT/saves/slime_visual_agent_poc}"
RUN_ID="${RUN_ID:-qwen3vl_fixed_update_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$POC_ROOT/$RUN_ID}"
PROMPT_DATA="${PROMPT_DATA:-$OUTPUT_DIR/data/train.jsonl}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$ACTOR_GPUS_PER_NODE}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
COLOCATE="${COLOCATE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-$ACTOR_GPUS_PER_NODE}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.55}"

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
[[ -d "$SLIME_ROOT/.git" && -f "$SLIME_ROOT/train.py" ]] || die "invalid SLIME_ROOT: $SLIME_ROOT"
[[ -n "$MODEL_PATH" ]] || die "set MODEL_PATH to the Qwen3-VL HF/SFT checkpoint"
[[ -f "$MODEL_PATH/config.json" ]] || die "model config not found: $MODEL_PATH/config.json"
[[ -f "$SOURCE_DATA" ]] || die "source dataset not found: $SOURCE_DATA"
[[ "$ACTOR_NUM_NODES" =~ ^[1-9][0-9]*$ ]] || die "ACTOR_NUM_NODES must be positive"
[[ "$ACTOR_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "ACTOR_GPUS_PER_NODE must be positive"
[[ "$ROLLOUT_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUT_NUM_GPUS must be positive"
[[ "$ROLLOUT_GPUS_PER_ENGINE" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUT_GPUS_PER_ENGINE must be positive"
[[ "$COLOCATE" =~ ^[01]$ ]] || die "COLOCATE must be 0 or 1"
[[ "$START_LOCAL_RAY" =~ ^[01]$ ]] || die "START_LOCAL_RAY must be 0 or 1"
[[ "$CONFIRM_DEDICATED_RESOURCES" =~ ^[01]$ ]] || die "CONFIRM_DEDICATED_RESOURCES must be 0 or 1"
if [[ "$DRY_RUN" != "1" && "$CONFIRM_DEDICATED_RESOURCES" != "1" ]]; then
  die "set CONFIRM_DEDICATED_RESOURCES=1 after confirming the Ray GPUs and visual-tool endpoint are not serving an existing run"
fi
if [[ "$START_LOCAL_RAY" == "1" && ( "$ACTOR_NUM_NODES" != "1" || "$COLOCATE" != "1" ) ]]; then
  die "START_LOCAL_RAY=1 is only for the isolated one-node colocated PoC; use a dedicated external Ray cluster otherwise"
fi
(( ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE == 0 )) || die "rollout GPU count must divide evenly into engines"
ACTOR_WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))
(( GLOBAL_BATCH_SIZE % ACTOR_WORLD_SIZE == 0 )) || die "GLOBAL_BATCH_SIZE must be divisible by actor world size"

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
with urllib.request.urlopen(sys.argv[1] + "/health", timeout=15) as response:
    payload = json.load(response)
print("visual-tool ready:", sys.argv[1], payload)
PYHEALTH
  done
fi

mkdir -p "$OUTPUT_DIR/data" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/ray"
prepare_args=(
  --input "$SOURCE_DATA"
  --output "$PROMPT_DATA"
  --system-prompt "$REPO_ROOT/prompts/visual_agent_rl_system.txt"
  --limit "$ROLLOUT_BATCH_SIZE"
  --force
)
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
  --multimodal-keys '{"image":"images"}'
  --apply-chat-template
  --custom-generate-function-path slime_visual_agent.rollout.generate
  --custom-rm-path slime_visual_agent.reward.compute_reward
  --reward-key score
  --custom-config-path "$REPO_ROOT/slime_visual_agent/config/rollout.yaml"
  --advantage-estimator grpo
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --num-rollout "$NUM_ROLLOUT"
  --rollout-max-response-len "$MAX_RESPONSE_LENGTH"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --actor-num-nodes "$ACTOR_NUM_NODES"
  --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE"
  --rollout-num-gpus "$ROLLOUT_NUM_GPUS"
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
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
  --kl-loss-coef 0.0
  --kl-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --save "$OUTPUT_DIR/checkpoints"
  --save-interval 1
  --dump-details "$OUTPUT_DIR/debug"
  --check-weight-update-equal
)
if [[ "$COLOCATE" == "1" ]]; then
  train_args+=(--colocate)
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
    if key.startswith("VISUAL_TOOL_") or key.startswith("VISUAL_AGENT_")
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
