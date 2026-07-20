#!/usr/bin/env bash
set -Eeuo pipefail

# One-step ModelArts GRPO smoke test for Qwen3-VL plus online visual tools.
# Default placement on an 8-GPU node:
#   GPUs 0-3: VERL actor/ref/vLLM workers
#   GPU 7:    SAM3 + GroundingDINO HTTP service

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
RL_CUDA_HOME="${RL_CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
TOOL_CUDA_HOME="${VISUAL_TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"

PLATFORM_DATASET_ROOT="${PLATFORM_DATASET_ROOT:-/opt/huawei/dataset}"
PLATFORM_ALGORITHM_ROOT="${PLATFORM_ALGORITHM_ROOT:-/opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl}"
PLATFORM_MODEL_ROOT="${PLATFORM_MODEL_ROOT:-/opt/huawei/quoteModel/xiaoyi_tmpstorage}"

RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TOOL_GPU="${TOOL_GPU:-7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_TURNS="${MAX_TURNS:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-$REPO_ROOT/data/visual_tool_rl_smoke}"
TRAIN_FILE="${TRAIN_FILE:-$SMOKE_DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$SMOKE_DATA_DIR/val.parquet}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_count_config.yaml}"

VISUAL_TOOL_HOST="${VISUAL_TOOL_HOST:-127.0.0.1}"
VISUAL_TOOL_PORT="${VISUAL_TOOL_PORT:-9000}"
export VISUAL_TOOL_API_BASE="${VISUAL_TOOL_API_BASE:-http://$VISUAL_TOOL_HOST:$VISUAL_TOOL_PORT}"
START_VISUAL_TOOL_SERVER="${START_VISUAL_TOOL_SERVER:-1}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$BASE/visual-tools/sam3/sam3.pt}"
GROUNDING_DINO_MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-$BASE/visual-tools/grounding-dino-base-transformers}"

RUN_ID="${RUN_ID:-visual_rl_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_tool_rl_smoke/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$BASE/logs/visual-tool-rl}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"
TOOL_LOG_FILE="${TOOL_LOG_FILE:-$LOG_DIR/$RUN_ID-tools.log}"

RL_PYTHON="$RL_ENV_DIR/bin/python"
TOOL_PYTHON="$TOOL_ENV_DIR/bin/python"
TOOL_PID=""

die() {
  echo "error: $*" >&2
  exit 2
}

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  if [[ -e "$link_path" || -L "$link_path" ]]; then
    return
  fi
  [[ -e "$target" ]] || die "platform symlink target does not exist: $target"
  mkdir -p "$(dirname "$link_path")"
  ln -s "$target" "$link_path"
  echo "Created platform symlink: $link_path -> $target"
}

cleanup() {
  local status=$?
  if [[ -n "$TOOL_PID" ]] && kill -0 "$TOOL_PID" 2>/dev/null; then
    echo "Stopping visual-tool server PID $TOOL_PID"
    kill "$TOOL_PID" 2>/dev/null || true
    wait "$TOOL_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

ensure_symlink "$PLATFORM_DATASET_ROOT" /opt/huawei/explorer-env/dataset
ensure_symlink "$PLATFORM_DATASET_ROOT" /home/ma-user/work/dataset
ensure_symlink "$PLATFORM_ALGORITHM_ROOT" /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink "$PLATFORM_MODEL_ROOT" /home/ma-user/work/model/xiaoyi_tmpstorage

[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -d "$RL_ROOT/verl" ]] || die "VERL source not found: $RL_ROOT"
[[ -x "$RL_PYTHON" ]] || die "RL Python not executable: $RL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die "visual-tools Python not executable: $TOOL_PYTHON"
[[ -f "$MODEL_PATH/config.json" ]] || die "HF checkpoint config not found: $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || die "HF checkpoint weights not found: $MODEL_PATH"
[[ -f "$TOOL_CONFIG_PATH" ]] || die "visual tool config not found: $TOOL_CONFIG_PATH"
[[ -f "$SAM3_MODEL_PATH" ]] || die "SAM3 checkpoint not found: $SAM3_MODEL_PATH"
[[ -f "$GROUNDING_DINO_MODEL_PATH/config.json" ]] || die "GroundingDINO model not found: $GROUNDING_DINO_MODEL_PATH"
[[ -d "$RL_CUDA_HOME" ]] || die "RL CUDA toolkit not found: $RL_CUDA_HOME"
[[ -d "$TOOL_CUDA_HOME" ]] || die "visual-tool CUDA toolkit not found: $TOOL_CUDA_HOME"

IFS=',' read -r -a RL_GPU_ARRAY <<< "$RL_CUDA_VISIBLE_DEVICES"
N_GPUS_PER_NODE="${#RL_GPU_ARRAY[@]}"
[[ "$N_GPUS_PER_NODE" -gt 0 ]] || die "RL_CUDA_VISIBLE_DEVICES is empty"
[[ "$TRAIN_BATCH_SIZE" -ge "$N_GPUS_PER_NODE" ]] || {
  die "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be at least N_GPUS_PER_NODE ($N_GPUS_PER_NODE)"
}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$SMOKE_DATA_DIR" "$BASE/cache/huggingface" "$BASE/cache/torch" "$BASE/cache/xdg"
touch "$LOG_FILE" "$TOOL_LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_HOME="$RL_CUDA_HOME"
export PATH="$RL_ENV_DIR/bin:$RL_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$RL_ENV_DIR/lib:$RL_CUDA_HOME/lib:$RL_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [[ -x "$RL_CUDA_HOME/bin/gcc" && -x "$RL_CUDA_HOME/bin/g++" ]]; then
  export CC="${CC:-$RL_CUDA_HOME/bin/gcc}"
  export CXX="${CXX:-$RL_CUDA_HOME/bin/g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
fi
export PYTHONPATH="$RL_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$BASE/cache/torch}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

echo "============================================================"
echo "Visual-agent Qwen3-VL RL smoke test"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repository: $REPO_ROOT"
echo "RL environment: $RL_ENV_DIR"
echo "Tool environment: $TOOL_ENV_DIR"
echo "Initial checkpoint: $MODEL_PATH"
echo "RL CUDA toolkit: $RL_CUDA_HOME"
echo "Tool CUDA toolkit: $TOOL_CUDA_HOME"
echo "RL GPUs: $RL_CUDA_VISIBLE_DEVICES ($N_GPUS_PER_NODE workers)"
echo "Tool GPU: $TOOL_GPU"
echo "Tool API: $VISUAL_TOOL_API_BASE"
echo "Training steps: $TOTAL_TRAINING_STEPS"
echo "Train batch / rollout n: $TRAIN_BATCH_SIZE / $ROLLOUT_N"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

"$RL_PYTHON" - <<'PYCHECK'
import ray
import torch
import transformers
import verl
import vllm
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__, "ray:", ray.__version__, "verl:", verl.__version__)
PYCHECK

PREPARE_ARGS=(
  --input "$REPO_ROOT/data/visual_tool_sft_v0.smoke.jsonl"
  --output-dir "$SMOKE_DATA_DIR"
  --train-samples "$TRAIN_BATCH_SIZE"
  --val-samples 2
)
"$RL_PYTHON" "$REPO_ROOT/scripts/prepare_visual_tool_rl_smoke.py" "${PREPARE_ARGS[@]}"

[[ -f "$TRAIN_FILE" && -f "$VAL_FILE" ]] || die "smoke parquet generation failed"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  START_VISUAL_TOOL_SERVER=0
fi

if [[ "$START_VISUAL_TOOL_SERVER" == "1" ]]; then
  echo "Starting SAM3 + GroundingDINO on physical GPU $TOOL_GPU"
  TOOL_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=$TOOL_GPU"
    "CUDA_HOME=$TOOL_CUDA_HOME"
    "PATH=$TOOL_ENV_DIR/bin:$TOOL_CUDA_HOME/bin:$PATH"
    "LD_LIBRARY_PATH=$TOOL_ENV_DIR/lib:$TOOL_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    "SAM3_MODEL_PATH=$SAM3_MODEL_PATH"
    "SAM3_DEVICE=cuda:0"
    "GROUNDING_DINO_MODEL_PATH=$GROUNDING_DINO_MODEL_PATH"
    "GROUNDING_DINO_DEVICE=cuda:0"
    "VISUAL_TOOL_BACKEND=all"
    "$TOOL_PYTHON"
    "$REPO_ROOT/scripts/visual_tool_server.py"
    --backend all
    --host "$VISUAL_TOOL_HOST"
    --port "$VISUAL_TOOL_PORT"
  )
  "${TOOL_COMMAND[@]}" >>"$TOOL_LOG_FILE" 2>&1 &
  TOOL_PID=$!

  "$RL_PYTHON" - "$VISUAL_TOOL_API_BASE" <<'PYHEALTH'
import json
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/")
last_error = None
for _ in range(180):
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            payload = json.load(response)
        if payload.get("sam3_loaded") and payload.get("grounding_dino_loaded"):
            print("Visual-tool service ready:", payload)
            break
    except Exception as exc:
        last_error = exc
    time.sleep(1)
else:
    raise SystemExit(f"visual-tool service did not become ready: {last_error}")
PYHEALTH
elif [[ "$START_VISUAL_TOOL_SERVER" != "0" ]]; then
  die "START_VISUAL_TOOL_SERVER must be 0 or 1"
fi

if [[ "$START_VISUAL_TOOL_SERVER" == "0" && "${DRY_RUN:-0}" != "1" ]]; then
  "$RL_PYTHON" - "$VISUAL_TOOL_API_BASE" <<'PYHEALTH'
import json
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/health", timeout=10) as response:
    payload = json.load(response)
assert payload.get("sam3_loaded") and payload.get("grounding_dino_loaded"), payload
print("Using existing visual-tool service:", payload)
PYHEALTH
fi

export CUDA_VISIBLE_DEVICES="$RL_CUDA_VISIBLE_DEVICES"
cd "$RL_ROOT"

HYDRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  HYDRA_ARGS=(--cfg job)
fi

TRAIN_ARGS=(
  "+debug=False"
  "+vs_debug=False"
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=[$TRAIN_FILE]"
  "data.val_files=[$VAL_FILE]"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.val_batch_size=2"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.return_raw_chat=True"
  "data.shuffle=False"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=1"
  "data.truncation=error"
  "data.image_key=images"
  "data.trust_remote_code=True"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.temperature=1"
  "actor_rollout_ref.rollout.max_num_batched_tokens=8192"
  "actor_rollout_ref.rollout.agent.activate_agent=False"
  "actor_rollout_ref.rollout.multi_turn.enable=True"
  "actor_rollout_ref.rollout.multi_turn.max_turns=$MAX_TURNS"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH"
  "+actor_rollout_ref.rollout.multi_turn.format=hermes"
  "actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True"
  "actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "reward_model.reward_manager=naive_async"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.nnodes=1"
  "trainer.val_before_train=False"
  "trainer.save_freq=-1"
  "trainer.test_freq=-1"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.resume_mode=disable"
  "trainer.project_name=visual-agent-rl-smoke"
  "trainer.experiment_name=$RUN_ID"
  "trainer.default_local_dir=$OUTPUT_DIR"
)

set -x
"$RL_PYTHON" -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "${HYDRA_ARGS[@]}"

status=$?
set +x
echo "RL smoke test finished with exit code $status at $(date '+%Y-%m-%d %H:%M:%S')"
exit "$status"
