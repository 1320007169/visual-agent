#!/usr/bin/env bash
set -Eeuo pipefail

# SLIME equivalent of the two-node GroundingDINO + actor-KL ZWZ experiment.
# Run this entrypoint on both 8-GPU ModelArts nodes. GPUs 0-6 are advertised to
# Ray for colocated FSDP/SGLang; GPU 7 hosts the local GroundingDINO pool.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
ZWZ_RL_DIR="${ZWZ_RL_DIR:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation}"

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ -z "${SLIME_ROOT:-}" ]]; then
  for candidate in "$BASE/slime" "$REPO_ROOT/slime" "$BASE/../slime"; do
    if [[ -d "$candidate/.git" && -f "$candidate/train.py" ]]; then
      SLIME_ROOT="$candidate"
      break
    fi
  done
fi
SLIME_ROOT="${SLIME_ROOT:-}"

export BASE REPO_ROOT RL_ROOT ZWZ_RL_DIR SLIME_ROOT
export SLIME_PIN="${SLIME_PIN:-0104a9e922ecd3cea768f7c2520e7d1a122afdc9}"
export RL_ENV_DIR="${SLIME_ENV_DIR:-${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export SLIME_PYTHON="${SLIME_PYTHON:-$RL_ENV_DIR/bin/python}"
export SLIME_RAY_BIN="${SLIME_RAY_BIN:-$RL_ENV_DIR/bin/ray}"
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

export NNODES="${NNODES:-2}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-604800}"
export RAY_INCLUDE_DASHBOARD=1
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

[[ -n "$SLIME_ROOT" && -f "$SLIME_ROOT/train.py" ]] || \
  die "set SLIME_ROOT to the pinned THUDM/slime checkout"
[[ -x "$SLIME_PYTHON" ]] || die "SLIME Python is not executable: $SLIME_PYTHON"
actual_slime_commit="$(git -C "$SLIME_ROOT" rev-parse HEAD 2>/dev/null)" || \
  die "SLIME_ROOT is not a Git checkout: $SLIME_ROOT"
if [[ "$actual_slime_commit" != "$SLIME_PIN" && "${ALLOW_UNPINNED_SLIME:-0}" != "1" ]]; then
  die "SLIME commit is $actual_slime_commit; expected $SLIME_PIN"
fi
env PYTHONPATH="$REPO_ROOT:$SLIME_ROOT:${PYTHONPATH:-}" "$SLIME_PYTHON" - <<'PYSLIMECHECK'
import inspect

import ray
import sglang
import slime
import transformers
from qwen_vl_utils import process_vision_info

if "image_patch_size" not in inspect.signature(process_vision_info).parameters:
    raise SystemExit(
        "SLIME requires a qwen-vl-utils build whose process_vision_info accepts image_patch_size"
    )
print(
    "SLIME environment ready:",
    f"ray={ray.__version__}",
    f"sglang={getattr(sglang, '__version__', 'unknown')}",
    f"transformers={transformers.__version__}",
    flush=True,
)
PYSLIMECHECK

export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export TRAIN_FILES="${TRAIN_FILES:-$ZWZ_RL_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ZWZ_RL_DIR/val.parquet}"
export SOURCE_DATA="${SOURCE_DATA:-$TRAIN_FILES}"
export PREPARE_SMOKE_DATA=0

# Match the VERL experiment: 112 prompts x 16 trajectories, split into four
# optimizer steps of 28 prompts x 16 trajectories (448 samples) each.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-14}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-28}"
[[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_BATCH_SIZE must be positive"
[[ "$PPO_MINI_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "PPO_MINI_BATCH_SIZE must be positive"
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || \
  die "TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE"
IFS=',' read -r -a rl_gpu_array <<< "$RL_CUDA_VISIBLE_DEVICES"
rl_gpus_per_node="${#rl_gpu_array[@]}"
export ACTOR_NUM_NODES="$NNODES"
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-$rl_gpus_per_node}"
[[ "$ACTOR_GPUS_PER_NODE" -eq "$rl_gpus_per_node" ]] || \
  die "ACTOR_GPUS_PER_NODE must match the $rl_gpus_per_node GPUs advertised to Ray"
export NUM_GPUS_PER_NODE="$ACTOR_GPUS_PER_NODE"
export ROLLOUT_NUM_GPUS="$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export COLOCATE=1
export ROLLOUT_BATCH_SIZE="$TRAIN_BATCH_SIZE"
export N_SAMPLES_PER_PROMPT="$ROLLOUT_N"
export NUM_STEPS_PER_ROLLOUT="$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))"
export GLOBAL_BATCH_SIZE="$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))"
export NUM_ROLLOUT="${NUM_ROLLOUT:-165}"
export TOTAL_TRAINING_STEPS="$NUM_ROLLOUT"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.5}"
# Sixteen in-flight requests per one-GPU engine gives the same cluster-wide
# concurrency target as VERL: 14 engines x 16 = 224 trajectories.
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-16}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-32768}"
export MAX_CONCURRENT_REQUESTS="$((ROLLOUT_NUM_GPUS * SGLANG_SERVER_CONCURRENCY))"
export ENABLE_CHUNKED_PREFILL=True
export MAX_TOKENS_PER_TURN=512
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_SHUFFLE=1
export BALANCE_DATA=1
export CALCULATE_PER_TOKEN_LOSS=1

export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export KL_LOSS_ENABLED=1
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-$MODEL_PATH}"
export ENTROPY_COEF="${ENTROPY_COEF:-0}"

export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-$REPO_ROOT/slime_visual_agent/config/groundingdino_kl.yaml}"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES="${VISUAL_AGENT_ALLOWED_TOOL_NAMES:-grounding_detect,crop_zoom}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
export VISUAL_AGENT_IMAGE_MAX_PIXELS="${VISUAL_AGENT_IMAGE_MAX_PIXELS:-2359296}"
export VISUAL_AGENT_IMAGE_PATCH_SIZE="${VISUAL_AGENT_IMAGE_PATCH_SIZE:-16}"

export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_groundingdino_config.yaml}"
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-3}"
export GROUNDING_DINO_MAX_TEXT_TOKENS="${GROUNDING_DINO_MAX_TEXT_TOKENS:-256}"
export VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT="${VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT:-60}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-zwz_original_relation_qwen3_base_2node_groundingdino_kl_slime_${JOB_TOKEN}}"
# ModelArts injects a broad OUTPUT_DIR. Use the same RL_OUTPUT_DIR override
# contract as the VERL launcher so all artifacts remain under this experiment.
export OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"
export LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}"
export PROMPT_DATA="${PROMPT_DATA:-$OUTPUT_DIR/data/train.jsonl}"
export PREPARE_LIMIT=0
export REQUIRE_IMAGES=0
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-0}"
export DUMP_DETAILS="${DUMP_DETAILS:-}"

# The outer launcher owns this dedicated Ray cluster and tool pool. The SLIME
# child only submits its driver to the already-running local dashboard.
export START_LOCAL_RAY=0
export RAY_MASTER_ADDR=127.0.0.1
export RAY_DASHBOARD_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT"
export CONFIRM_DEDICATED_RESOURCES=1
export TRAIN_DRIVER_SCRIPT="$REPO_ROOT/scripts/run_visual_agent_slime_poc.sh"

exec bash "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"
