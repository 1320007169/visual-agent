#!/usr/bin/env bash
set -Eeuo pipefail

# Multi-node 100-step continuation from the preserved KL step130 checkpoint.
# Defaults to 4 nodes / 32 physical GPUs; NNODES=8 selects the 64-GPU run.
# A: Agent-only K=4. D: Agent K=4 + tool-free Native K=4 in one PPO batch.
# D intentionally has twice as many trajectories per outer step; compare it to
# A as an objective ablation, not as an equal-GPU-hour result.

STREAM_MODE="${STREAM_MODE:-${1:-}}"
case "$STREAM_MODE" in
  A|agent) STREAM_MODE=A ;;
  D|dual) STREAM_MODE=D ;;
  *) echo "usage: STREAM_MODE=A|D $0" >&2; exit 2 ;;
esac

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/zwz_rl_vqa/rl_original_relation}"
RL_ENV_DIR="${RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
OPENCV_HEADLESS_SITE_PACKAGES="${OPENCV_HEADLESS_SITE_PACKAGES:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/fineanno_wdz/lib/python3.10/site-packages}"
OPENCV_HEADLESS_OVERLAY="${OPENCV_HEADLESS_OVERLAY:-/tmp/visual-agent-opencv-python-headless-5.0.0.93}"

export BASE REPO_ROOT RL_ROOT RL_ENV_DIR
export NNODES="${NNODES:-4}"
PHYSICAL_GPU_COUNT="$((NNODES * 8))"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-7}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/preserved_groundingdino_kl_step130_huggingface}"
export TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA_DIR/val.parquet}"
export DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-messages}"
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$RL_ROOT/verl/utils/dataset/zwz_original_relation_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-ZwzOriginalRelationDataset}"
export PREPARE_SMOKE_DATA=0

# vLLM uses spawned Python processes. Force the known-good headless OpenCV
# build so those children do not import the GUI build that requires libGL.so.1.
[[ -d "$OPENCV_HEADLESS_SITE_PACKAGES/cv2" ]] || {
  echo "error: headless OpenCV cv2 directory not found: $OPENCV_HEADLESS_SITE_PACKAGES/cv2" >&2
  exit 2
}
[[ -d "$OPENCV_HEADLESS_SITE_PACKAGES/opencv_python_headless.libs" ]] || {
  echo "error: headless OpenCV libraries not found: $OPENCV_HEADLESS_SITE_PACKAGES/opencv_python_headless.libs" >&2
  exit 2
}
mkdir -p "$OPENCV_HEADLESS_OVERLAY"
[[ -e "$OPENCV_HEADLESS_OVERLAY/cv2" || -L "$OPENCV_HEADLESS_OVERLAY/cv2" ]] || \
  ln -s "$OPENCV_HEADLESS_SITE_PACKAGES/cv2" "$OPENCV_HEADLESS_OVERLAY/cv2"
[[ -e "$OPENCV_HEADLESS_OVERLAY/opencv_python_headless.libs" || -L "$OPENCV_HEADLESS_OVERLAY/opencv_python_headless.libs" ]] || \
  ln -s "$OPENCV_HEADLESS_SITE_PACKAGES/opencv_python_headless.libs" "$OPENCV_HEADLESS_OVERLAY/opencv_python_headless.libs"
export PYTHONPATH="$OPENCV_HEADLESS_OVERLAY:${PYTHONPATH:-}"

"$RL_ENV_DIR/bin/python" - <<'PYOPENCV'
import cv2

gui = next(
    line.split(":", 1)[1].strip()
    for line in cv2.getBuildInformation().splitlines()
    if line.strip().startswith("GUI:")
)
if gui != "NONE":
    raise RuntimeError(f"Headless OpenCV overlay is not active: GUI={gui}")
print(f"Headless OpenCV preflight passed: {cv2.__version__} ({cv2.__file__})", flush=True)
PYOPENCV

EXPECTED_TRAIN_SHA256="${EXPECTED_TRAIN_SHA256:-a5d501796fe345f0fffd572a356114ad2bbb566e02bf665a8e5359befc6ed82c}"
EXPECTED_VAL_SHA256="${EXPECTED_VAL_SHA256:-fc84967878a34973396d529f9ca0aae206d9bfe55be43972a1f8a4868b8609d5}"
for data_file in "$TRAIN_FILES" "$VAL_FILES"; do
  [[ -f "$data_file" ]] || { echo "error: dataset file not found: $data_file" >&2; exit 2; }
done
ACTUAL_TRAIN_SHA256="$(sha256sum "$TRAIN_FILES" | awk '{print $1}')"
ACTUAL_VAL_SHA256="$(sha256sum "$VAL_FILES" | awk '{print $1}')"
[[ "$ACTUAL_TRAIN_SHA256" == "$EXPECTED_TRAIN_SHA256" ]] || {
  echo "error: train parquet hash changed: $ACTUAL_TRAIN_SHA256" >&2
  exit 2
}
[[ "$ACTUAL_VAL_SHA256" == "$EXPECTED_VAL_SHA256" ]] || {
  echo "error: validation parquet hash changed: $ACTUAL_VAL_SHA256" >&2
  exit 2
}
for dataset_spec in "$TRAIN_FILES:$EXPECTED_TRAIN_SHA256" "$VAL_FILES:$EXPECTED_VAL_SHA256"; do
  dataset_path="${dataset_spec%%:*}"
  expected_hash="${dataset_spec##*:}"
  [[ -f "$dataset_path" ]] || { echo "error: dataset not found: $dataset_path" >&2; exit 2; }
  actual_hash="$(sha256sum "$dataset_path" | awk '{print $1}')"
  [[ "$actual_hash" == "$expected_hash" ]] || {
    echo "error: frozen dataset hash mismatch: $dataset_path" >&2
    echo "expected: $expected_hash" >&2
    echo "actual:   $actual_hash" >&2
    exit 2
  }
done

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-28}"
export ROLLOUT_N=4
export AGENT_ROLLOUT_N=4
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TOTAL_EPOCHS=1
export ACTOR_LR="${ACTOR_LR:-5e-7}"
export ACTOR_USE_KL_LOSS=True
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
export ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
export ACTOR_ENTROPY_COEFF="${ACTOR_ENTROPY_COEFF:-0}"
export ACTOR_LOSS_AGG_MODE=seq-mean-token-mean

if [[ "$STREAM_MODE" == "D" ]]; then
  export DUAL_STREAM_ENABLE=True
  export VISUAL_AGENT_DUAL_STREAM=1
  export NATIVE_ROLLOUT_N=4
  # VERL scales this value by rollout.n. 56 gives D the same four optimizer
  # steps per outer step as A, despite D containing twice as many trajectories.
  export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-56}"
  # Keep the corrected Native protocol isolated from the old run whose Native
  # branch received zero reward.  Reusing that directory would auto-resume the
  # invalid checkpoint, including its optimizer state.
  DEFAULT_RUN_ID="step130_dual_agent4_native4_nativefmtfix_100step_${PHYSICAL_GPU_COUNT}gpu"
else
  export DUAL_STREAM_ENABLE=False
  export VISUAL_AGENT_DUAL_STREAM=0
  export NATIVE_ROLLOUT_N=0
  export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-28}"
  DEFAULT_RUN_ID="step130_agent4_100step_${PHYSICAL_GPU_COUNT}gpu"
fi

export RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
export OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_execution_plan/$RUN_ID}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-execution-plan/$RUN_ID}"
export LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-execution-plan}"
# A first submission starts from MODEL_PATH. Re-submitting the same A or D
# entrypoint discovers the newest complete global_step_* checkpoint under its
# stable OUTPUT_DIR and restores model, optimizer, scheduler, and data state.
export RESUME_MODE="${RESUME_MODE:-auto}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export SAVE_HF_MODEL=1
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export MAX_CHECKPOINTS_TO_KEEP=2
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN=False
export TRAIN_SHUFFLE=True
export BALANCE_BATCH=False

export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
export VISUAL_AGENT_NATIVE_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_NATIVE_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_native_system.txt}"
if [[ "$STREAM_MODE" == "D" ]]; then
  grep -Fq '<answer>' "$VISUAL_AGENT_NATIVE_SYSTEM_PROMPT_FILE" && \
    grep -Fq '</answer>' "$VISUAL_AGENT_NATIVE_SYSTEM_PROMPT_FILE" || {
      echo "error: Native prompt must require a final <answer>...</answer> response" >&2
      exit 2
    }
fi
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_groundingdino_config.yaml}"
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-3}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
# Match the proven ZWZ RL preprocessing. The preserved step130 processor has
# max_pixels=null, so uncapped high-resolution images can exceed the 8192-token
# prompt limit before the first rollout starts.
export VISUAL_AGENT_IMAGE_MAX_PIXELS="${VISUAL_AGENT_IMAGE_MAX_PIXELS:-2359296}"
export VISUAL_AGENT_IMAGE_PATCH_SIZE="${VISUAL_AGENT_IMAGE_PATCH_SIZE:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-224}"
export VISUAL_AGENT_CHAT_COMPLETION_RETRIES="${VISUAL_AGENT_CHAT_COMPLETION_RETRIES:-2}"
export ENABLE_CHUNKED_PREFILL=True
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-512}"
export MAX_TURNS="${MAX_TURNS:-6}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export FILTER_OVERLONG_PROMPTS=False
export TRAINER_PROJECT_NAME="visual-agent-step130-stream-ablation"
export POST_TRAIN_SCRIPT="${POST_TRAIN_SCRIPT:-$REPO_ROOT/scripts/run_visual_agent_step130_final_eval_8gpu.sh}"

export ENABLE_API_JUDGE="${ENABLE_API_JUDGE:-1}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://api.deepseek.com}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4-flash}"
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-${DEEPSEEK_API_KEY:-}}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.modelarts.com:80}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"

echo "Stream ablation: $STREAM_MODE"
echo "Dataset: $TRAIN_FILES"
echo "Start checkpoint: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

exec bash "$REPO_ROOT/scripts/run_visual_tool_rl_2node_16gpu.sh"
