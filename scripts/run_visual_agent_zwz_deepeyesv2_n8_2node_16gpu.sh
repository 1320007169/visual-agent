#!/usr/bin/env bash
set -Eeuo pipefail

# New mixed-data run; inherit the GroundingDINO KL-safe configuration.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
# Ignore a stale job-level RL_ENV_DIR pointing at the older DeepEyes env.
# These mount paths refer to the same shared Qwen3-VL environment.
export RL_ENV_DIR="${MIXED_RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
if [[ -z "${MIXED_RL_ENV_DIR:-}" && ! -x "$RL_ENV_DIR/bin/python" ]]; then
  for dataset_root in /home/ma-user/work/dataset /opt/huawei/dataset; do
    candidate="$dataset_root/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl"
    if [[ -x "$candidate/bin/python" ]]; then
      export RL_ENV_DIR="$candidate"
      break
    fi
  done
fi
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://api-cn.hi-code.cc/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4.1-flash}"
export LLM_AS_A_JUDGE_BACKUP_BASE="${LLM_AS_A_JUDGE_BACKUP_BASE:-https://api.deepseek.com/v1}"
export LLM_AS_A_JUDGE_BACKUP_MODEL="${LLM_AS_A_JUDGE_BACKUP_MODEL:-deepseek-v4-flash}"
BACKUP_JUDGE_KEY_FILE="${BACKUP_JUDGE_KEY_FILE:-$BASE/secrets/deepseek_api_key.txt}"
if [[ -z "${LLM_AS_A_JUDGE_BACKUP_KEY:-}" && "${MIXED_CONFIG_ONLY:-0}" != "1" ]]; then
  [[ -r "$BACKUP_JUDGE_KEY_FILE" ]] || { echo "error: backup judge key file is not readable: $BACKUP_JUDGE_KEY_FILE" >&2; exit 2; }
  IFS= read -r LLM_AS_A_JUDGE_BACKUP_KEY < "$BACKUP_JUDGE_KEY_FILE" || true
  LLM_AS_A_JUDGE_BACKUP_KEY="${LLM_AS_A_JUDGE_BACKUP_KEY%$'\r'}"
  [[ -n "$LLM_AS_A_JUDGE_BACKUP_KEY" ]] || { echo "error: backup judge key file is empty" >&2; exit 2; }
  export LLM_AS_A_JUDGE_BACKUP_KEY
fi
JUDGE_KEY_FILE="${JUDGE_KEY_FILE:-$BASE/secrets/hicode_judge_api_key.txt}"
if [[ -z "${LLM_AS_A_JUDGE_KEY:-}" && "${MIXED_CONFIG_ONLY:-0}" != "1" ]]; then
  [[ -r "$JUDGE_KEY_FILE" ]] || { echo "error: judge key file is not readable: $JUDGE_KEY_FILE" >&2; exit 2; }
  IFS= read -r LLM_AS_A_JUDGE_KEY < "$JUDGE_KEY_FILE" || true
  LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY%$'\r'}"
  [[ -n "$LLM_AS_A_JUDGE_KEY" ]] || { echo "error: judge key file is empty" >&2; exit 2; }
  export LLM_AS_A_JUDGE_KEY
fi
export ZWZ_RL_DIR="${ZWZ_RL_DIR:-$REPO_ROOT/data/zwz_deepeyesv2_3k_nocount_hr4k_v1}"
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$RL_ROOT/verl/utils/dataset/zwz_deepeyesv2_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-ZwzDeepEyesV2Dataset}"
export NNODES="${NNODES:-2}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU=7
export TOOL_CUDA_VISIBLE_DEVICES=7
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-28}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-112}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export RESUME_MODE="${RESUME_MODE:-auto}"
# Hydra null lets the trainer calculate one epoch after prompt filtering.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
RESOLVED_ENV="$("${CONFIG_PYTHON:-python}" "$REPO_ROOT/scripts/resolve_mixed_rl_resume.py")"
eval "$RESOLVED_ENV"
# Skip the full image/token scan at startup; overlong samples still raise on read.
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
# Avoid forking filtering workers inside the Ray training process.
export FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-1}"
export VAL_BEFORE_TRAIN=False
export VAL_FILES="${VAL_FILES:-$REPO_ROOT/data/hrbench4k_rl_validation/val.parquet}"
export TEST_FREQ="${TEST_FREQ:-40}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
# The FSDP writer prunes before saving. Keep the prior checkpoint during the
# write; trainer-level pruning leaves just one after the new save succeeds.
export MAX_ACTOR_CKPT_TO_KEEP=2
export MAX_CHECKPOINTS_TO_KEEP=1
export SAVE_BEST_ONLY=False
export SAVE_BEST_HF_MODEL="${SAVE_BEST_HF_MODEL:-True}"
export BEST_METRIC=val-core/visual-agent-hrbench4k/acc/mean@1
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$RL_OUTPUT_DIR/validation}"
# Periodic in-training HRBench4K replaces the duplicate post-training run.
export POST_TRAIN_SCRIPT=""
if [[ "${MIXED_CONFIG_ONLY:-0}" == "1" ]]; then
  for key in RL_ENV_DIR NNODES RUN_ID RL_OUTPUT_DIR MODEL_PATH RESUME_MODE RESUME_FROM_PATH \
    WARM_START_GLOBAL_STEP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N \
    TOTAL_TRAINING_STEPS TOTAL_EPOCHS VAL_FILES TEST_FREQ SAVE_FREQ BEST_METRIC \
    RL_CUDA_VISIBLE_DEVICES VISUAL_TOOL_DEVICE VISUAL_TOOL_SERVERS_PER_NODE \
    GROUNDING_DINO_REPLICAS SAVE_BEST_HF_MODEL; do
    printf '%s=%s\n' "$key" "${!key:-}"
  done
  exit 0
fi
[[ -x "$RL_ENV_DIR/bin/python" ]] || { echo "error: Qwen3-VL environment is missing: $RL_ENV_DIR" >&2; exit 2; }
# vLLM uses spawned processes; reuse the headless overlay from the step130 launcher.
OPENCV_HEADLESS_SITE_PACKAGES="${OPENCV_HEADLESS_SITE_PACKAGES:-${RL_ENV_DIR%/*}/fineanno_wdz/lib/python3.10/site-packages}"
OPENCV_HEADLESS_OVERLAY="${OPENCV_HEADLESS_OVERLAY:-/tmp/visual-agent-opencv-python-headless-5.0.0.93}"
for package in cv2 opencv_python_headless.libs; do
  [[ -d "$OPENCV_HEADLESS_SITE_PACKAGES/$package" ]] || {
    echo "error: headless OpenCV package not found: $OPENCV_HEADLESS_SITE_PACKAGES/$package" >&2
    exit 2
  }
done
mkdir -p "$OPENCV_HEADLESS_OVERLAY"
for package in cv2 opencv_python_headless.libs; do
  [[ -e "$OPENCV_HEADLESS_OVERLAY/$package" || -L "$OPENCV_HEADLESS_OVERLAY/$package" ]] || \
    ln -s "$OPENCV_HEADLESS_SITE_PACKAGES/$package" "$OPENCV_HEADLESS_OVERLAY/$package"
done
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
"$RL_ENV_DIR/bin/python" - "$MODEL_PATH" <<'PYPROCESSOR'
import sys
import transformers
from transformers import AutoProcessor, Qwen3VLProcessor

processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
if not isinstance(processor, Qwen3VLProcessor):
    raise RuntimeError(f"Expected Qwen3VLProcessor, got {type(processor).__name__}")
print(f"Qwen3-VL preflight: python={sys.executable}, transformers={transformers.__version__}, processor={type(processor).__name__}", flush=True)
PYPROCESSOR
exec bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh"
