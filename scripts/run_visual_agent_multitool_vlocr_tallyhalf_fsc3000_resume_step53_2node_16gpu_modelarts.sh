#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the 56-rank step 53 checkpoint on 14 training GPUs in a separate run.
export NNODES=2
export ALLOW_FSDP_WORLD_SIZE_CHANGE=True
export DATALOADER_NUM_WORKERS=2
export TRAIN_BATCH_SIZE=336
export PPO_MINI_BATCH_SIZE=112
export VAL_BATCH_SIZE=336
export ROLLOUT_N=16
export MAX_CONCURRENT_REQUESTS=112
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_2node_from_8node_step53}"
export TRAIN_RUN_TOKEN=resume53_16gpu_20261007_gitfix2
export TRAIN_SHUFFLE=True
export RL_SPLIT_OCR=0
export RL_CHART_PARSE=1
export VTS_VL_OCR=1
export TRAINER_STOP_AFTER_SECONDS=0
# Allocate a new run namespace before selecting the source checkpoint.
export RESUME_MODE=disable
export TOOL_CONFIG_PATH=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE=/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/prompts/visual_agent_rl_system_multitool_vlocr.txt

# Generated ModelArts entrypoint; run the same file on every node.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
export COUNT_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
export RL_ENV_DIR="${MULTITOOL_RL_ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl}"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004}"
export TRAIN_FILES="${TRAIN_FILES:-$MULTITOOL_DATA_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$MULTITOOL_DATA_DIR/val.parquet}"
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-55}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_53}"
export TEST_FREQ="${TEST_FREQ:-40}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-1}"
export SAVE_BEST_ONLY=False
export SAVE_BEST_HF_MODEL="${SAVE_BEST_HF_MODEL:-True}"
export BEST_METRIC="${BEST_METRIC:-val-core/visual-agent/acc/macro_mean}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export POST_TRAIN_SCRIPT=""
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
export FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-1}"
export VISUAL_AGENT_IMAGE_TRANSPORT="${VISUAL_AGENT_IMAGE_TRANSPORT:-source_cached}"
export GROUNDING_DINO_MAX_QUERY_WORDS="${GROUNDING_DINO_MAX_QUERY_WORDS:-12}"
export GROUNDING_QUERY_MAX_WORDS="${GROUNDING_QUERY_MAX_WORDS:-12}"

export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$RUN_ID}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$RL_OUTPUT_DIR/validation}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-zwz-rl/$RUN_ID}"
export RL_LOG_DIR="${RL_LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}"

export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://43.155.134.160:8080/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4-flash}"
export LLM_AS_A_JUDGE_BACKUP_BASE="${LLM_AS_A_JUDGE_BACKUP_BASE:-https://api.deepseek.com/v1}"
export LLM_AS_A_JUDGE_BACKUP_MODEL="${LLM_AS_A_JUDGE_BACKUP_MODEL:-deepseek-v4-flash}"


launcher="$REPO_ROOT"/scripts/run_visual_agent_multitool_vlocr_2node_16gpu.sh
if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
    export RESUME_MODE=resume_path
    for key in RL_ENV_DIR MODEL_PATH TRAIN_FILES VAL_FILES COUNT_SERVICE_CONFIG NNODES RL_CUDA_VISIBLE_DEVICES \
        TOOL_GPU RUN_ID RL_OUTPUT_DIR RL_LOG_DIR ROLLOUT_DATA_DIR RESUME_MODE TRAIN_SHUFFLE TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE VAL_BATCH_SIZE \
        ROLLOUT_N TOTAL_TRAINING_STEPS TEST_FREQ SAVE_FREQ BEST_METRIC VAL_BEFORE_TRAIN \
        VISUAL_AGENT_IMAGE_TRANSPORT MAX_CONCURRENT_REQUESTS RL_SPLIT_OCR TOOL_CONFIG_PATH \
        VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE TRAINER_STOP_AFTER_SECONDS RESUME_FROM_PATH SYNC_DIR \
        ALLOW_FSDP_WORLD_SIZE_CHANGE DATALOADER_NUM_WORKERS; do
        printf '%s=%s\n' "$key" "${!key:-}"
    done
    printf 'LAUNCHER=%s\n' "$launcher"
    exit 0
fi

mkdir -p /opt/huawei/explorer-env /home/ma-user/work/algorithm /home/ma-user/work/model
ensure_symlink() {
    local target="$1" link_path="$2"
    if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
        ln -s "$target" "$link_path"
    fi
}
ensure_symlink /opt/huawei/dataset /opt/huawei/explorer-env/dataset
ensure_symlink /opt/huawei/dataset /home/ma-user/work/dataset
ensure_symlink /opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage
source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
export RESUME_MODE=resume_path

# Keep Ray session files separate from other runs and use a short socket path.
run_path_hash="$(printf '%s' "$RUN_ID" | sha256sum)"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/va-ray-${run_path_hash:0:12}}"
DIAGNOSTIC_DIR="$RL_LOG_DIR/diagnostics/${HOSTNAME:-node}"
mkdir -p "$DIAGNOSTIC_DIR"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_FR_DUMP_TEMP_FILE="$DIAGNOSTIC_DIR/nccl_trace_rank_"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$TORCH_FR_DUMP_TEMP_FILE"
# main_ppo.py sets NCCL_DEBUG=WARN in the Ray runtime environment.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE="$DIAGNOSTIC_DIR/nccl.%h.%p.log"

capture_node_state() {
    local phase="$1" counter
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$DIAGNOSTIC_DIR/$phase.timestamp"
    nvidia-smi -q > "$DIAGNOSTIC_DIR/$phase.nvidia-smi.txt" 2>&1 || true
    dmesg -T > "$DIAGNOSTIC_DIR/$phase.dmesg.txt" 2>&1 || true
    {
        for counter in /sys/class/infiniband/*/ports/*/{counters,hw_counters}/*; do
            [[ -f "$counter" ]] || continue
            printf '%s=' "$counter"
            cat "$counter"
        done
    } > "$DIAGNOSTIC_DIR/$phase.rdma-counters.txt" 2>&1 || true
}

preserve_node_logs() {
    local status=$? ray_logs ray_name
    trap - EXIT
    set +e
    printf '%s\n' "$status" > "$DIAGNOSTIC_DIR/exit_status.txt"
    capture_node_state exit
    local -a ray_log_dirs=(/tmp/va-ray-*/session_latest/logs)
    if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
        ray_log_dirs=("$RAY_TEMP_DIR/session_latest/logs")
    fi
    for ray_logs in "${ray_log_dirs[@]}"; do
        [[ -d "$ray_logs" ]] || continue
        ray_name="$(basename "$(dirname "$(dirname "$ray_logs")")")"
        cp -a "$ray_logs" "$DIAGNOSTIC_DIR/$ray_name" 2>> "$DIAGNOSTIC_DIR/collection-errors.txt"
    done
    exit "$status"
}
trap preserve_node_logs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

capture_node_state start
{
    printf 'ENTRYPOINT=%s\n' "${BASH_SOURCE[0]}"
    for key in RUN_ID RESUME_MODE RESUME_FROM_PATH TOTAL_TRAINING_STEPS TRAIN_FILES VAL_FILES \
        RL_OUTPUT_DIR RL_LOG_DIR ROLLOUT_DATA_DIR SYNC_DIR TORCH_NCCL_ASYNC_ERROR_HANDLING \
        TORCH_NCCL_ENABLE_MONITORING TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT \
        TORCH_NCCL_DESYNC_DEBUG TORCH_FR_DUMP_TEMP_FILE NCCL_DEBUG NCCL_DEBUG_FILE NCCL_IB_DISABLE \
        ALLOW_FSDP_WORLD_SIZE_CHANGE DATALOADER_NUM_WORKERS RAY_TEMP_DIR; do
        printf '%s=%s\n' "$key" "${!key:-}"
    done
    # Record file hashes without requiring Git access on shared ModelArts mounts.
    sha256sum "${BASH_SOURCE[0]}" "$REPO_ROOT/reinforcement_learning/verl/workers/fsdp_workers.py"
} > "$DIAGNOSTIC_DIR/launch.txt"

[[ -s "$RESUME_FROM_PATH/data.pt" ]] || {
    echo "error: checkpoint data cursor is missing: $RESUME_FROM_PATH/data.pt" >&2
    exit 2
}
resume_rank0_files=("$RESUME_FROM_PATH"/actor/model_world_size_*_rank_0.pt)
[[ ${#resume_rank0_files[@]} == 1 && -s "${resume_rank0_files[0]}" ]] || {
    echo "error: missing or ambiguous checkpoint topology: $RESUME_FROM_PATH" >&2
    exit 2
}
resume_world_size="${resume_rank0_files[0]##*/model_world_size_}"
resume_world_size="${resume_world_size%_rank_0.pt}"
[[ "$resume_world_size" == 56 || "$resume_world_size" == 14 ]] || {
    echo "error: expected a 56-rank source or a 14-rank continuation checkpoint" >&2
    exit 2
}
for ((rank = 0; rank < resume_world_size; rank++)); do
    for component in model optim extra_state; do
        checkpoint_file="$RESUME_FROM_PATH/actor/${component}_world_size_${resume_world_size}_rank_${rank}.pt"
        [[ -s "$checkpoint_file" ]] || {
            echo "error: checkpoint shard is missing: $checkpoint_file" >&2
            exit 2
        }
    done
done

if [[ -z "${MULTITOOL_RL_ENV_DIR:-}" && ! -x "$RL_ENV_DIR/bin/python3" ]]; then
    for dataset_root in /home/ma-user/work/dataset /opt/huawei/dataset; do
        candidate="$dataset_root/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl"
        if [[ -x "$candidate/bin/python3" ]]; then
            export RL_ENV_DIR="$candidate"
            break
        fi
    done
fi
[[ -x "$RL_ENV_DIR/bin/python3" ]] || { echo "error: Qwen3-VL environment is missing: $RL_ENV_DIR" >&2; exit 2; }
export RL_PYTHON="$RL_ENV_DIR/bin/python3"

export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

load_key() {
    local variable_name="$1" key_file="$2" value="${!1:-}"
    if [[ -z "$value" ]]; then
        [[ -r "$key_file" ]] || { echo "error: judge key file is not readable: $key_file" >&2; exit 2; }
        IFS= read -r value < "$key_file" || true
        value="${value%$'\r'}"
        [[ -n "$value" ]] || { echo "error: judge key file is empty: $key_file" >&2; exit 2; }
        printf -v "$variable_name" '%s' "$value"
        export "$variable_name"
    fi
}
if [[ "${ENABLE_API_JUDGE:-1}" != "0" ]]; then
    load_key LLM_AS_A_JUDGE_BACKUP_KEY "${BACKUP_JUDGE_KEY_FILE:-$BASE/secrets/deepseek_api_key.txt}"
    load_key LLM_AS_A_JUDGE_KEY "${JUDGE_KEY_FILE:-$BASE/secrets/judge_api_43_155_134_160_key.txt}"
fi

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
"$RL_ENV_DIR/bin/python3" - "$MODEL_PATH" <<'PYMODEL'
import sys
import cv2
from transformers import AutoProcessor, Qwen3VLProcessor

gui = next(line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
           if line.strip().startswith("GUI:"))
if gui != "NONE":
    raise RuntimeError(f"Headless OpenCV overlay is not active: GUI={gui}")
processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
if not isinstance(processor, Qwen3VLProcessor):
    raise RuntimeError(f"Expected Qwen3VLProcessor, got {type(processor).__name__}")
print(f"ModelArts preflight passed: {type(processor).__name__}, OpenCV GUI={gui}", flush=True)
PYMODEL

# Keep this shell alive so its EXIT trap can preserve node-local logs.
bash "$launcher" "$@"
