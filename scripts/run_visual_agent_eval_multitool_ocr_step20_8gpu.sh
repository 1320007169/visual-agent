#!/usr/bin/env bash
set -Eeuo pipefail

# Keep the previous step80 eight-GPU protocol; update checkpoint and OCR tools.
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export RUN_ID="${RUN_ID:-multitool_ocr_64gpu_step20_24gpu_step40_$(date +%Y%m%dT%H%M%S%N)_$$}"
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/multitool_ocr_step20_8gpu/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-eval}"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"
launcher_log="$LOG_DIR/$RUN_ID-launcher.log"
exec > >(tee -a "$launcher_log") 2>&1
trap 'printf "error: launcher failed at line %s, exit code %s\n" "$LINENO" "$?" >&2' ERR
printf 'Launcher log: %s\nResults: %s\n' "$launcher_log" "$WORK_ROOT"

PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
set -a
source "$PIPELINE_ROOT/.env"
set +a

export VTS_DEPTH_ENV=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/starVLA_flash_dzw1
export VTS_OCR_ENDPOINT=http://127.0.0.1:9002
export VTS_DEPTH_ENDPOINT=http://127.0.0.1:9003
export VTS_COUNT_ENDPOINT=http://127.0.0.1:9004
export VTS_TOOL_BRIDGE_ROOT="$VTS_OUTPUT_ROOT/visual_agent_bridge/$RUN_ID"
export PADDLEX_MODEL_ROOT="${PADDLEX_MODEL_ROOT:-$BASE/visual-tools/paddlex-cache/official_models}"
export LMUData="$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export ENV_DIR=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM
export SKIP_CONDA_ACTIVATION=1
export VLMEVAL_IMPORT_PREFLIGHT=1
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export MODEL_CUDA_HOME="$CUDA_HOME"
export MODEL_CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export MODEL_CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export TOOL_CUDA_HOME=/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118
export MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export TOOL_CUDA_VISIBLE_DEVICES=7
export VLMEVAL_API_NPROC=14
export MODEL_SERVER_BACKEND=vllm
export EVAL_MODELS=dino_latest
export EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench8K OCRBench MME-RealWorld-Lite HRBench4K MME-RealWorld-CN CV-Bench-2D CV-Bench-3D ChartQA_TEST FSC147_TEST}"
export VLMEVAL_CHARTQA_RULE_ONLY=1
FSC147_ANNOTATION_FILE="${FSC147_ANNOTATION_FILE:-$BASE/visual-tools/src/CountGDPlusPlus/data/fscd147/countgd_test_fscd147.json}"
FSC147_IMAGE_ROOT="${FSC147_IMAGE_ROOT:-$BASE/visual-tools/datasets/fsc147/images_384_VarV2}"
STEP20_MODEL_PATH="${STEP20_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_text_det_rec_n16_8node_20260928T154734173299_33c232ee/global_step_20/actor/huggingface}"
STEP24_MODEL_PATH="${STEP24_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_text_det_rec_n16_3node_20260928T101658645173_53c116e6/global_step_40/actor/huggingface}"
export VISUAL_TOOL_BACKEND=groundingdino SAM3_REPLICAS=0 GROUNDING_DINO_REPLICAS=1
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_depth_count_ocr.txt"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=crop_zoom,grounding_detect,depth_measure,object_count,text_detect,text_recognize
export VISUAL_AGENT_MAX_TURNS=8 VISUAL_AGENT_MAX_TOKENS=512
read -r -a checkpoints <<< "${EVAL_CHECKPOINTS:-64gpu_step20 24gpu_step40}"
[[ "${#checkpoints[@]}" -gt 0 ]] || { echo "error: no checkpoints selected" >&2; exit 2; }
for checkpoint in "${checkpoints[@]}"; do
    case "$checkpoint" in
        64gpu_step20|24gpu_step40) ;;
        *) echo "error: unsupported checkpoint: $checkpoint" >&2; exit 2 ;;
    esac
done

eval_script="$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh"
[[ -f "$eval_script" ]] || { echo "error: evaluation launcher is missing: $eval_script" >&2; exit 2; }
base_run_id="$RUN_ID"
base_work_root="$WORK_ROOT"
select_checkpoint() {
    local checkpoint="$1"
    export RUN_ID="${base_run_id}_${checkpoint}"
    export WORK_ROOT="$base_work_root/$checkpoint"
    if [[ "$checkpoint" == 64gpu_step20 ]]; then
        export RL_DINO_LATEST_STEP=20 RL_DINO_LATEST_MODEL_PATH="$STEP20_MODEL_PATH"
    else
        export RL_DINO_LATEST_STEP=40 RL_DINO_LATEST_MODEL_PATH="$STEP24_MODEL_PATH"
    fi
}
if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    # The child launcher creates WORK_ROOT even during preflight. Keep those
    # directories separate so the same run can still start with a fresh output.
    preflight_work_root="$(mktemp -d "${TMPDIR:-/tmp}/visual-agent-eval-preflight.XXXXXXXX")"
    cleanup_preflight() {
        local status=$?
        trap - EXIT INT TERM
        rm -rf -- "$preflight_work_root" || true
        exit "$status"
    }
    trap cleanup_preflight EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    for checkpoint in "${checkpoints[@]}"; do
        select_checkpoint "$checkpoint"
        export WORK_ROOT="$preflight_work_root/$checkpoint"
        bash "$eval_script"
    done
    exit 0
fi

# Refuse reuse of an existing experiment directory.
mkdir -p "$(dirname "$WORK_ROOT")"
mkdir "$WORK_ROOT"
mkdir -p "$WORK_ROOT/services" "$VTS_TOOL_BRIDGE_ROOT"
if [[ " $EVAL_DATASETS " == *" FSC147_TEST "* ]]; then
    "$ENV_DIR/bin/python" "$REPO_ROOT/scripts/prepare_fsc147_eval.py" \
        --annotation-file "$FSC147_ANNOTATION_FILE" --image-root "$FSC147_IMAGE_ROOT" \
        --point-annotation-file "${FSC147_POINT_ANNOTATION_FILE:-$BASE/datasets/fsc147/annotation_FSC147_384.json}" \
        --output "$LMUData/FSC147_TEST.tsv" --download-images "${FSC147_DOWNLOAD_IMAGES:-0}"
fi
service_pids=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${service_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${service_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    printf 'Launcher exit code: %s\n' "$status"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES=7 "$VTS_DEPTH_ENV/bin/python3" \
    -m vts.tool_server --config "$PIPELINE_ROOT/configs/services/depth_anything_3.yaml" \
    >"$WORK_ROOT/services/depth.log" 2>&1 &
service_pids+=("$!")
CUDA_VISIBLE_DEVICES=7 "$VTS_COUNT_ENV/bin/python3" \
    -m vts.tool_server --config "$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml" \
    >"$WORK_ROOT/services/count.log" 2>&1 &
service_pids+=("$!")
CUDA_VISIBLE_DEVICES=7 "$VTS_OCR_ENV/bin/python3" \
    "$REPO_ROOT/scripts/paddleocr_split_server.py" --model-root "$PADDLEX_MODEL_ROOT" --port 9002 \
    >"$WORK_ROOT/services/ocr.log" 2>&1 &
service_pids+=("$!")

for endpoint in "$VTS_DEPTH_ENDPOINT" "$VTS_COUNT_ENDPOINT" "$VTS_OCR_ENDPOINT"; do
    deadline=$((SECONDS + 3600))
    until curl --noproxy '*' --fail --silent "$endpoint/health" >/dev/null; do
        for pid in "${service_pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                printf 'error: VTS service exited before becoming ready: %s\n' "$endpoint" >&2
                exit 1
            fi
        done
        if (( SECONDS >= deadline )); then
            printf 'error: VTS service startup timed out: %s\n' "$endpoint" >&2
            exit 1
        fi
        sleep 5
    done
done

printf 'checkpoint\texit_code\tresult_dir\n' >"$base_work_root/status.tsv"
status=0
for checkpoint in "${checkpoints[@]}"; do
    select_checkpoint "$checkpoint"
    printf 'Evaluating %s: %s\n' "$checkpoint" "$RL_DINO_LATEST_MODEL_PATH"
    checkpoint_status=0
    bash "$eval_script" || checkpoint_status=$?
    if [[ "${VISUAL_AGENT_FAILURE_REPORT:-0}" == "1" ]]; then
        "$ENV_DIR/bin/python" "$REPO_ROOT/scripts/summarize_visual_agent_eval_failures.py" \
            --work-dir "$WORK_ROOT/dino_latest" --output "$WORK_ROOT/failure_summary.json" || checkpoint_status=1
    fi
    printf '%s\t%s\t%s\n' "$checkpoint" "$checkpoint_status" "$WORK_ROOT/dino_latest" >>"$base_work_root/status.tsv"
    if [[ "$checkpoint_status" != 0 ]]; then status=$checkpoint_status; break; fi
done
exit "$status"
