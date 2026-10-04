#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate a VL-OCR RL checkpoint with its training tools, prompt, and PaddleOCR-VL backend.
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
[[ -n "${VLOCR_MODEL_PATH:-}" && -n "${VLOCR_STEP:-}" ]] || {
    echo "error: set VLOCR_MODEL_PATH (global_step_*/actor/huggingface) and VLOCR_STEP" >&2
    exit 2
}
export RUN_ID="${RUN_ID:-multitool_vlocr_step${VLOCR_STEP}_native${VLOCR_NATIVE_TOOLS:-1}_$(date +%Y%m%dT%H%M%S%N)_$$}"
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/multitool_vlocr_8gpu/$RUN_ID}"
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

# Deliberately overrides .env; VLOCR_DEPTH_ENV exists only so tests can use a fake service.
export VTS_DEPTH_ENV="${VLOCR_DEPTH_ENV:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/starVLA_flash_dzw1}"
export VTS_DEPTH_ENDPOINT=http://127.0.0.1:9003
export VTS_COUNT_ENDPOINT=http://127.0.0.1:9004
export VTS_CHART_ENDPOINT=http://127.0.0.1:9007
export VTS_VL_OCR=1
export PADDLEOCR_VL_MODEL_ROOT="${PADDLEOCR_VL_MODEL_ROOT:-$BASE/visual-tools/paddlex-cache/official_models/PaddleOCR-VL-1.6}"
export VTS_CHART_ENV="${VTS_CHART_ENV:-$BASE/conda_envs/qwen38-vllm-clean}"
export VTS_TOOL_BRIDGE_ROOT="$VTS_OUTPUT_ROOT/visual_agent_bridge/$RUN_ID"
export LMUData="$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export ENV_DIR="${ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM}"
export SKIP_CONDA_ACTIVATION=1
export VLMEVAL_IMPORT_PREFLIGHT=1
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export MODEL_CUDA_HOME="$CUDA_HOME"
export MODEL_CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export MODEL_CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-7}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-7}"
export MODEL_SERVER_BACKEND=vllm
export EVAL_MODELS=dino_latest
export EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench8K OCRBench MME-RealWorld-Lite HRBench4K MME-RealWorld-CN CV-Bench-2D CV-Bench-3D ChartQA_TEST FSC147_TEST}"
export VLMEVAL_CHARTQA_RULE_ONLY=1
FSC147_ANNOTATION_FILE="${FSC147_ANNOTATION_FILE:-$BASE/visual-tools/src/CountGDPlusPlus/data/fscd147/countgd_test_fscd147.json}"
FSC147_IMAGE_ROOT="${FSC147_IMAGE_ROOT:-$BASE/visual-tools/datasets/fsc147/images_384_VarV2}"
export VISUAL_TOOL_BACKEND=groundingdino SAM3_REPLICAS=0 GROUNDING_DINO_REPLICAS=1
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=crop_zoom,grounding_detect,depth_measure,object_count,ocr_read
export VISUAL_AGENT_MAX_TURNS=8 VISUAL_AGENT_MAX_TOKENS=512
# Native tools render the training tool schemas as the Qwen <tools> block and use the RL
# rollout's hermes parser. VLOCR_NATIVE_TOOLS=0 reproduces the historical prompt-only protocol.
case "${VLOCR_NATIVE_TOOLS:-1}" in
    1)
        export VISUAL_AGENT_TOOL_CONFIG_PATH="$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml"
        export VLLM_TOOL_CALL_PARSER=hermes
        ;;
    0) unset VISUAL_AGENT_TOOL_CONFIG_PATH VLLM_TOOL_CALL_PARSER ;;
    *) echo "error: VLOCR_NATIVE_TOOLS must be 0 or 1" >&2; exit 2 ;;
esac
export RL_DINO_LATEST_STEP="$VLOCR_STEP" RL_DINO_LATEST_MODEL_PATH="$VLOCR_MODEL_PATH"

eval_script="$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh"
[[ -f "$eval_script" ]] || { echo "error: evaluation launcher is missing: $eval_script" >&2; exit 2; }
if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    # The child launcher creates WORK_ROOT even during preflight; keep it out of the real output.
    preflight_work_root="$(mktemp -d "${TMPDIR:-/tmp}/visual-agent-eval-preflight.XXXXXXXX")"
    trap 'rm -rf -- "$preflight_work_root"' EXIT
    WORK_ROOT="$preflight_work_root" bash "$eval_script"
    exit 0
fi

# Refuse reuse of an existing experiment directory.
mkdir -p "$(dirname "$WORK_ROOT")"
mkdir "$WORK_ROOT"
mkdir -p "$WORK_ROOT/services" "$VTS_TOOL_BRIDGE_ROOT"
if [[ -n "${VLOCR_REUSE_GROUP_ROOT:-}" ]]; then
    export VLMEVAL_EVAL_ID="${VLMEVAL_EVAL_ID:-$RUN_ID}"
    python3 - "$VLOCR_REUSE_GROUP_ROOT" "$VLOCR_MODEL_PATH" \
        "$WORK_ROOT/dino_latest/VisualAgent-vllm/$VLMEVAL_EVAL_ID" <<'PY'
import csv
import os
from pathlib import Path
import shutil
import sys

previous, model, destination = map(Path, sys.argv[1:])
with (previous / "checkpoints.tsv").open() as stream:
    for checkpoint in csv.DictReader(stream, delimiter="\t"):
        if Path(checkpoint["model_path"]).resolve() != model.resolve():
            continue
        label = f'{checkpoint["checkpoint"]}_step{checkpoint["step"]}'
        source = previous / label / "dino_latest/VisualAgent-vllm"
        if not source.is_dir():
            continue
        prediction_roots = sorted(path for path in source.iterdir() if path.is_dir())
        if not prediction_roots:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        # Seed predictions only; VLMEvalKit retries API failures and recomputes scores.
        for dataset in os.environ["EVAL_DATASETS"].split():
            for suffix in (".xlsx", "_supp.pkl"):
                cached = prediction_roots[-1] / f"VisualAgent-vllm_{dataset}{suffix}"
                if cached.is_file():
                    shutil.copy2(cached, destination / cached.name)
                    print(f"Reusing predictions: {cached}", flush=True)
PY
fi
if [[ " $EVAL_DATASETS " == *" FSC147_TEST "* ]]; then
    "$ENV_DIR/bin/python" "$REPO_ROOT/scripts/prepare_fsc147_eval.py" \
        --annotation-file "$FSC147_ANNOTATION_FILE" --image-root "$FSC147_IMAGE_ROOT" \
        --output "$LMUData/FSC147_TEST.tsv" --download-images "${FSC147_DOWNLOAD_IMAGES:-0}"
fi
service_pids=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    # Signal each service group so all subprocesses release their GPU and port.
    for pid in "${service_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${service_pids[@]}"; do
        for _ in $(seq 1 30); do
            kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    printf 'Launcher exit code: %s\n' "$status"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" "$VTS_DEPTH_ENV/bin/python3" \
    -m vts.tool_server --config "$PIPELINE_ROOT/configs/services/depth_anything_3.yaml" \
    >"$WORK_ROOT/services/depth.log" 2>&1 &
service_pids+=("$!")
setsid env CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" "$VTS_COUNT_ENV/bin/python3" \
    -m vts.tool_server --config "$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml" \
    >"$WORK_ROOT/services/count.log" 2>&1 &
service_pids+=("$!")
# Keep the chart environment libraries and inherited platform driver paths.
chart_library_path="$VTS_CHART_ENV/lib"
for torch_library in "$VTS_CHART_ENV"/lib/python*/site-packages/torch/lib; do
    [[ ! -d "$torch_library" ]] || chart_library_path+=":$torch_library"
done
chart_library_path+="${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
setsid env CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH="$chart_library_path" \
    "$VTS_CHART_ENV/bin/python3" "$REPO_ROOT/scripts/paddleocr_vl_chart_server.py" \
    --model-root "$PADDLEOCR_VL_MODEL_ROOT" --allowed-root "$VTS_TOOL_BRIDGE_ROOT" \
    --max-new-tokens "${CHART_PARSE_MAX_NEW_TOKENS:-1024}" --port 9007 \
    >"$WORK_ROOT/services/chart.log" 2>&1 &
service_pids+=("$!")

for endpoint in "$VTS_DEPTH_ENDPOINT" "$VTS_COUNT_ENDPOINT" "$VTS_CHART_ENDPOINT"; do
    deadline=$((SECONDS + 3600))
    until curl --noproxy '*' --fail --silent "$endpoint/health" >/dev/null; do
        for pid in "${service_pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                printf 'error: VTS service exited before becoming ready: %s; logs: %s\n' "$endpoint" "$WORK_ROOT/services" >&2
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

printf 'Evaluating step %s: %s\n' "$VLOCR_STEP" "$VLOCR_MODEL_PATH"
bash "$eval_script"
