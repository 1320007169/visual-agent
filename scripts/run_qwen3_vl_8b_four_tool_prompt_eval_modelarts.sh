#!/usr/bin/env bash
set -Eeuo pipefail

# ModelArts single-node evaluation of Qwen3-VL-8B with five prompted visual
# tools. The direct-answer baseline has already been evaluated, so this entry
# runs only the tool-guided treatment.

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

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
MINICONDA_PATH="${MINICONDA_PATH:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3}"
MODEL_PATH="${FIVE_TOOL_MODEL_PATH:-${FOUR_TOOL_MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}}"
PROMPT_FILE="${FIVE_TOOL_PROMPT_FILE:-${FOUR_TOOL_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_eval_five_tools.txt}}"
CONFIG_ONLY="${FIVE_TOOL_CONFIG_ONLY:-${FOUR_TOOL_CONFIG_ONLY:-0}}"
GROUP_ID="${FIVE_TOOL_RUN_ID:-${FOUR_TOOL_RUN_ID:-qwen3_vl_8b_five_tool_prompt_$(date +%Y%m%d_%H%M%S)}}"
GROUP_ROOT="${FIVE_TOOL_OUTPUT_ROOT:-${FOUR_TOOL_OUTPUT_ROOT:-$REPO_ROOT/outputs/vlmeval/five_tool_prompt}}/$GROUP_ID"

[[ -f "$PIPELINE_ROOT/.env" ]] || { echo "Error: missing VTS environment file: $PIPELINE_ROOT/.env" >&2; exit 2; }
set -a
source "$PIPELINE_ROOT/.env"
set +a

# Keep the model-serving runtime identical to the proven Qwen3.8 synthesis
# entrypoint. In particular, do not inherit ModelArts' /usr/local/cuda value.
export CONDA_NO_PLUGINS=true
export VLLM_SOURCE="${VLLM_SOURCE:-$BASE/visual-tools/src/vllm-cu128}"
export OPENAI_OVERLAY="${OPENAI_OVERLAY:-$BASE/python-overlays/vllm026}"
export CUDA_HOME="${VTS_CUDA_HOME:-$BASE/cuda-12.8-toolkit}"
export VTS_TORCH_LIB="${VTS_TORCH_LIB:-$VTS_QWEN_ENV/lib/python3.10/site-packages/torch/lib}"
export PYTHONPATH="$PIPELINE_ROOT/src:$VLLM_SOURCE:$OPENAI_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VTS_QWEN_ENV/lib:$VTS_TORCH_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ENV_DIR="${ENV_DIR:-$VTS_QWEN_ENV}"
export VLLM_PYTHON="${VLLM_PYTHON:-$ENV_DIR/bin/python}"
export VLMEVAL_ENV_DIR="${VLMEVAL_ENV_DIR:-$BASE/conda_envs/visual-agent-eval}"
export VLMEVAL_PYTHON="${VLMEVAL_PYTHON:-$VLMEVAL_ENV_DIR/bin/python}"
export VLMEVAL_LD_LIBRARY_PATH="${VLMEVAL_LD_LIBRARY_PATH:-$VLMEVAL_ENV_DIR/lib}"
export VLMEVAL_PYTHONPATH="${VLMEVAL_PYTHONPATH-}"
export VLMEVAL_IMPORT_PREFLIGHT=1
default_config_python="$MINICONDA_PATH/bin/python3"
[[ -x "$default_config_python" ]] || default_config_python="$VTS_QWEN_ENV/bin/python3"
export CONFIG_PYTHON="${CONFIG_PYTHON:-$default_config_python}"
export TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export PADDLEX_MODEL_ROOT="${PADDLEX_MODEL_ROOT:-$BASE/visual-tools/paddlex-cache/official_models}"
export MODEL_CUDA_HOME="$CUDA_HOME"
export MODEL_CC="${MODEL_CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
export MODEL_CXX="${MODEL_CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
export SKIP_CONDA_ACTIVATION=1

# Reuse the converted local benchmark data and model cache.
export LMUData="${LMUData:-$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# GPU 0: GroundingDINO; GPU 1: OCR + depth; GPU 2: counting; GPUs 3-7:
# five independent Qwen3-VL replicas. OCR and depth use the same placement as
# the existing synthesis job and have already been exercised together.
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0}"
export VTS_OCR_GPU="${VTS_OCR_GPU:-1}"
export VTS_DEPTH_GPU="${VTS_DEPTH_GPU:-1}"
export VTS_COUNT_GPU="${VTS_COUNT_GPU:-2}"
export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-3,4,5,6,7}"
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0

export EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K OCRBench MME-RealWorld-Lite MME-RealWorld-CN}"
export VISUAL_AGENT_MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-6}"
export VISUAL_AGENT_MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-6144}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export MODEL_SERVER_BACKEND="${MODEL_SERVER_BACKEND:-vllm}"
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-20}"
export VLMEVAL_FAILED_SAMPLE_RETRIES="${VLMEVAL_FAILED_SAMPLE_RETRIES:-5}"
export EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-0}"

export VTS_OCR_ENDPOINT="${VTS_OCR_ENDPOINT:-http://127.0.0.1:9002}"
export VTS_DEPTH_ENDPOINT="${VTS_DEPTH_ENDPOINT:-http://127.0.0.1:9003}"
export VTS_COUNT_ENDPOINT="${VTS_COUNT_ENDPOINT:-http://127.0.0.1:9004}"
export VTS_TOOL_BRIDGE_ROOT="${VTS_TOOL_BRIDGE_ROOT:-$VTS_OUTPUT_ROOT/visual_agent_bridge/$GROUP_ID}"
export VTS_TOOL_TIMEOUT="${VTS_TOOL_TIMEOUT:-300}"
export LOG_DIR="$GROUP_ROOT/logs"
export VTS_SERVICE_DIR="$GROUP_ROOT/tool_services"

OCR_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/paddlex_ocrv5.yaml"
OCR_PIPELINE_CONFIG="$PIPELINE_ROOT/configs/tools/paddlex_ocrv5_server.yaml"
OCR_MODEL_PREPARE="$PIPELINE_ROOT/scripts/prepare_paddlex_ocrv5_models.sh"

case "$CONFIG_ONLY" in 0|1) ;; *) echo "FIVE_TOOL_CONFIG_ONLY must be 0 or 1" >&2; exit 2 ;; esac
[[ "$GROUP_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid FIVE_TOOL_RUN_ID: $GROUP_ID" >&2; exit 2; }

required_paths=(
  "$CONFIG_PYTHON"
  "$ENV_DIR/bin/python"
  "$VLMEVAL_PYTHON"
  "$TOOL_ENV_DIR/bin/python"
  "$MODEL_PATH/config.json"
  "$MODEL_PATH/model.safetensors.index.json"
  "$PROMPT_FILE"
  "$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh"
  "$MODEL_CC"
  "$MODEL_CXX"
  "$LMUData"
)
if [[ "$CONFIG_ONLY" == 0 ]]; then
  required_paths+=(
    "$OCR_SERVICE_CONFIG"
    "$OCR_PIPELINE_CONFIG"
    "$OCR_MODEL_PREPARE"
    "$PIPELINE_ROOT/configs/services/depth_anything_3.yaml"
    "$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
    "$VTS_OCR_ENV/bin/python3"
    "$VTS_DEPTH_ENV/bin/python3"
    "$VTS_COUNT_ENV/bin/python3"
    "$DEPTH_ANYTHING_3_MODEL/config.json"
    "$COUNTGDPP_CHECKPOINT"
  )
fi
for required in "${required_paths[@]}"; do
  [[ -e "$required" ]] || { echo "Error: missing required path: $required" >&2; exit 2; }
done

if [[ "$CONFIG_ONLY" == 0 ]]; then
  bash "$OCR_MODEL_PREPARE"
  for model in PP-OCRv5_server_det PP-OCRv5_server_rec PP-LCNet_x1_0_textline_ori; do
    for artifact in inference.json inference.pdiparams inference.yml; do
      required="$PADDLEX_MODEL_ROOT/$model/$artifact"
      [[ -s "$required" ]] || { echo "Error: incomplete OCR model artifact: $required" >&2; exit 2; }
    done
  done
fi

"$CONFIG_PYTHON" - "$MODEL_PATH" "$PROMPT_FILE" "$EVAL_DATASETS" "$GROUP_ROOT" \
  "$GROUP_ID" "$CONFIG_ONLY" "$VISUAL_AGENT_MAX_TURNS" "$VISUAL_AGENT_MAX_TOKENS" \
  "$MAX_MODEL_LEN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(model_text, prompt_text, dataset_text, output_text, run_id, config_only,
 turns_text, tokens_text, context_text) = sys.argv[1:]
datasets = dataset_text.split()
if not datasets or len(datasets) != len(set(datasets)):
    raise SystemExit("EVAL_DATASETS must be a nonempty unique list")
turns, tokens, context = int(turns_text), int(tokens_text), int(context_text)
if turns < 2 or tokens < 1 or context < tokens:
    raise SystemExit("Invalid turn/token/context budget")

model = Path(model_text).resolve()
config = json.loads((model / "config.json").read_text())
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards or any(not (model / shard).is_file() or (model / shard).stat().st_size == 0 for shard in shards):
    raise SystemExit(f"Incomplete model checkpoint: {model}")

prompt_path = Path(prompt_text).resolve()
prompt = prompt_path.read_text(encoding="utf-8").strip()
allowed = ["crop_zoom", "grounding_detect", "object_count", "ocr_read", "ground_depth"]
for name in allowed:
    if name not in prompt:
        raise SystemExit(f"Tool prompt does not mention {name}")
for forbidden in ["depth_measure", "sam3_"]:
    if forbidden in prompt:
        raise SystemExit(f"Tool prompt exposes forbidden tool {forbidden}")

protocol = {
    "experiment": "qwen3-vl-8b prompted five-visual-tool evaluation",
    "checkpoint": str(model),
    "model_type": config.get("model_type"),
    "mode": "five_tools",
    "datasets": datasets,
    "tools": allowed,
    "prompt_file": str(prompt_path),
    "prompt_sha256": hashlib.sha256((prompt + "\n").encode()).hexdigest(),
    "max_turns": turns,
    "max_tokens_per_turn": tokens,
    "max_model_len": context,
    "temperature": 0,
}
print(json.dumps(protocol, ensure_ascii=False, indent=2))
print(f"Results: {output_text}")
if config_only == "0":
    output = Path(output_text)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "tool_services").mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    (output / "five_tools_prompt.txt").write_text(prompt + "\n")
PY

[[ "$CONFIG_ONLY" == 1 ]] && exit 0

mkdir -p "$VTS_TOOL_BRIDGE_ROOT"
command -v conda >/dev/null 2>&1 || export PATH="$MINICONDA_PATH/bin:$PATH"
command -v conda >/dev/null 2>&1 || { echo "Error: conda is unavailable" >&2; exit 2; }

VTS_SERVICE_PIDS=()
cleanup_services() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${VTS_SERVICE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${VTS_SERVICE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_vts_service() {
  local name="$1" environment="$2" gpu="$3" config="$4"
  echo "Starting VTS $name service on GPU $gpu"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -p "$environment" \
      python3 -m vts.tool_server --config "$config" \
      >"$VTS_SERVICE_DIR/$name.log" 2>&1 &
  VTS_SERVICE_PIDS+=("$!")
  echo "$!" >"$VTS_SERVICE_DIR/$name.pid"
}

wait_vts_service() {
  local name="$1" endpoint="$2" pid="$3" deadline=$((SECONDS + ${VTS_SERVICE_STARTUP_TIMEOUT:-3600}))
  until "$CONFIG_PYTHON" - "$endpoint" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
    sys.argv[1].rstrip("/") + "/health", timeout=5
) as response:
    payload = json.load(response)
    raise SystemExit(0 if response.status < 400 and payload.get("status") == "ok" else 1)
PY
  do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: VTS $name service exited during startup" >&2
      tail -n 100 "$VTS_SERVICE_DIR/$name.log" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Error: VTS $name service startup timed out: $endpoint" >&2
      tail -n 100 "$VTS_SERVICE_DIR/$name.log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "VTS $name service ready: $endpoint"
}

warmup_ocr_service() {
  local endpoint="$1" warmup_root="$VTS_TOOL_BRIDGE_ROOT/ocr_warmup"
  mkdir -p "$warmup_root/artifacts"
  "$CONFIG_PYTHON" - "$endpoint" "$warmup_root" <<'PY'
import json
import os
from pathlib import Path
import struct
import sys
import urllib.request

endpoint, root_text = sys.argv[1:]
root = Path(root_text)
image = root / "warmup.bmp"
width, height = 640, 160
pixels = bytearray([255]) * (width * height * 3)
for top, bottom, left, right in ((35, 125, 55, 95), (35, 70, 95, 175),
                                 (90, 125, 95, 175), (35, 125, 205, 245),
                                 (35, 70, 245, 325), (90, 125, 245, 325),
                                 (35, 125, 355, 395), (35, 70, 395, 475),
                                 (90, 125, 395, 475)):
    for y in range(top, bottom):
        start = (y * width + left) * 3
        end = (y * width + right) * 3
        pixels[start:end] = bytes(end - start)
pixel_size = len(pixels)
bitmap_header = (
    b"BM"
    + struct.pack("<IHHI", 54 + pixel_size, 0, 0, 54)
    + struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_size, 2835, 2835, 0, 0)
)
image.write_bytes(bitmap_header + pixels)

payload = {
    "tool": "ocr_read",
    "args": {"image_id": 0, "minimum_confidence": 0.0},
    "context": {
        "uid": "ocr-model-warmup",
        "images": [{"image_id": 0, "path": str(image)}],
        "artifact_dir": str(root / "artifacts"),
        "branch_id": "preflight",
    },
}
headers = {"Content-Type": "application/json"}
token = os.environ.get("VTS_TOOL_SERVICE_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"
request = urllib.request.Request(
    endpoint.rstrip("/") + "/execute",
    data=json.dumps(payload).encode(),
    headers=headers,
    method="POST",
)
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=900) as response:
    result = json.load(response)
if result.get("status") != "success" and result.get("error_code") != "no_text":
    raise SystemExit(f"OCR warmup failed: {json.dumps(result, ensure_ascii=False)}")
print(f"OCR model warmup passed: status={result.get('status')} error_code={result.get('error_code')}")
PY
}

start_vts_service ocr "$VTS_OCR_ENV" "$VTS_OCR_GPU" "$OCR_SERVICE_CONFIG"
start_vts_service depth "$VTS_DEPTH_ENV" "$VTS_DEPTH_GPU" "$PIPELINE_ROOT/configs/services/depth_anything_3.yaml"
start_vts_service count "$VTS_COUNT_ENV" "$VTS_COUNT_GPU" "$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
wait_vts_service ocr "$VTS_OCR_ENDPOINT" "${VTS_SERVICE_PIDS[0]}"
warmup_ocr_service "$VTS_OCR_ENDPOINT"
wait_vts_service depth "$VTS_DEPTH_ENDPOINT" "${VTS_SERVICE_PIDS[1]}"
wait_vts_service count "$VTS_COUNT_ENDPOINT" "${VTS_SERVICE_PIDS[2]}"

export RUN_ID="${GROUP_ID}_five_tools"
export VLMEVAL_EVAL_ID="T_${RUN_ID}"
export WORK_ROOT="$GROUP_ROOT/five_tools"
export EVAL_MODELS=dino_latest
export RL_DINO_LATEST_STEP=0
export RL_DINO_LATEST_MODEL_PATH="$MODEL_PATH"
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$GROUP_ROOT/five_tools_prompt.txt"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=crop_zoom,grounding_detect,object_count,ocr_read,ground_depth

echo "Starting five-tool prompt evaluation on $MODEL_PATH: $EVAL_DATASETS"
started=$SECONDS
status=0
bash "$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh" || status=$?
printf 'mode\texit_code\tseconds\tresult_dir\n' >"$GROUP_ROOT/status.tsv"
printf 'five_tools\t%s\t%s\t%s\n' "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >>"$GROUP_ROOT/status.tsv"

if [[ -f "$REPO_ROOT/scripts/summarize_four_tool_prompt_eval.py" ]]; then
  env PYTHONPATH="$VLMEVAL_PYTHONPATH" LD_LIBRARY_PATH="$VLMEVAL_LD_LIBRARY_PATH" \
    "$VLMEVAL_PYTHON" "$REPO_ROOT/scripts/summarize_four_tool_prompt_eval.py" \
    "$GROUP_ROOT" --output "$GROUP_ROOT/behavior_summary.json" || {
      echo "Warning: behavior summary generation failed" >&2
      [[ "$status" -ne 0 ]] || status=1
    }
fi

echo "Five-tool evaluation finished: $GROUP_ROOT/status.tsv"
exit "$status"
