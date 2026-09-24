#!/usr/bin/env bash
set -Eeuo pipefail

# Train the original Qwen3-VL-8B-Instruct with raw depth and TallyQA counting QA.
# Per node: GPUs 0-6 run VERL; GPU 7 runs two GroundingDINO, two depth,
# and two count model processes.
BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924}"
export BASE REPO_ROOT PIPELINE_ROOT
export RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOOL_GPU="${TOOL_GPU:-7}"
export TOOL_CUDA_VISIBLE_DEVICES="$TOOL_GPU"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
export VAL_FILES="${VAL_FILES:-$DATA_DIR/val.parquet}"
export DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-messages}"
export CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-$REPO_ROOT/reinforcement_learning/verl/utils/dataset/zwz_deepeyesv2_dataset.py}"
export CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-ZwzDeepEyesV2Dataset}"
export TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_config.yaml}"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="${VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_multitool_depth_count.txt}"
export VISUAL_TOOL_BACKEND=groundingdino
export SAM3_REPLICAS=0
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-1}"
export VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-2}"

export NNODES="${NNODES:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-276}"
export MAX_TURNS="${MAX_TURNS:-8}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export SAVE_FREQ="${SAVE_FREQ:-40}"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_depth_tallyqa5k_multitool_n16_${JOB_TOKEN}}"

ensure_symlink() {
  local target="$1" link_path="$2"
  if [[ ! -e "$link_path" && ! -L "$link_path" ]]; then
    mkdir -p "$(dirname "$link_path")"
    ln -s "$target" "$link_path"
  fi
}

ensure_symlink /opt/huawei/quoteModel/xiaoyi_tmpstorage /home/ma-user/work/model/xiaoyi_tmpstorage
[[ -f "$PIPELINE_ROOT/.env" ]] || { echo "error: missing VTS environment file: $PIPELINE_ROOT/.env" >&2; exit 2; }
set -a
source "$PIPELINE_ROOT/.env"
set +a

export VTS_TOOL_BRIDGE_ROOT="${VTS_TOOL_BRIDGE_ROOT:-${VTS_OUTPUT_ROOT:-$BASE/outputs}/visual_agent_bridge/$RUN_ID/${HOSTNAME:-node}}"
export COUNT_SERVICE_CONFIG="${COUNT_SERVICE_CONFIG:-$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml}"
DEPTH_SERVICE_CONFIG="${DEPTH_SERVICE_CONFIG:-$PIPELINE_ROOT/configs/services/depth_anything_3.yaml}"
VTS_DEPTH_PORTS="${VTS_DEPTH_PORTS:-9003,9005}"
VTS_COUNT_PORTS="${VTS_COUNT_PORTS:-9004,9006}"
START_VTS_SERVICES="${START_VTS_SERVICES:-1}"
VTS_SERVICE_STARTUP_TIMEOUT="${VTS_SERVICE_STARTUP_TIMEOUT:-3600}"
[[ "$VTS_SERVICE_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: VTS_SERVICE_STARTUP_TIMEOUT must be a positive integer" >&2
  exit 2
}
IFS=',' read -r -a depth_ports <<< "$VTS_DEPTH_PORTS"
IFS=',' read -r -a count_ports <<< "$VTS_COUNT_PORTS"
[[ ${#depth_ports[@]} -eq 2 && ${#count_ports[@]} -eq 2 ]] || {
  echo "error: configure exactly two depth ports and two count ports" >&2
  exit 2
}
for port in "${depth_ports[@]}" "${count_ports[@]}"; do
  [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) || {
    echo "error: invalid VTS port: $port" >&2
    exit 2
  }
done
unique_port_count="$(printf '%s\n' "${depth_ports[@]}" "${count_ports[@]}" | sort -u | wc -l)"
(( unique_port_count == 4 )) || {
  echo "error: VTS ports must be distinct" >&2
  exit 2
}
export VTS_DEPTH_ENDPOINT="${VTS_DEPTH_ENDPOINT:-http://127.0.0.1:${depth_ports[0]},http://127.0.0.1:${depth_ports[1]}}"
export VTS_COUNT_ENDPOINT="${VTS_COUNT_ENDPOINT:-http://127.0.0.1:${count_ports[0]},http://127.0.0.1:${count_ports[1]}}"
if [[ "$START_VTS_SERVICES" == "1" ]]; then
  [[ -n "${VTS_DEPTH_ENV:-}" && -n "${VTS_COUNT_ENV:-}" ]] || {
    echo "error: VTS_DEPTH_ENV and VTS_COUNT_ENV are required" >&2
    exit 2
  }
  export VTS_DEPTH_ENDPOINT="http://127.0.0.1:${depth_ports[0]},http://127.0.0.1:${depth_ports[1]}"
  export VTS_COUNT_ENDPOINT="http://127.0.0.1:${count_ports[0]},http://127.0.0.1:${count_ports[1]}"
fi
[[ "$START_VTS_SERVICES" == "0" || "$START_VTS_SERVICES" == "1" ]] || {
  echo "error: START_VTS_SERVICES must be 0 or 1" >&2
  exit 2
}
for required in "$COUNT_SERVICE_CONFIG" "$DEPTH_SERVICE_CONFIG"; do
  [[ -f "$required" ]] || { echo "error: missing VTS service config: $required" >&2; exit 2; }
done
token_env="${VTS_TOOL_SERVICE_TOKEN_ENV:-VTS_TOOL_SERVICE_TOKEN}"
if [[ -z "${!token_env:-}" ]]; then
  echo "error: set $token_env for the VTS depth and count services" >&2
  exit 2
fi

VTS_SERVICE_PIDS=()
cleanup_vts() {
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
trap cleanup_vts EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$START_VTS_SERVICES" == "1" ]]; then
  for required in "$VTS_DEPTH_ENV/bin/python3" "$VTS_COUNT_ENV/bin/python3"; do
    [[ -x "$required" ]] || { echo "error: VTS Python is missing: $required" >&2; exit 2; }
  done
  command -v conda >/dev/null 2>&1 || export PATH="${MINICONDA_PATH:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3}/bin:$PATH"
  command -v conda >/dev/null 2>&1 || { echo "error: conda is unavailable" >&2; exit 2; }
  export PYTHONPATH="$PIPELINE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  VTS_SERVICE_DIR="${VTS_SERVICE_DIR:-$BASE/logs/visual-agent-multitool/$RUN_ID/${HOSTNAME:-node}}"
  mkdir -p "$VTS_SERVICE_DIR" "$VTS_TOOL_BRIDGE_ROOT"
  "$VTS_DEPTH_ENV/bin/python3" - "${depth_ports[@]}" "${count_ports[@]}" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", int(value)))
        except OSError as exc:
            raise SystemExit(f"VTS port {value} is unavailable: {exc}") from exc
PY

  create_vts_config() {
    local source="$1" target="$2" port="$3" python="$4"
    "$python" - "$source" "$target" "$port" <<'PY'
from pathlib import Path
import sys
import yaml

source, target, port = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
if not isinstance(config, dict) or not isinstance(config.get("server"), dict):
    raise SystemExit(f"VTS config has no server section: {source}")
config["server"]["host"] = "127.0.0.1"
config["server"]["port"] = int(port)
Path(target).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
  }

  start_vts_service() {
    local name="$1" environment="$2" config="$3" port="$4"
    local generated_config="$VTS_SERVICE_DIR/$name.yaml"
    create_vts_config "$config" "$generated_config" "$port" "$environment/bin/python3"
    setsid env CUDA_VISIBLE_DEVICES="$TOOL_GPU" PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -p "$environment" \
        python3 -m vts.tool_server --config "$generated_config" \
        >"$VTS_SERVICE_DIR/$name.log" 2>&1 &
    VTS_SERVICE_PIDS+=("$!")
    echo "Started VTS $name on GPU $TOOL_GPU, port $port; log: $VTS_SERVICE_DIR/$name.log"
  }

  for index in 0 1; do
    start_vts_service "depth$index" "$VTS_DEPTH_ENV" "$DEPTH_SERVICE_CONFIG" "${depth_ports[index]}"
  done
  for index in 0 1; do
    start_vts_service "count$index" "$VTS_COUNT_ENV" "$COUNT_SERVICE_CONFIG" "${count_ports[index]}"
  done
fi

IFS=',' read -r -a vts_endpoints <<< "$VTS_DEPTH_ENDPOINT,$VTS_COUNT_ENDPOINT"
for index in "${!vts_endpoints[@]}"; do
  endpoint="${vts_endpoints[index]}"
  deadline=$((SECONDS + VTS_SERVICE_STARTUP_TIMEOUT))
  until curl --noproxy '*' --fail --silent --max-time 5 "$endpoint/health" >/dev/null; do
    if [[ "$START_VTS_SERVICES" == "1" ]] && ! kill -0 "${VTS_SERVICE_PIDS[index]}" 2>/dev/null; then
      echo "error: VTS service exited before becoming healthy: $endpoint" >&2
      exit 2
    fi
    (( SECONDS < deadline )) || { echo "error: VTS service did not become healthy: $endpoint" >&2; exit 2; }
    sleep 5
  done
  echo "VTS service ready: $endpoint"
done

bash "$REPO_ROOT/scripts/run_visual_agent_zwz_rl_2node_16gpu.sh"
