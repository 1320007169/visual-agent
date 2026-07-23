#!/usr/bin/env bash
set -Eeuo pipefail

# One-to-four-node ModelArts GRPO launcher for Visual-Agent VLMs.
# Run the same command once on every node:
#   bash /home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/scripts/run_visual_tool_rl_2node_16gpu.sh
#
# Default placement on each 8-GPU node:
#   GPUs 0-6: VERL actor/ref/vLLM workers
#   GPU 7:    local SAM3 + GroundingDINO HTTP service
#
# ModelArts normally supplies the node list through VC_WORKER_HOSTS. If it does
# not, set the same MASTER_ADDR on every node and use ranks 0 to NNODES-1.

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/reinforcement_learning}"
RL_ENV_DIR="${RL_ENV_DIR:-$BASE/conda_envs/deepeyes-rl-conda}"
TOOL_ENV_DIR="${TOOL_ENV_DIR:-$BASE/conda_envs/visual-tools}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/saves/visual_tool_sft_v0/full/sft_visual_sft_20260714_021005/checkpoint-207}"
CUDA_HOME="${CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
CUDA_LIBRARY_DIR="${CUDA_LIBRARY_DIR:-$CUDA_HOME/lib64}"
TOOL_CUDA_HOME="${TOOL_CUDA_HOME:-${VISUAL_TOOL_CUDA_HOME:-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}}"
TOOL_CUDA_LIBRARY_DIR="${TOOL_CUDA_LIBRARY_DIR:-$TOOL_CUDA_HOME/lib64}"

PLATFORM_DATASET_ROOT="${PLATFORM_DATASET_ROOT:-/opt/huawei/dataset}"
PLATFORM_ALGORITHM_ROOT="${PLATFORM_ALGORITHM_ROOT:-/opt/huawei/schedule-train/algorithm/algorithmrefs/synaflow_wl}"
PLATFORM_MODEL_ROOT="${PLATFORM_MODEL_ROOT:-/opt/huawei/quoteModel/xiaoyi_tmpstorage}"

NNODES="${NNODES:-2}"
RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
TOOL_GPU="${TOOL_GPU:-7}"
TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-$TOOL_GPU}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-14}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-}"
MAX_TURNS="${MAX_TURNS:-9}"
TOOL_CALL_FORMAT="${TOOL_CALL_FORMAT:-hermes}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-28}"
PREPARE_SMOKE_DATA="${PREPARE_SMOKE_DATA:-1}"
DATA_PROMPT_KEY="${DATA_PROMPT_KEY:-prompt}"
CUSTOM_DATASET_PATH="${CUSTOM_DATASET_PATH:-}"
CUSTOM_DATASET_NAME="${CUSTOM_DATASET_NAME:-}"
FILTER_OVERLONG_WORKERS="${FILTER_OVERLONG_WORKERS:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-False}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_FREQ="${SAVE_FREQ:--1}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-0}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
WARM_START_DATA_PATH="${WARM_START_DATA_PATH:-}"
WARM_START_GLOBAL_STEP="${WARM_START_GLOBAL_STEP:-0}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-visual-agent-rl-smoke-2node}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLUSTER_TIMEOUT="${RAY_CLUSTER_TIMEOUT:-600}"
# Full ZWZ runs take several days. Keep worker nodes alive for one week while
# they wait for the node-0 driver to finish.
WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-604800}"

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-$REPO_ROOT/data/visual_tool_rl_smoke_2node}"
TRAIN_FILE="${TRAIN_FILE:-$SMOKE_DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$SMOKE_DATA_DIR/val.parquet}"
TRAIN_FILES="${TRAIN_FILES:-$TRAIN_FILE}"
VAL_FILES="${VAL_FILES:-$VAL_FILE}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$RL_ROOT/examples/sglang_multiturn/config/tool_config/visual_tool_config.yaml}"

VISUAL_TOOL_HOST="${VISUAL_TOOL_HOST:-0.0.0.0}"
VISUAL_TOOL_PORT="${VISUAL_TOOL_PORT:-9000}"
VISUAL_TOOL_SERVERS_PER_NODE="${VISUAL_TOOL_SERVERS_PER_NODE:-1}"
START_VISUAL_TOOL_SERVER="${START_VISUAL_TOOL_SERVER:-1}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$BASE/visual-tools/sam3/sam3.pt}"
GROUNDING_DINO_MODEL_PATH="${GROUNDING_DINO_MODEL_PATH:-$BASE/visual-tools/grounding-dino-base-transformers}"
SAM3_REPLICAS="${SAM3_REPLICAS:-1}"
GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-1}"
VISUAL_TOOL_STARTUP_TIMEOUT="${VISUAL_TOOL_STARTUP_TIMEOUT:-600}"

JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
RUN_ID="${RUN_ID:-visual_rl_2node16_${JOB_TOKEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/saves/visual_tool_rl_smoke_2node/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$BASE/logs/visual-tool-rl-2node}"
SYNC_DIR="${SYNC_DIR:-$BASE/tmp/visual-tool-rl-${NNODES}node/$RUN_ID}"
DONE_FILE="$SYNC_DIR/driver.done"

RL_PYTHON="$RL_ENV_DIR/bin/python"
RAY_BIN="$RL_ENV_DIR/bin/ray"
TOOL_PYTHON="$TOOL_ENV_DIR/bin/python"
TOOL_PIDS=()
# Do not use the generic RANK variable here: some ModelArts images set RANK=0
# independently on every worker before the user entrypoint starts. Prefer
# node-level platform variables, then fall back to matching the host list.
NODE_RANK="${NODE_RANK:-${MA_NODE_RANK:-${VC_TASK_INDEX:-${SLURM_NODEID:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-}}}}}}"
MASTER_ADDR="${MASTER_ADDR:-${MA_MASTER_ADDR:-}}"

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

ensure_symlink "$PLATFORM_DATASET_ROOT" /opt/huawei/explorer-env/dataset
ensure_symlink "$PLATFORM_DATASET_ROOT" /home/ma-user/work/dataset
ensure_symlink "$PLATFORM_ALGORITHM_ROOT" /home/ma-user/work/algorithm/synaflow_wl
ensure_symlink "$PLATFORM_MODEL_ROOT" /home/ma-user/work/model/xiaoyi_tmpstorage

[[ "$NNODES" =~ ^[1-4]$ ]] || die "this launcher supports NNODES from 1 to 4, got $NNODES"
[[ -d "$REPO_ROOT" ]] || die "repository not found: $REPO_ROOT"
[[ -d "$RL_ROOT/verl" ]] || die "VERL source not found: $RL_ROOT"
[[ -x "$RL_PYTHON" ]] || die "RL Python not executable: $RL_PYTHON"
[[ -x "$RAY_BIN" ]] || die "Ray executable not found: $RAY_BIN"
[[ -x "$TOOL_PYTHON" ]] || die "visual-tools Python not executable: $TOOL_PYTHON"
[[ -f "$MODEL_PATH/config.json" ]] || die "HF checkpoint config not found: $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || die "HF checkpoint weights not found: $MODEL_PATH"
[[ -f "$TOOL_CONFIG_PATH" ]] || die "visual tool config not found: $TOOL_CONFIG_PATH"
[[ -f "$SAM3_MODEL_PATH" ]] || die "SAM3 checkpoint not found: $SAM3_MODEL_PATH"
[[ -f "$GROUNDING_DINO_MODEL_PATH/config.json" ]] || die "GroundingDINO model not found: $GROUNDING_DINO_MODEL_PATH"
[[ -d "$CUDA_HOME" ]] || die "CUDA toolkit not found: $CUDA_HOME"
[[ -d "$CUDA_LIBRARY_DIR" ]] || die "CUDA library directory not found: $CUDA_LIBRARY_DIR"
[[ -d "$TOOL_CUDA_HOME" ]] || die "tool CUDA toolkit not found: $TOOL_CUDA_HOME"
[[ -d "$TOOL_CUDA_LIBRARY_DIR" ]] || die "tool CUDA library directory not found: $TOOL_CUDA_LIBRARY_DIR"

# Accept JSON arrays, comma-separated host lists, and ModelArts host-list envs.
HOST_LIST_RAW="${VC_WORKER_HOSTS:-${MA_WORKER_HOSTS:-${WORKER_HOSTS:-}}}"
if [[ -z "$MASTER_ADDR" || -z "$NODE_RANK" ]]; then
  if [[ -n "$HOST_LIST_RAW" ]]; then
    mapfile -t DETECTED_CLUSTER < <(
      "$RL_PYTHON" - "$HOST_LIST_RAW" "${MA_CURRENT_HOST:-${HOSTNAME:-}}" <<'PYCLUSTER'
import json
import socket
import sys

raw, current = sys.argv[1:]
try:
    parsed = json.loads(raw)
except json.JSONDecodeError:
    parsed = [item.strip().strip("'\"") for item in raw.strip("[]").split(",") if item.strip()]
if isinstance(parsed, dict):
    for key in ("hosts", "worker_hosts", "workers"):
        if key in parsed:
            parsed = parsed[key]
            break
hosts = [str(item) for item in parsed]
if not hosts:
    raise SystemExit("empty worker host list")

candidates = {current, socket.gethostname(), socket.getfqdn()}
candidates |= {item.split(".")[0] for item in list(candidates) if item}
rank = -1
for index, host in enumerate(hosts):
    if host in candidates or host.split(".")[0] in candidates:
        rank = index
        break
print(hosts[0])
print(rank)
PYCLUSTER
    )
    [[ -n "$MASTER_ADDR" ]] || MASTER_ADDR="${DETECTED_CLUSTER[0]:-}"
    if [[ -z "$NODE_RANK" && "${DETECTED_CLUSTER[1]:--1}" != "-1" ]]; then
      NODE_RANK="${DETECTED_CLUSTER[1]}"
    fi
  fi
fi

[[ -n "$MASTER_ADDR" ]] || die "MASTER_ADDR was not detected; set it to the node-0 hostname or IP"
if [[ "$NNODES" == "1" ]]; then
  [[ "${NODE_RANK:-0}" == "0" ]] || die "NODE_RANK must be 0 for a one-node run, got '${NODE_RANK:-unset}'"
  NODE_RANK=0
else
  [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be an integer; got '${NODE_RANK:-unset}'"
  (( NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
fi

MASTER_IP="$($RL_PYTHON - "$MASTER_ADDR" <<'PYRESOLVE'
import socket
import sys
print(socket.gethostbyname(sys.argv[1]))
PYRESOLVE
)"
NODE_IP="${NODE_IP:-$($RL_PYTHON - <<'PYNODEIP'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PYNODEIP
)}"

[[ "$VISUAL_TOOL_SERVERS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "VISUAL_TOOL_SERVERS_PER_NODE must be a positive integer"
LOCAL_VISUAL_TOOL_API_BASES="$($RL_PYTHON - "$VISUAL_TOOL_PORT" "$VISUAL_TOOL_SERVERS_PER_NODE" <<'PYLOCALTOOLS'
import sys

base_port, server_count = map(int, sys.argv[1:])
print(",".join(f"http://127.0.0.1:{base_port + index}" for index in range(server_count)))
PYLOCALTOOLS
)"
IFS=',' read -r -a LOCAL_VISUAL_TOOL_ENDPOINT_ARRAY <<< "$LOCAL_VISUAL_TOOL_API_BASES"
LOCAL_VISUAL_TOOL_API_BASE="${LOCAL_VISUAL_TOOL_ENDPOINT_ARRAY[0]}"

# The rollout driver runs on node 0, so localhost would leave node 1's tool
# GPU idle. Resolve every worker address and expose the resulting comma-
# separated endpoint list to all OnlineVisualTool instances.
VISUAL_TOOL_API_BASES="${VISUAL_TOOL_API_BASES:-${VISUAL_TOOL_API_BASE:-}}"
if [[ -z "$VISUAL_TOOL_API_BASES" ]]; then
  if [[ -n "$HOST_LIST_RAW" ]]; then
    VISUAL_TOOL_API_BASES="$($RL_PYTHON - "$HOST_LIST_RAW" "$VISUAL_TOOL_PORT" "$NNODES" "$VISUAL_TOOL_SERVERS_PER_NODE" <<'PYTOOLS'
import json
import socket
import sys

raw, port, node_count, servers_per_node = sys.argv[1:]
try:
    parsed = json.loads(raw)
except json.JSONDecodeError:
    parsed = [item.strip().strip("'\"") for item in raw.strip("[]").split(",") if item.strip()]
if isinstance(parsed, dict):
    for key in ("hosts", "worker_hosts", "workers"):
        if key in parsed:
            parsed = parsed[key]
            break
hosts = [str(item) for item in parsed][:int(node_count)]
if len(hosts) != int(node_count):
    raise SystemExit(f"expected {node_count} worker hosts, got {len(hosts)}")
endpoints = []
for host in hosts:
    address = socket.gethostbyname(host)
    endpoints.extend(
        f"http://{address}:{int(port) + index}"
        for index in range(int(servers_per_node))
    )
print(",".join(endpoints))
PYTOOLS
)"
  elif [[ "$NNODES" == "1" ]]; then
    VISUAL_TOOL_API_BASES="$($RL_PYTHON - "$NODE_IP" "$VISUAL_TOOL_PORT" "$VISUAL_TOOL_SERVERS_PER_NODE" <<'PYTOOLS'
import sys

host, port, server_count = sys.argv[1:]
print(",".join(f"http://{host}:{int(port) + index}" for index in range(int(server_count))))
PYTOOLS
)"
  else
    die "worker host list was not detected; set VISUAL_TOOL_API_BASES to all node endpoints"
  fi
fi
export VISUAL_TOOL_API_BASES
IFS=',' read -r -a VISUAL_TOOL_ENDPOINT_ARRAY <<< "$VISUAL_TOOL_API_BASES"
EXPECTED_TOOL_ENDPOINTS="$((NNODES * VISUAL_TOOL_SERVERS_PER_NODE))"
[[ "${#VISUAL_TOOL_ENDPOINT_ARRAY[@]}" -eq "$EXPECTED_TOOL_ENDPOINTS" ]] || {
  die "expected $EXPECTED_TOOL_ENDPOINTS visual-tool endpoints, got ${#VISUAL_TOOL_ENDPOINT_ARRAY[@]}: $VISUAL_TOOL_API_BASES"
}

[[ "$SAM3_REPLICAS" =~ ^[1-9][0-9]*$ ]] || die "SAM3_REPLICAS must be a positive integer"
[[ "$GROUNDING_DINO_REPLICAS" =~ ^[1-9][0-9]*$ ]] || die "GROUNDING_DINO_REPLICAS must be a positive integer"
[[ "$VISUAL_TOOL_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "VISUAL_TOOL_STARTUP_TIMEOUT must be a positive integer"
[[ "$WORKER_WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "WORKER_WAIT_TIMEOUT must be a positive integer"
[[ "$SAVE_HF_MODEL" == "0" || "$SAVE_HF_MODEL" == "1" ]] || die "SAVE_HF_MODEL must be 0 or 1"
[[ "$WARM_START_GLOBAL_STEP" =~ ^[0-9]+$ ]] || die "WARM_START_GLOBAL_STEP must be a non-negative integer"
if [[ -n "$WARM_START_DATA_PATH" ]]; then
  [[ -f "$WARM_START_DATA_PATH" || -f "$WARM_START_DATA_PATH/data.pt" ]] || {
    die "warm-start dataloader state not found: $WARM_START_DATA_PATH"
  }
fi
case "$RESUME_MODE" in
  auto|disable) ;;
  resume_path)
    [[ -n "$RESUME_FROM_PATH" ]] || die "RESUME_FROM_PATH is required when RESUME_MODE=resume_path"
    [[ "$RESUME_FROM_PATH" == *global_step_* ]] || die "RESUME_FROM_PATH must point to a global_step_* directory"
    ;;
  *) die "RESUME_MODE must be auto, disable, or resume_path; got $RESUME_MODE" ;;
esac

# The ModelArts node-local /tmp has a small quota. Large GRPO batches force
# Ray to spill many objects, so put spill objects on the shared high-capacity
# model volume. Keep the Ray session/socket root at a short local path because
# Linux AF_UNIX socket paths are limited to 107 bytes; the session logs are
# small and were not the source of the previous 50 GB quota exhaustion.
RAY_STORAGE_ROOT="${RAY_STORAGE_ROOT:-$BASE/tmp/ray-${NNODES}node/$RUN_ID}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/va-ray-${NODE_RANK}}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-$RAY_STORAGE_ROOT/node${NODE_RANK}/spill}"
export RAY_TMPDIR="$RAY_TEMP_DIR"

IFS=',' read -r -a RL_GPU_ARRAY <<< "$RL_CUDA_VISIBLE_DEVICES"
N_GPUS_PER_NODE="${#RL_GPU_ARRAY[@]}"
TOTAL_RL_GPUS="$((NNODES * N_GPUS_PER_NODE))"
(( N_GPUS_PER_NODE > 0 && N_GPUS_PER_NODE <= 8 )) || {
  die "expected between one and eight RL GPUs per node, got $N_GPUS_PER_NODE"
}
IFS=',' read -r -a TOOL_GPU_ARRAY <<< "$TOOL_CUDA_VISIBLE_DEVICES"
TOOL_GPU_COUNT="${#TOOL_GPU_ARRAY[@]}"
(( TOOL_GPU_COUNT > 0 )) || die "TOOL_CUDA_VISIBLE_DEVICES must contain at least one GPU"
for gpu in "${RL_GPU_ARRAY[@]}" "${TOOL_GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || die "GPU index must be between 0 and 7, got '$gpu'"
done
for rl_gpu in "${RL_GPU_ARRAY[@]}"; do
  for tool_gpu in "${TOOL_GPU_ARRAY[@]}"; do
    [[ "$rl_gpu" != "$tool_gpu" ]] || die "GPU $rl_gpu is assigned to both RL and visual tools"
  done
done
(( VISUAL_TOOL_SERVERS_PER_NODE >= TOOL_GPU_COUNT )) || {
  die "VISUAL_TOOL_SERVERS_PER_NODE must be at least the number of tool GPUs ($TOOL_GPU_COUNT)"
}
[[ "$TRAIN_BATCH_SIZE" -ge "$TOTAL_RL_GPUS" ]] || {
  die "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be at least total RL GPUs ($TOTAL_RL_GPUS)"
}
(( TRAIN_BATCH_SIZE % TOTAL_RL_GPUS == 0 )) || {
  die "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be divisible by total RL GPUs ($TOTAL_RL_GPUS)"
}
[[ "$ROLLOUT_N" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUT_N must be a positive integer, got $ROLLOUT_N"
[[ "$PPO_MINI_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  die "PPO_MINI_BATCH_SIZE must be a positive integer, got $PPO_MINI_BATCH_SIZE"
}
PPO_SCALED_BATCH="$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))"
(( PPO_SCALED_BATCH % TOTAL_RL_GPUS == 0 )) || {
  die "PPO_MINI_BATCH_SIZE * ROLLOUT_N ($PPO_SCALED_BATCH) must be divisible by total RL GPUs ($TOTAL_RL_GPUS)"
}
PPO_MINI_BATCH_PER_GPU="$((PPO_SCALED_BATCH / TOTAL_RL_GPUS))"
TRAJECTORIES_PER_GPU="$((TRAIN_BATCH_SIZE * ROLLOUT_N / TOTAL_RL_GPUS))"
(( TRAJECTORIES_PER_GPU % PPO_MINI_BATCH_PER_GPU == 0 )) || {
  die "per-GPU trajectories ($TRAJECTORIES_PER_GPU) must be divisible by normalized PPO mini batch ($PPO_MINI_BATCH_PER_GPU)"
}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$SMOKE_DATA_DIR" "$SYNC_DIR" \
  "$RAY_TEMP_DIR" "$RAY_SPILL_DIR" \
  "$BASE/cache/huggingface" "$BASE/cache/torch" "$BASE/cache/xdg"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID-node${NODE_RANK}.log}"
TOOL_LOG_FILE="${TOOL_LOG_FILE:-$LOG_DIR/$RUN_ID-node${NODE_RANK}-tools.log}"
touch "$LOG_FILE" "$TOOL_LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

export PATH="$RL_ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$RL_ENV_DIR/lib:$CUDA_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
export CC="${CC:-$RL_ENV_DIR/bin/gcc}"
export CXX="${CXX:-$RL_ENV_DIR/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
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
export RAY_ADDRESS="$MASTER_IP:$RAY_PORT"

cleanup() {
  local status=$?
  if [[ "$NODE_RANK" == "0" ]]; then
    printf '%s\n' "$status" > "$DONE_FILE" 2>/dev/null || true
  fi
  for tool_pid in "${TOOL_PIDS[@]:-}"; do
    if [[ -n "$tool_pid" ]] && kill -0 "$tool_pid" 2>/dev/null; then
      kill "$tool_pid" 2>/dev/null || true
      wait "$tool_pid" 2>/dev/null || true
    fi
  done
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "Visual-agent VLM RL ${NNODES}-node run"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node rank / IP: $NODE_RANK / $NODE_IP"
echo "Platform ids: MA_NODE_RANK=${MA_NODE_RANK:-unset} VC_TASK_INDEX=${VC_TASK_INDEX:-unset} RANK=${RANK:-unset}"
echo "Ray head: $MASTER_IP:$RAY_PORT"
echo "Ray temp directory: $RAY_TEMP_DIR"
echo "Ray spill directory: $RAY_SPILL_DIR"
echo "Repository: $REPO_ROOT"
echo "RL environment: $RL_ENV_DIR"
echo "Tool environment: $TOOL_ENV_DIR"
echo "Initial checkpoint: $MODEL_PATH"
echo "RL GPUs per node: $RL_CUDA_VISIBLE_DEVICES ($N_GPUS_PER_NODE)"
echo "Total RL GPUs: $TOTAL_RL_GPUS"
echo "Local tool GPUs: $TOOL_CUDA_VISIBLE_DEVICES ($TOOL_GPU_COUNT)"
echo "Local tool APIs: $LOCAL_VISUAL_TOOL_API_BASES"
echo "Rollout tool APIs: $VISUAL_TOOL_API_BASES"
echo "Tool servers per node: $VISUAL_TOOL_SERVERS_PER_NODE"
echo "Tool replicas per server (SAM3 / GroundingDINO): $SAM3_REPLICAS / $GROUNDING_DINO_REPLICAS"
echo "Train batch / rollout n: $TRAIN_BATCH_SIZE / $ROLLOUT_N"
echo "PPO mini batch: $PPO_MINI_BATCH_SIZE"
echo "Max concurrent trajectories: $MAX_CONCURRENT_REQUESTS"
echo "Max tokens per model turn: ${MAX_TOKENS_PER_TURN:-rollout default}"
echo "Training steps: $TOTAL_TRAINING_STEPS"
echo "Save portable HuggingFace model: $SAVE_HF_MODEL"
echo "Resume mode / path: $RESUME_MODE / ${RESUME_FROM_PATH:-automatic-or-none}"
echo "Warm-start data / logical step: ${WARM_START_DATA_PATH:-disabled} / $WARM_START_GLOBAL_STEP"
echo "Worker wait timeout: $WORKER_WAIT_TIMEOUT seconds"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Rollout traces: ${ROLLOUT_DATA_DIR:-disabled}"
echo "============================================================"

echo "Environment fingerprint before importing the RL stack"
echo "Host: ${HOSTNAME:-unknown}"
echo "Python: $RL_PYTHON"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_LIBRARY_DIR: $CUDA_LIBRARY_DIR"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi: not found"
fi

"$RL_PYTHON" - <<'PYCHECK'
import os
import sys
import torch

print("python:", sys.version.replace("\n", " "), flush=True)
print(
    "torch:", torch.__version__,
    "torch_cuda:", torch.version.cuda,
    "cuda_available:", torch.cuda.is_available(),
    "device_count:", torch.cuda.device_count(),
    flush=True,
)
if torch.cuda.is_available():
    print("device_0:", torch.cuda.get_device_name(0), flush=True)

import ray
import transformers
import verl
import vllm
print("transformers:", transformers.__version__, flush=True)
print("vllm:", vllm.__version__, "ray:", ray.__version__, "verl:", verl.__version__, flush=True)
PYCHECK

# Keep the formal run's CUDA/driver initialization identical to the successful
# smoke path. Only after torch, vLLM, Ray and VERL have imported successfully do
# we validate the optional external judge API.
if [[ "${JUDGE_ENABLED:-0}" == "1" ]]; then
  if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
    echo "DeepSeek egress proxy: configured"
  else
    echo "DeepSeek egress proxy: not configured"
  fi
  "$RL_PYTHON" - <<'PYJUDGECHECK'
import os
from openai import OpenAI

base_url = os.environ["LLM_AS_A_JUDGE_BASE"].rstrip("/")
if not base_url.endswith("/v1"):
    base_url += "/v1"

client = OpenAI(
    api_key=os.environ["LLM_AS_A_JUDGE_KEY"],
    base_url=base_url,
    timeout=float(os.environ.get("LLM_AS_A_JUDGE_TIMEOUT", "20")),
    max_retries=0,
)
if os.environ.get("LLM_AS_A_JUDGE_PREFLIGHT", "1") == "1":
    response = client.chat.completions.create(
        model=os.environ["LLM_AS_A_JUDGE_MODEL"],
        messages=[{"role": "user", "content": "Return exactly TRUE."}],
        temperature=0.0,
        max_tokens=4,
        extra_body={"thinking": {"type": "disabled"}},
    )
    verdict = (response.choices[0].message.content or "").strip().upper()
    if not verdict.startswith("TRUE"):
        raise SystemExit(f"DeepSeek judge preflight returned an unexpected response: {verdict!r}")
    print("DeepSeek API judge preflight passed", flush=True)
else:
    print("DeepSeek API judge preflight skipped", flush=True)
PYJUDGECHECK
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  START_VISUAL_TOOL_SERVER=0
fi

if [[ "$START_VISUAL_TOOL_SERVER" == "1" ]]; then
  echo "Starting $VISUAL_TOOL_SERVERS_PER_NODE local visual-tool servers on physical GPUs $TOOL_CUDA_VISIBLE_DEVICES"
  for server_index in "${!LOCAL_VISUAL_TOOL_ENDPOINT_ARRAY[@]}"; do
    server_port="$((VISUAL_TOOL_PORT + server_index))"
    server_tool_gpu="${TOOL_GPU_ARRAY[$((server_index % TOOL_GPU_COUNT))]}"
    server_log="${TOOL_LOG_FILE%.log}-server${server_index}.log"
    env \
      CUDA_VISIBLE_DEVICES="$server_tool_gpu" \
      CUDA_HOME="$TOOL_CUDA_HOME" \
      PATH="$TOOL_ENV_DIR/bin:$TOOL_CUDA_HOME/bin:$PATH" \
      LD_LIBRARY_PATH="$TOOL_ENV_DIR/lib:$TOOL_CUDA_LIBRARY_DIR:${LD_LIBRARY_PATH:-}" \
      SAM3_MODEL_PATH="$SAM3_MODEL_PATH" \
      SAM3_DEVICE=cuda:0 \
      SAM3_REPLICAS="$SAM3_REPLICAS" \
      GROUNDING_DINO_MODEL_PATH="$GROUNDING_DINO_MODEL_PATH" \
      GROUNDING_DINO_DEVICE=cuda:0 \
      GROUNDING_DINO_REPLICAS="$GROUNDING_DINO_REPLICAS" \
      VISUAL_TOOL_BACKEND=all \
      "$TOOL_PYTHON" "$REPO_ROOT/scripts/visual_tool_server.py" \
        --backend all --host "$VISUAL_TOOL_HOST" --port "$server_port" \
        >>"$server_log" 2>&1 &
    TOOL_PIDS+=("$!")
    echo "Started visual-tool server $server_index on GPU $server_tool_gpu, port $server_port; log: $server_log"
  done

  "$RL_PYTHON" - "$LOCAL_VISUAL_TOOL_API_BASES" "$SAM3_REPLICAS" "$GROUNDING_DINO_REPLICAS" "$VISUAL_TOOL_STARTUP_TIMEOUT" <<'PYHEALTH'
import json
import sys
import time
import urllib.request

base_urls = [item.strip().rstrip("/") for item in sys.argv[1].split(",") if item.strip()]
expected_sam3, expected_grounding = map(int, sys.argv[2:4])
timeout = int(sys.argv[4])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
pending = set(base_urls)
last_errors = {}
deadline = time.time() + timeout
while time.time() < deadline:
    for base in tuple(pending):
        try:
            with opener.open(base + "/health", timeout=3) as response:
                payload = json.load(response)
            if (
                payload.get("sam3_loaded")
                and payload.get("grounding_dino_loaded")
                and payload.get("sam3_replicas") == expected_sam3
                and payload.get("grounding_dino_replicas") == expected_grounding
            ):
                print("Local visual-tool service ready:", base, payload)
                pending.remove(base)
        except Exception as exc:
            last_errors[base] = str(exc)
    if not pending:
        break
    time.sleep(1)
else:
    raise SystemExit(f"visual-tool services did not become ready: {sorted(pending)}; errors={last_errors}")
PYHEALTH
elif [[ "$START_VISUAL_TOOL_SERVER" == "0" && "${DRY_RUN:-0}" != "1" ]]; then
  "$RL_PYTHON" - "$LOCAL_VISUAL_TOOL_API_BASES" <<'PYHEALTH'
import json
import sys
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for base_url in [item.strip().rstrip("/") for item in sys.argv[1].split(",") if item.strip()]:
    with opener.open(base_url + "/health", timeout=10) as response:
        payload = json.load(response)
    assert payload.get("sam3_loaded") and payload.get("grounding_dino_loaded"), payload
    print("Using existing local visual-tool service:", base_url, payload)
PYHEALTH
elif [[ "$START_VISUAL_TOOL_SERVER" != "0" ]]; then
  die "START_VISUAL_TOOL_SERVER must be 0 or 1"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
# Ray must only see GPUs 0-6. GPU 7 remains isolated for the local tool service.
export CUDA_VISIBLE_DEVICES="$RL_CUDA_VISIBLE_DEVICES"
"$RAY_BIN" stop --force >/dev/null 2>&1 || true

if [[ "$NODE_RANK" == "0" ]]; then
  rm -f "$DONE_FILE"
  "$RAY_BIN" start --head \
    --node-ip-address="$NODE_IP" \
    --port="$RAY_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --include-dashboard=false \
    --num-gpus="$N_GPUS_PER_NODE" \
    --temp-dir="$RAY_TEMP_DIR" \
    --object-spilling-directory="$RAY_SPILL_DIR" \
    --disable-usage-stats
else
  JOINED=0
  for ((attempt=1; attempt<=120; attempt++)); do
    if "$RAY_BIN" start \
      --address="$MASTER_IP:$RAY_PORT" \
      --node-ip-address="$NODE_IP" \
      --num-gpus="$N_GPUS_PER_NODE" \
      --temp-dir="$RAY_TEMP_DIR" \
      --object-spilling-directory="$RAY_SPILL_DIR" \
      --disable-usage-stats; then
      JOINED=1
      break
    fi
    echo "Waiting for Ray head, attempt $attempt/120"
    sleep 2
  done
  [[ "$JOINED" == "1" ]] || die "unable to join Ray head at $MASTER_IP:$RAY_PORT"

  echo "Ray worker joined; waiting for the driver to finish"
  waited=0
  while [[ ! -f "$DONE_FILE" ]]; do
    sleep 5
    waited=$((waited + 5))
    (( waited < WORKER_WAIT_TIMEOUT )) || die "timed out waiting for node-0 driver"
  done
  DRIVER_STATUS="$(sed -n '1p' "$DONE_FILE")"
  echo "Node-0 driver finished with status ${DRIVER_STATUS:-unknown}"
  [[ "$DRIVER_STATUS" == "0" ]]
  exit $?
fi

# Node 0 waits until every seven-GPU Ray node is visible.
"$RL_PYTHON" - "$RAY_ADDRESS" "$NNODES" "$TOTAL_RL_GPUS" "$RAY_CLUSTER_TIMEOUT" <<'PYRAYWAIT'
import sys
import time
import ray

address, expected_nodes, expected_gpus, timeout = sys.argv[1:]
expected_nodes, expected_gpus, timeout = int(expected_nodes), int(expected_gpus), int(timeout)
ray.init(address=address, ignore_reinit_error=True)
deadline = time.time() + timeout
while time.time() < deadline:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    resources = ray.cluster_resources()
    gpus = int(resources.get("GPU", 0))
    print(f"Ray cluster: alive_nodes={len(nodes)}/{expected_nodes}, GPUs={gpus}/{expected_gpus}")
    if len(nodes) >= expected_nodes and gpus >= expected_gpus:
        break
    time.sleep(5)
else:
    raise SystemExit("Ray cluster did not reach the requested resources")
ray.shutdown()
PYRAYWAIT

# By this point every worker has passed its local health check. Confirm from
# the node-0 driver that cross-node routing is also reachable.
"$RL_PYTHON" - "$VISUAL_TOOL_API_BASES" "$SAM3_REPLICAS" "$GROUNDING_DINO_REPLICAS" <<'PYREMOTEHEALTH'
import json
import sys
import urllib.request

base_urls = [item.strip().rstrip("/") for item in sys.argv[1].split(",") if item.strip()]
expected_sam3, expected_grounding = map(int, sys.argv[2:])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for base_url in base_urls:
    with opener.open(base_url + "/health", timeout=15) as response:
        payload = json.load(response)
    assert payload.get("sam3_replicas") == expected_sam3, (base_url, payload)
    assert payload.get("grounding_dino_replicas") == expected_grounding, (base_url, payload)
    print("Driver can reach visual-tool endpoint:", base_url, payload)
PYREMOTEHEALTH

else
  echo "DRY_RUN=1: skipping Ray cluster startup and multi-node rendezvous"
  export CUDA_VISIBLE_DEVICES="$RL_CUDA_VISIBLE_DEVICES"
fi

if [[ "$PREPARE_SMOKE_DATA" == "1" ]]; then
  PREPARE_ARGS=(
    --input "$REPO_ROOT/data/visual_tool_sft_v0.smoke.jsonl"
    --output-dir "$SMOKE_DATA_DIR"
    --train-samples "$TRAIN_BATCH_SIZE"
    --val-samples "$VAL_BATCH_SIZE"
  )
  "$RL_PYTHON" "$REPO_ROOT/scripts/prepare_visual_tool_rl_smoke.py" "${PREPARE_ARGS[@]}"
elif [[ "$PREPARE_SMOKE_DATA" != "0" ]]; then
  die "PREPARE_SMOKE_DATA must be 0 or 1"
fi

"$RL_PYTHON" - "$TRAIN_FILES" "$VAL_FILES" <<'PYDATAFILES'
from pathlib import Path
import sys
for label, value in zip(("train", "val"), sys.argv[1:]):
    files = [Path(item) for item in value.split(",") if item]
    if not files:
        raise SystemExit(f"{label} file list is empty")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing {label} parquet files: {missing[:10]}")
    print(f"{label} parquet files: {len(files)}")
PYDATAFILES

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
  "data.train_files=[$TRAIN_FILES]"
  "data.val_files=[$VAL_FILES]"
  "data.prompt_key=$DATA_PROMPT_KEY"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.val_batch_size=$VAL_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.return_raw_chat=True"
  "data.shuffle=$TRAIN_SHUFFLE"
  "data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS"
  "data.filter_overlong_prompts_workers=$FILTER_OVERLONG_WORKERS"
  "+data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  "data.truncation=error"
  "data.image_key=images"
  "data.trust_remote_code=True"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
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
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  "actor_rollout_ref.rollout.agent.activate_agent=False"
  "actor_rollout_ref.rollout.multi_turn.enable=True"
  "actor_rollout_ref.rollout.multi_turn.max_turns=$MAX_TURNS"
  "+actor_rollout_ref.rollout.multi_turn.max_concurrent_requests=$MAX_CONCURRENT_REQUESTS"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH"
  "+actor_rollout_ref.rollout.multi_turn.format=$TOOL_CALL_FORMAT"
  "actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True"
  "actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "reward_model.reward_manager=naive_async"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.nnodes=$NNODES"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.project_name=$TRAINER_PROJECT_NAME"
  "trainer.experiment_name=$RUN_ID"
  "trainer.default_local_dir=$OUTPUT_DIR"
)
if [[ "$SAVE_HF_MODEL" == "1" ]]; then
  TRAIN_ARGS+=("actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra']")
fi
if [[ "$RESUME_MODE" == "resume_path" ]]; then
  TRAIN_ARGS+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi
if [[ -n "$WARM_START_DATA_PATH" || "$WARM_START_GLOBAL_STEP" != "0" ]]; then
  TRAIN_ARGS+=(
    "+trainer.warm_start_data_path=$WARM_START_DATA_PATH"
    "+trainer.warm_start_global_step=$WARM_START_GLOBAL_STEP"
  )
fi
if [[ -n "$MAX_TOKENS_PER_TURN" ]]; then
  [[ "$MAX_TOKENS_PER_TURN" =~ ^[1-9][0-9]*$ ]] || {
    die "MAX_TOKENS_PER_TURN must be a positive integer, got $MAX_TOKENS_PER_TURN"
  }
  TRAIN_ARGS+=("+actor_rollout_ref.rollout.multi_turn.max_tokens_per_turn=$MAX_TOKENS_PER_TURN")
fi
if [[ -n "$ROLLOUT_DATA_DIR" ]]; then
  TRAIN_ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [[ -n "$CUSTOM_DATASET_PATH" || -n "$CUSTOM_DATASET_NAME" ]]; then
  [[ -n "$CUSTOM_DATASET_PATH" && -n "$CUSTOM_DATASET_NAME" ]] || {
    die "CUSTOM_DATASET_PATH and CUSTOM_DATASET_NAME must be set together"
  }
  [[ -f "$CUSTOM_DATASET_PATH" ]] || die "custom dataset file not found: $CUSTOM_DATASET_PATH"
  TRAIN_ARGS+=(
    "data.custom_cls.path=$CUSTOM_DATASET_PATH"
    "data.custom_cls.name=$CUSTOM_DATASET_NAME"
  )
fi

set -x
"$RL_PYTHON" -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "${HYDRA_ARGS[@]}"
status=$?
set +x
echo "$NNODES-node RL run finished with exit code $status at $(date '+%Y-%m-%d %H:%M:%S')"
exit "$status"
