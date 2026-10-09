#!/usr/bin/env bash
set -Eeuo pipefail

# Re-evaluate the new 64-GPU step80 checkpoint on four benchmarks with one 8-GPU node.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
[[ "${NNODES:-1}" == 1 ]] || { echo "error: submit evaluation on one 8-GPU node" >&2; exit 2; }
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" != 1 ]]; then
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
fi

export VLOCR_STEP=80
export VLOCR_MODEL_PATH="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_80/actor/huggingface"
launcher="$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh"
[[ -f "$launcher" ]] || { echo "error: repository launcher is missing: $launcher" >&2; exit 2; }
python3 - "$VLOCR_MODEL_PATH" <<'PY'
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
json.loads((model / "config.json").read_text())
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
    raise SystemExit(f"Missing or empty HF weight shards: {model}")
print(f"Checkpoint ready: {model}; shards={len(shards)}", flush=True)
PY

unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY VLMEVAL_EVAL_ID MME_PROMPT_VARIANT COUNT_SERVICE_CONFIG
export VISUAL_AGENT_OCR_RAW_BACKSLASH=0 VLOCR_NATIVE_TOOLS=1
export VISUAL_AGENT_NORMALIZE_ANSWER_BACKSLASH=1 VISUAL_AGENT_FAILURE_REPORT=1
export GPU_MEMORY_UTILIZATION=0.80 VLMEVAL_API_NPROC=7
export MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOOL_CUDA_VISIBLE_DEVICES=7
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export EVAL_DATASETS="HRBench4K HRBench8K MME-RealWorld-Lite MME-RealWorld-CN"
group_id="${RUN_ID:-vlocr_new64_step80_four_bench_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/vlocr_new64_step80_four_bench_8gpu/$group_id}"
printf 'Results: %s\nDatasets: %s\nMax model length: %s\n' "$group_root" "$EVAL_DATASETS" "$MAX_MODEL_LEN"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "$gpu_count" -ge 8 ]] || { echo "error: this evaluation requires 8 visible GPUs; found $gpu_count" >&2; exit 2; }

mkdir -p "$(dirname "$group_root")"
mkdir "$group_root"
printf 'checkpoint\tstep\tmodel_path\n64gpu\t80\t%s\n' "$VLOCR_MODEL_PATH" > "$group_root/checkpoints.tsv"
printf 'checkpoint\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
export RUN_ID="${group_id}_64gpu_step80" WORK_ROOT="$group_root/64gpu_step80"
export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
started=$SECONDS
status=0
bash "$launcher" "$@" || status=$?
printf '64gpu_step80\t%s\t%s\t%s\n' "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >> "$group_root/status.tsv"
exit "$status"
