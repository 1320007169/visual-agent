#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate this fixed checkpoint pair sequentially on one ModelArts 8-GPU node.
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

labels=(64gpu 16gpu)
steps=(53 210)
models=(
    "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_53/actor/huggingface"
    "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67/global_step_210/actor/huggingface"
)
launcher="$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh"
[[ -f "$launcher" ]] || { echo "error: repository launcher is missing: $launcher" >&2; exit 2; }
python3 - "${models[@]}" <<'PY'
import json
from pathlib import Path
import sys

for filename in sys.argv[1:]:
    model = Path(filename)
    json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
        raise SystemExit(f"Missing or empty HF weight shards: {model}")
    print(f"Checkpoint ready: {model}; shards={len(shards)}", flush=True)
PY

# Keep both runs on the existing full ten-benchmark evaluation protocol.
unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY VLMEVAL_EVAL_ID
export VISUAL_AGENT_OCR_RAW_BACKSLASH=0 VLOCR_NATIVE_TOOLS=1
export VISUAL_AGENT_NORMALIZE_ANSWER_BACKSLASH=1
export GPU_MEMORY_UTILIZATION=0.80 VLMEVAL_API_NPROC=7
export MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOOL_CUDA_VISIBLE_DEVICES=7
export EVAL_DATASETS="VStarBench HRBench8K OCRBench MME-RealWorld-Lite HRBench4K MME-RealWorld-CN CV-Bench-2D CV-Bench-3D ChartQA_TEST FSC147_TEST"
group_id="${RUN_ID:-vlocr_64gpu_step53_16gpu_step210_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/vlocr_64gpu_step53_16gpu_step210_8gpu/$group_id}"
printf 'Results: %s\nDatasets: %s\n' "$group_root" "$EVAL_DATASETS"
printf 'Final-answer backslash normalization: enabled; original responses remain in traces\n'
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p "$(dirname "$group_root")"
mkdir "$group_root"
printf 'checkpoint\tstep\tmodel_path\n' > "$group_root/checkpoints.tsv"
for index in "${!labels[@]}"; do
    printf '%s\t%s\t%s\n' "${labels[$index]}" "${steps[$index]}" "${models[$index]}" >> "$group_root/checkpoints.tsv"
done
printf 'checkpoint\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for index in "${!labels[@]}"; do
    label="${labels[$index]}_step${steps[$index]}"
    export RUN_ID="${group_id}_${label}" WORK_ROOT="$group_root/$label"
    export VLOCR_STEP="${steps[$index]}" VLOCR_MODEL_PATH="${models[$index]}"
    export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
    printf 'Evaluating %s\nFailure traces: %s\n' "$label" "$VISUAL_AGENT_FAILURE_TRACE_DIR"
    started=$SECONDS
    status=0
    bash "$launcher" "$@" || status=$?
    printf '%s\t%s\t%s\t%s\n' "$label" "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >> "$group_root/status.tsv"
    if (( status != 0 )); then
        printf 'Evaluation failed: %s (exit=%s)\n' "$label" "$status" >&2
        failed=1
    fi
done
exit "$failed"
