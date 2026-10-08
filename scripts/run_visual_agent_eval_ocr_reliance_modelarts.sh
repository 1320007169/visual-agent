#!/usr/bin/env bash
set -Eeuo pipefail

# Run identical OCRBench questions in clean, training-fault, and unseen-fault conditions.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
[[ "${NNODES:-1}" == 1 ]] || { echo "error: submit on one 8-GPU node" >&2; exit 2; }
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" != 1 && "${SKIP_MODELARTS_BOOTSTRAP:-0}" != 1 ]]; then
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

checkpoint_root="$BASE/visual-agent/saves/visual_agent_zwz_rl/qwen3"
v2_run="$checkpoint_root/qwen3base_multitool_vlocr_reliance_v2lite_n16_2node_20261006T141420415217_5464e0b8"
read -r -a labels <<< "${EVAL_CHECKPOINTS:-base vanilla53 v2_step40 v2_step50}"
models=()
steps=()
for label in "${labels[@]}"; do
    case "$label" in
        base) models+=("$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct"); steps+=(0) ;;
        vanilla53)
            models+=("$checkpoint_root/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_53/actor/huggingface")
            steps+=(53)
            ;;
        v2_step30) models+=("${V2_STEP30_MODEL_PATH:-$v2_run/evaluation_snapshots/step30_huggingface}"); steps+=(30) ;;
        v2_step40) models+=("$v2_run/evaluation_snapshots/step40_huggingface"); steps+=(40) ;;
        v2_step45) models+=("$v2_run/evaluation_snapshots/step45_huggingface"); steps+=(45) ;;
        v2_step50) models+=("$v2_run/evaluation_snapshots/step50_huggingface"); steps+=(50) ;;
        *) echo "error: unknown checkpoint label: $label" >&2; exit 2 ;;
    esac
done
python3 - "${models[@]}" <<'PY'
import json
from pathlib import Path
import sys

for filename in sys.argv[1:]:
    model = Path(filename)
    json.loads((model / "config.json").read_text())
    shards = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
        raise SystemExit(f"Missing or empty weight shards: {model}")
    print(f"Checkpoint ready: {model}; shards={len(shards)}")
PY

export EVAL_DATASETS=OCRBench PRED_FORMAT=json VLOCR_NATIVE_TOOLS=1
export VISUAL_AGENT_OCR_RAW_BACKSLASH=0 VISUAL_AGENT_NORMALIZE_ANSWER_BACKSLASH=1
export VISUAL_AGENT_FAULT_SEED="${VISUAL_AGENT_FAULT_SEED:-20261008}"
export MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOOL_CUDA_VISIBLE_DEVICES=7
export GPU_MEMORY_UTILIZATION=0.80 VLMEVAL_API_NPROC=7 EVAL_TIMEOUT_SECONDS=0
unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY
group_id="${RUN_ID:-ocr_reliance_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/ocr_reliance/$group_id}"
printf 'Results: %s\nCheckpoints: %s\nConditions: clean training ocr_confusable\n' "$group_root" "${labels[*]}"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi
mkdir -p "$(dirname "$group_root")"
mkdir "$group_root"
printf 'checkpoint\tstep\tcondition\tmodel_path\tfault_seed\n' > "$group_root/checkpoints.tsv"
printf 'checkpoint\tcondition\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
for index in "${!labels[@]}"; do
    for condition in clean training ocr_confusable; do
        export VISUAL_AGENT_FAULT_TOOLS=ocr_read VISUAL_AGENT_FAULT_TYPE="$condition"
        if [[ "$condition" == clean ]]; then
            export VISUAL_AGENT_FAULT_TOOLS="" VISUAL_AGENT_FAULT_TYPE=training
        fi
        export VLOCR_STEP="${steps[$index]}" VLOCR_MODEL_PATH="${models[$index]}"
        export RUN_ID="${group_id}_${labels[$index]}_${condition}" VLMEVAL_EVAL_ID="${group_id}_${labels[$index]}_${condition}"
        export WORK_ROOT="$group_root/${labels[$index]}/$condition"
        export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
        printf '%s\t%s\t%s\t%s\t%s\n' "${labels[$index]}" "$VLOCR_STEP" "$condition" "$VLOCR_MODEL_PATH" "$VISUAL_AGENT_FAULT_SEED" >> "$group_root/checkpoints.tsv"
        started=$SECONDS
        status=0
        bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh" "$@" || status=$?
        printf '%s\t%s\t%s\t%s\t%s\n' "${labels[$index]}" "$condition" "$status" "$((SECONDS - started))" "$WORK_ROOT" >> "$group_root/status.tsv"
        if (( status != 0 )); then
            exit "$status"
        fi
    done
done
if [[ " ${labels[*]} " == *" base "* ]]; then
    python3 "$REPO_ROOT/scripts/summarize_ocr_reliance_eval.py" "$group_root" --output "$group_root/comparison.json"
fi
