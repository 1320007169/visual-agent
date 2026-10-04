#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate all 1000 OCRBench questions, changing only OCR text backslash escaping.
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

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
[[ "${NNODES:-1}" == 1 ]] || { echo "error: submit on one 8-GPU node" >&2; exit 2; }
export VLOCR_STEP=80
export VLOCR_MODEL_PATH="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c/global_step_80/actor/huggingface"
export EVAL_DATASETS=OCRBench PRED_FORMAT=json
export VLOCR_NATIVE_TOOLS=1 GPU_MEMORY_UTILIZATION=0.80
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-7}"
export EVAL_TIMEOUT_SECONDS=0
unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY
group_run_id="${RUN_ID:-ocrbench_backslash_ab_64gpu_step80_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/ocrbench_backslash_ab_64gpu_step80/$group_run_id}"
source_tsv="$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData/OCRBench.tsv"
python3 - "$VLOCR_MODEL_PATH" "$source_tsv" <<'PY'
import csv
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
json.loads((model / 'config.json').read_text())
index = json.loads((model / 'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
    raise SystemExit(f'Missing or empty HF weight shards: {model}')
csv.field_size_limit(sys.maxsize)
with open(sys.argv[2]) as stream:
    rows = csv.DictReader(stream, delimiter='\t')
    samples = [(int(row['index']), row['category']) for row in rows]
if [index for index, _ in samples] != list(range(1000)):
    raise SystemExit('Expected the full 1000 OCRBench samples, indices 0-999')
if sum(category == 'Handwritten Mathematical Expression Recognition' for _, category in samples) != 100:
    raise SystemExit('Expected 100 HME samples within OCRBench')
print(f'Checkpoint ready: {model}; shards={len(shards)}')
print('OCRBench: all 1000 samples, including 100 HME samples; temperature=0; max_turns=8; max_tokens=512')
PY
printf 'Results: %s\nArms: json_observation, raw_ocr_backslash\nEvaluation time limit: disabled\n' "$group_root"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p "$(dirname "$group_root")"
mkdir "$group_root"
printf 'arm\tstep\tmodel_path\tocr_raw_backslash\n' > "$group_root/checkpoints.tsv"
printf 'arm\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for arm in json_observation raw_ocr_backslash; do
    export VISUAL_AGENT_OCR_RAW_BACKSLASH=0
    if [[ "$arm" == raw_ocr_backslash ]]; then
        export VISUAL_AGENT_OCR_RAW_BACKSLASH=1
    fi
    export RUN_ID="${group_run_id}_${arm}" VLMEVAL_EVAL_ID="${group_run_id}_${arm}"
    export WORK_ROOT="$group_root/$arm"
    export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
    printf '%s\t80\t%s\t%s\n' "$arm" "$VLOCR_MODEL_PATH" "$VISUAL_AGENT_OCR_RAW_BACKSLASH" >> "$group_root/checkpoints.tsv"
    started=$SECONDS
    status=0
    bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh" "$@" || status=$?
    printf '%s\t%s\t%s\t%s\n' "$arm" "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >> "$group_root/status.tsv"
    if (( status != 0 )); then
        failed=1
    fi
done
if (( failed == 0 )); then
    python3 "$REPO_ROOT/scripts/summarize_ocr_hme_ab.py" "$group_root"
fi
exit "$failed"
