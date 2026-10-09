#!/usr/bin/env bash
set -Eeuo pipefail

# Compare the two counting backends with the same checkpoint and agent protocol.
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
export PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
[[ "${NNODES:-1}" == 1 ]] || { echo "error: submit on one 8-GPU node" >&2; exit 2; }
export VLOCR_STEP=80
export VLOCR_MODEL_PATH="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_hme_chartqa_tallyhalf_fsc3000_n16_8node_20261003T202135115738_f9c41af7/global_step_80/actor/huggingface"
export EVAL_DATASETS=FSC147_TEST VLOCR_NATIVE_TOOLS=1
export GPU_MEMORY_UTILIZATION=0.80 VLMEVAL_API_NPROC=7
export MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOOL_CUDA_VISIBLE_DEVICES=7
export VISUAL_AGENT_OCR_RAW_BACKSLASH=0 VISUAL_AGENT_NORMALIZE_ANSWER_BACKSLASH=1
export EVAL_TIMEOUT_SECONDS=0 VISUAL_AGENT_FAILURE_REPORT=1
export FSC147_POINT_ANNOTATION_FILE="$BASE/datasets/fsc147/annotation_FSC147_384.json"
unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY VLMEVAL_EVAL_ID VISUAL_AGENT_FAULT_TOOLS
group_id="${RUN_ID:-fsc_count_ab_step80_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/fsc_count_ab_8gpu/$group_id}"

python3 - "$VLOCR_MODEL_PATH" "$FSC147_POINT_ANNOTATION_FILE" "$BASE/datasets/fsc147/Train_Test_Val_FSC_147.json" <<'PY'
import json
from pathlib import Path
import sys

model, annotation, split = map(Path, sys.argv[1:])
json.loads((model / 'config.json').read_text())
shards = set(json.loads((model / 'model.safetensors.index.json').read_text())['weight_map'].values())
if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
    raise SystemExit(f'Missing or empty checkpoint shards: {model}')
points = json.loads(annotation.read_text())
test = json.loads(split.read_text())['test']
if len(test) != 1190 or len(set(test)) != 1190 or any(not points[name]['points'] for name in test):
    raise SystemExit('Expected 1190 official FSC test images with point annotations')
print(f'Checkpoint ready: {model}; shards={len(shards)}')
print('FSC147_TEST: 1190 images; official point labels; count-only observations; unchanged prompt')
PY
for config in countgd_plusplus.yaml countgd_plusplus_pseudo_eval.yaml; do
    [[ -s "$PIPELINE_ROOT/configs/services/$config" ]] || { echo "error: missing count config: $config" >&2; exit 2; }
done
printf 'Results: %s\nArms: text_only, auto_exemplar\nModel GPUs: %s; tool GPU: %s\n' \
    "$group_root" "$MODEL_CUDA_VISIBLE_DEVICES" "$TOOL_CUDA_VISIBLE_DEVICES"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p "$(dirname "$group_root")"
mkdir "$group_root"
python3 - "$group_root" "$REPO_ROOT" "$VLOCR_MODEL_PATH" "$FSC147_POINT_ANNOTATION_FILE" "$PIPELINE_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output, repo, model, annotation, pipeline = map(Path, sys.argv[1:])
files = [repo / 'prompts/visual_agent_rl_system_multitool_vlocr.txt',
         repo / 'reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml',
         model / 'config.json', model / 'model.safetensors.index.json', annotation,
         pipeline / 'configs/services/countgd_plusplus.yaml',
         pipeline / 'configs/services/countgd_plusplus_pseudo_eval.yaml']
protocol = {
    'checkpoint': str(model), 'dataset': 'FSC147_TEST', 'samples_per_arm': 1190,
    'changed_variable': 'count backend only', 'arms': ['text_only', 'auto_exemplar'],
    'observation': 'count only', 'max_turns': 8, 'max_tokens': 512, 'temperature': 0,
    'native_tools': True, 'api_concurrency': 7, 'model_gpus': [0, 1, 2, 3, 4, 5, 6], 'tool_gpu': 7,
    'gpu_memory_utilization': 0.80, 'adaptive_cropping': False,
    'fresh_inference': True, 'training_changes': False,
    'timing': 'End-to-end seconds including data verification, service startup, inference, scoring and cleanup',
    'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
PY
printf 'arm\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
trap 'exit 130' INT
trap 'exit 143' TERM
for arm in text_only auto_exemplar; do
    export COUNT_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/countgd_plusplus.yaml"
    if [[ "$arm" == auto_exemplar ]]; then
        export COUNT_SERVICE_CONFIG="$PIPELINE_ROOT/configs/services/countgd_plusplus_pseudo_eval.yaml"
    fi
    export RUN_ID="${group_id}_${arm}" VLMEVAL_EVAL_ID="${group_id}_${arm}"
    export WORK_ROOT="$group_root/$arm"
    export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
    started=$SECONDS
    status=0
    bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh" "$@" || status=$?
    printf '%s\t%s\t%s\t%s\n' "$arm" "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >> "$group_root/status.tsv"
    if (( status != 0 )); then
        exit "$status"
    fi
done
eval_python="${ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM}/bin/python3"
"$eval_python" "$REPO_ROOT/scripts/summarize_fsc_count_ab.py" "$group_root"
