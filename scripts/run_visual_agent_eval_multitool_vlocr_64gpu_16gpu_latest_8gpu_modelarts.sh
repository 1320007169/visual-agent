#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the latest complete checkpoints from both active VL-OCR training runs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export BASE="${BASE:-$(dirname "$REPO_ROOT")}"
export CUDA_HOME="$BASE/conda_envs/spacetools-rl"
export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
[[ "${NNODES:-1}" == 1 ]] || { echo "error: submit evaluation on one 8-GPU node" >&2; exit 2; }

labels=(64gpu 16gpu)
runs=(
    "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c"
    "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67"
)
steps=()
models=()
for run in "${runs[@]}"; do
    step="$(cat "$run/latest_checkpointed_iteration.txt")"
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid saved checkpoint step in $run" >&2; exit 2; }
    steps+=("$step")
    models+=("$run/global_step_$step/actor/huggingface")
done

# Check every HF shard before creating output or starting any GPU service.
python3 - "${models[@]}" <<'PY'
import json
from pathlib import Path
import sys

for value in sys.argv[1:]:
    model = Path(value)
    json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards or any(not (model / name).is_file() or (model / name).stat().st_size == 0 for name in shards):
        raise SystemExit(f"Missing or empty HF weight shards: {model}")
    print(f"Checkpoint ready: {model}; shards={len(shards)}", flush=True)
PY

group_run_id="${RUN_ID:-multitool_vlocr_64gpu_16gpu_latest_$(date +%Y%m%dT%H%M%S%N)_$$}"
group_root="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/multitool_vlocr_64gpu_16gpu_latest_8gpu/$group_run_id}"
export LOG_DIR="${LOG_DIR:-$BASE/logs/visual-agent-eval}"
for index in "${!labels[@]}"; do
    printf '%s_step%s: %s\n' "${labels[$index]}" "${steps[$index]}" "${models[$index]}"
done
printf 'Results: %s\n' "$group_root"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

if [[ "${EVAL_PREFLIGHT_ONLY:-0}" != 1 ]]; then
    mkdir -p "$(dirname "$group_root")"
    mkdir "$group_root"
    mkdir "$group_root/checkpoints"
    printf 'checkpoint\tstep\tsource_model\teval_model\n' > "$group_root/checkpoints.tsv"
    # Hard links keep selected weights available when training rotates checkpoints.
    # WORK_ROOT must use the same filesystem as the training model directories.
    for index in "${!labels[@]}"; do
        snapshot="$group_root/checkpoints/${labels[$index]}_step${steps[$index]}"
        cp -al "${models[$index]}" "$snapshot"
        printf '%s\t%s\t%s\t%s\n' "${labels[$index]}" "${steps[$index]}" "${models[$index]}" "$snapshot" >> "$group_root/checkpoints.tsv"
        models[$index]="$snapshot"
    done
    printf 'checkpoint\texit_code\tseconds\tresult_dir\n' > "$group_root/status.tsv"
fi

trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for index in "${!labels[@]}"; do
    label="${labels[$index]}_step${steps[$index]}"
    export RUN_ID="${group_run_id}_${label}"
    export WORK_ROOT="$group_root/$label"
    export VLOCR_STEP="${steps[$index]}" VLOCR_MODEL_PATH="${models[$index]}"
    started=$SECONDS
    status=0
    bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh" "$@" || status=$?
    if [[ "${EVAL_PREFLIGHT_ONLY:-0}" != 1 ]]; then
        printf '%s\t%s\t%s\t%s\n' "$label" "$status" "$((SECONDS - started))" "$WORK_ROOT/dino_latest" >> "$group_root/status.tsv"
    fi
    if (( status != 0 )); then
        printf 'Evaluation failed: %s (exit=%s)\n' "$label" "$status" >&2
        failed=1
    fi
done
exit "$failed"
