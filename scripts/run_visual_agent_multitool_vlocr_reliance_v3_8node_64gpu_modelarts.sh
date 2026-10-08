#!/usr/bin/env bash
set -Eeuo pipefail

# Run matched P/F/V experiments; each arm resumes with its original topology.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export V3_VARIANT="${V3_VARIANT:-paired}"
case "$V3_VARIANT" in
    paired) data_name=vlocr_reliance_pairs_v3_fixed_20261008 ;;
    factual) data_name=vlocr_reliance_factual_v3_fixed_20261008 ;;
    unprefixed) data_name=vlocr_reliance_unprefixed_v3_fixed_20261008 ;;
    *) echo "error: V3_VARIANT must be paired, factual or unprefixed" >&2; exit 2 ;;
esac
export NNODES="${NNODES:-8}"
case "$NNODES" in
    2) batch_size=126; mini_batch_size=42 ;;
    8) batch_size=336; mini_batch_size=112 ;;
    *) echo "error: NNODES must be 2 or 8" >&2; exit 2 ;;
esac
export TRAIN_BATCH_SIZE="$batch_size" PPO_MINI_BATCH_SIZE="$mini_batch_size" VAL_BATCH_SIZE="$batch_size"
export ROLLOUT_N=16 MAX_CONCURRENT_REQUESTS="$((NNODES * 56))"
export TRAIN_SHUFFLE=True
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-0}"
export RL_SPLIT_OCR=0 RL_CHART_PARSE=1 VTS_VL_OCR=1
export TOOL_CONFIG_PATH="$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/$data_name}"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet" VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export MODEL_PATH="$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_reliance_v3_${V3_VARIANT}_k2_n16_${NNODES}node}"
export TEST_FREQ="${TEST_FREQ:-5}" SAVE_FREQ="${SAVE_FREQ:-5}"
python3 - "$REPO_ROOT" "$MULTITOOL_DATA_DIR" "$V3_VARIANT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repo, data = map(Path, sys.argv[1:3])
variant = {"paired": "P", "factual": "F", "unprefixed": "V"}[sys.argv[3]]
manifest = json.loads((data / "manifest.json").read_text())
provenance = json.loads((repo / "configs/tool_reliance_v3_k2_data_20261008.json").read_text())
expected = provenance["datasets"][variant]
if manifest.get("generation_commit") != provenance["generation_commit"]:
    raise SystemExit("Dataset does not identify the published generation commit")
if manifest.get("generation_code_sha256") != provenance["generation_code_sha256"]:
    raise SystemExit("Dataset generation code differs from the published version")
for filename, digest in provenance["generation_code_sha256"].items():
    if hashlib.sha256((repo / filename).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Generation/replay code differs from the pinned version: {filename}")
for filename, key in (("train.parquet", "train_sha256"), ("val.parquet", "val_sha256")):
    if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected[key]:
        raise SystemExit(f"Dataset content differs from the pinned {variant} data: {filename}")
if manifest.get("dataset_variant") != variant or manifest.get("counterfactual_variants") != 2:
    raise SystemExit("Expected the matched k=2 dataset for this arm")
if manifest.get("replacement_scope") != "same_source" or manifest["train_sources_before"] != manifest["train_sources_after"]:
    raise SystemExit("Expected source-matched v3 training data")
if manifest["replacement_groups_before"] != manifest["replacement_groups_after"]:
    raise SystemExit("Expected unchanged original-source counts, including HME")
if manifest.get("ocr_fault_policy") != "hme_visible_symbols_answer_excluded":
    raise SystemExit("Expected repaired OCR faults with answer exclusion")
if manifest.get("max_replaced_fraction_per_original_source") != 0.5 or any(
    count * 2 > manifest["replacement_groups_before"][source] for source, count in manifest["prefixed_groups"].items()
):
    raise SystemExit("Expected at least half of each original source to remain unprefixed")
print(f"V3_DATA_COMMIT={provenance['generation_commit']}")
print(f"V3_DATA_VARIANT={variant}")
print(f"V3_TRAIN_SHA256={expected['train_sha256']}")
PY

export RESUME_MODE=disable
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH%/}"
if [[ -n "$RESUME_FROM_PATH" ]]; then
    [[ "$RESUME_FROM_PATH" == "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/${RUN_ID}_"*/global_step_* ]] || {
        echo "error: resume checkpoint must belong to this v3 arm and topology: $RUN_ID" >&2
        exit 2
    }
    [[ -s "$RESUME_FROM_PATH/data.pt" ]] || {
        echo "error: missing resume dataloader state: $RESUME_FROM_PATH/data.pt" >&2
        exit 2
    }
    world_size=$((NNODES * 7))
    for ((rank = 0; rank < world_size; rank++)); do
        for component in model optim extra_state; do
            checkpoint_file="$RESUME_FROM_PATH/actor/${component}_world_size_${world_size}_rank_${rank}.pt"
            [[ -s "$checkpoint_file" ]] || {
                echo "error: missing resume shard: $checkpoint_file" >&2
                exit 2
            }
        done
    done
    # Allocate fresh synchronization/output paths before enabling full resume.
    source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
    export RESUME_MODE=resume_path
fi
if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    printf 'RESUME_FROM_PATH=%s\n' "$RESUME_FROM_PATH"
fi
exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh" "$@"
