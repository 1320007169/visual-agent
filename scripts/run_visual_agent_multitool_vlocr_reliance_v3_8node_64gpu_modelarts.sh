#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare the source-matched experiment using the vanilla 64-GPU training setup.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
case "${V3_VARIANT:-paired}" in
    paired) data_name=vlocr_reliance_pairs_v3_fixed_20261008 ;;
    factual) data_name=vlocr_reliance_factual_v3_fixed_20261008 ;;
    *) echo "error: V3_VARIANT must be paired or factual" >&2; exit 2 ;;
esac
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/$data_name}"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet" VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export MODEL_PATH="$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_reliance_v3_${V3_VARIANT:-paired}_n16_8node}"
export TEST_FREQ="${TEST_FREQ:-5}" SAVE_FREQ="${SAVE_FREQ:-5}"
python3 - "$MULTITOOL_DATA_DIR/manifest.json" "${V3_VARIANT:-paired}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
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
if manifest["factual_only"] != (sys.argv[2] == "factual"):
    raise SystemExit("Dataset branch does not match V3_VARIANT")
print(f"V3 data ready: rows={manifest['train_rows']}, pairs={manifest['pairs']}")
PY
exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_vlocr_tallyhalf_fsc3000_8node_64gpu_modelarts.sh" "$@"
