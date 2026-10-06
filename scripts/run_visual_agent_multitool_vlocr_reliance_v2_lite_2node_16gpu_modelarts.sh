#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the counterfactual launcher with the prepared v2-lite data.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/vlocr_reliance_pairs_v2_lite_20261006}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_reliance_v2lite_n16_2node}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-36000}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_vlocr_reliance_2node_16gpu_modelarts.sh" "$@"
