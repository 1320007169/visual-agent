#!/usr/bin/env bash
set -Eeuo pipefail

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline}"
export COUNT_SERVICE_CONFIG="${COUNT_SERVICE_CONFIG:-$PIPELINE_ROOT/configs/services/countgd_plusplus_pseudo_eval.yaml}"
export NNODES="${NNODES:-2}"
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
# Data lives in the shared checkout, which a branch checkout under REPO_ROOT may not contain.
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$BASE/visual-agent/data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004}"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet" VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export VISUAL_AGENT_ONLINE_FAULTS_CONFIG="$REPO_ROOT/configs/online_crt_v1.json"
export MODEL_PATH="$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_online_crt_n16_${NNODES}node}"
export TEST_FREQ="${TEST_FREQ:-5}" SAVE_FREQ="${SAVE_FREQ:-5}"
[[ -s "$VISUAL_AGENT_ONLINE_FAULTS_CONFIG" ]] || {
    echo "error: missing online fault config: $VISUAL_AGENT_ONLINE_FAULTS_CONFIG" >&2
    exit 2
}
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
export RESUME_MODE=disable
if [[ -n "$RESUME_FROM_PATH" ]]; then
    [[ -s "$RESUME_FROM_PATH/online_faults.json" && -s "$RESUME_FROM_PATH/data.pt" ]] || {
        echo "error: resume checkpoint lacks online fault or dataloader state: $RESUME_FROM_PATH" >&2
        exit 2
    }
    source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
    export RESUME_MODE=resume_path
fi
if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == 1 ]]; then
    printf 'REPO_ROOT=%s\nCURRENT_COMMIT=%s\nVISUAL_AGENT_ONLINE_FAULTS_CONFIG=%s\nRESUME_FROM_PATH=%s\n' \
        "$REPO_ROOT" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$VISUAL_AGENT_ONLINE_FAULTS_CONFIG" "$RESUME_FROM_PATH"
fi
exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh" "$@"
