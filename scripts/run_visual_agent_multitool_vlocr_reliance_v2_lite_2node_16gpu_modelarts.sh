#!/usr/bin/env bash
set -Eeuo pipefail

# Train v2-lite from base weights, or resume a complete v2-lite checkpoint.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
export MODEL_PATH="${MODEL_PATH:-$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
export MULTITOOL_DATA_DIR="${MULTITOOL_DATA_DIR:-$REPO_ROOT/data/vlocr_reliance_pairs_v2_lite_20261006}"
export TRAIN_FILES="$MULTITOOL_DATA_DIR/train.parquet"
export VAL_FILES="$MULTITOOL_DATA_DIR/val.parquet"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_reliance_v2lite_n16_2node}"

export NNODES=2
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-126}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-42}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-126}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-112}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_SHUFFLE=True
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TRAINER_STOP_AFTER_SECONDS="${TRAINER_STOP_AFTER_SECONDS:-0}"

export RL_SPLIT_OCR=0
export RL_CHART_PARSE=1
export VTS_VL_OCR=1
export TOOL_CONFIG_PATH="$REPO_ROOT/reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml"
export VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://api-cn.hi-code.cc/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-deepseek-v4.1-flash}"
export JUDGE_KEY_FILE="${JUDGE_KEY_FILE:-$BASE/secrets/hicode_judge_api_key.txt}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" != "1" ]]; then
    [[ -s "$MULTITOOL_DATA_DIR/manifest.json" ]] || {
        echo "error: prepared v2-lite data is missing: $MULTITOOL_DATA_DIR" >&2
        exit 2
    }
fi

export RESUME_MODE=disable
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH%/}"
if [[ -n "$RESUME_FROM_PATH" ]]; then
    [[ "$RESUME_FROM_PATH" == "$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/${RUN_ID}_"*/global_step_* ]] || {
        echo "error: resume checkpoint must belong to the v2-lite run: $RUN_ID" >&2
        exit 2
    }
    [[ -s "$RESUME_FROM_PATH/data.pt" ]] || {
        echo "error: missing resume dataloader state: $RESUME_FROM_PATH/data.pt" >&2
        exit 2
    }
    for ((rank = 0; rank < 14; rank++)); do
        for component in model optim extra_state; do
            checkpoint_file="$RESUME_FROM_PATH/actor/${component}_world_size_14_rank_${rank}.pt"
            [[ -s "$checkpoint_file" ]] || {
                echo "error: missing resume shard: $checkpoint_file" >&2
                exit 2
            }
        done
    done
    # Keep the source checkpoint and allocate fresh job synchronization paths.
    source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"
    export RESUME_MODE=resume_path
fi

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    printf 'RESUME_FROM_PATH=%s\n' "$RESUME_FROM_PATH"
fi

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh" "$@"
