#!/usr/bin/env bash
set -Eeuo pipefail

export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent-online-crt}"
export NNODES=2 ALLOW_FSDP_WORLD_SIZE_CHANGE=True DATALOADER_NUM_WORKERS=2
# Preserve the source batch boundaries and optimizer schedule when reducing ranks.
export TRAIN_BATCH_SIZE=336 PPO_MINI_BATCH_SIZE=112 VAL_BATCH_SIZE=336
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/qwen3base_multitool_vlocr_online_crt_n16_8node_20261009T194559026966_2bf73edd/global_step_4}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_online_crt_n16_2node_resume_step4}"
export TEST_FREQ=40 SAVE_FREQ=1
export MAX_ACTOR_CKPT_TO_KEEP='' MAX_CHECKPOINTS_TO_KEEP=''
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_vlocr_online_crt_modelarts.sh" "$@"
