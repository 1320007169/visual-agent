#!/usr/bin/env bash
set -Eeuo pipefail
set -x
export HYDRA_FULL_ERROR=1


PROJECT_NAME="${PROJECT_NAME:-visual-agent-qwen3vl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl-8b-deepeyesv2}"
export SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-./save_checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$RL_ROOT/.." && pwd)"
RL_DATA_DIR="${RL_DATA_DIR:-$REPO_ROOT/model/xiaoyi_tmpstorage/haohang/min/gx/datasets/DeepEyesV2_RL}"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/DeepEyesV2/models/Qwen3-VL-8B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/visual-agent-qwen3vl-rl/bin/python}"
export PYTHONPATH="$RL_ROOT:${PYTHONPATH:-}"
cd "$RL_ROOT"


PERCEPTION_TRAIN_PARQUET_1="$RL_DATA_DIR/perception_all_1.parquet"
PERCEPTION_TRAIN_PARQUET_2="$RL_DATA_DIR/perception_all_2.parquet"
PERCEPTION_TRAIN_PARQUET_3="$RL_DATA_DIR/perception_all_3.parquet"
PERCEPTION_TRAIN_PARQUET_4="$RL_DATA_DIR/perception_all_4.parquet"
PERCEPTION_TRAIN_PARQUET_5="$RL_DATA_DIR/perception_all_5.parquet"
SEARCH_TRAIN_PARQUET="$RL_DATA_DIR/search.parquet"
REASON_TRAIN_PARQUET="$RL_DATA_DIR/reason.parquet"

VSTAR_TEST_PARQUET="$RL_DATA_DIR/vstar_test.parquet"

CUSTOM_STOP='["</code>","</tool_call>"]'
LOSS_AGG_MODE="token-mean"
export WORKING_DIR="${WORKING_DIR:-$RL_ROOT}"
export RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

REF_MODEL_PATH="${REF_MODEL_PATH:-$MODEL_PATH}"
INCLUDE_SEARCH_DATA="${INCLUDE_SEARCH_DATA:-0}"
TRAIN_FILES="[${PERCEPTION_TRAIN_PARQUET_1},${PERCEPTION_TRAIN_PARQUET_2},${PERCEPTION_TRAIN_PARQUET_3},${PERCEPTION_TRAIN_PARQUET_4},${PERCEPTION_TRAIN_PARQUET_5},${REASON_TRAIN_PARQUET}]"
if [[ "$INCLUDE_SEARCH_DATA" == "1" ]]; then
    [[ -f "$SEARCH_TRAIN_PARQUET" ]] || { echo "error: search parquet not found: $SEARCH_TRAIN_PARQUET" 1>&2; exit 1; }
    [[ -n "${DEEPEYES_SEARCH_CACHE_PATHS:-}" ]] || echo "warning: search data enabled without DEEPEYES_SEARCH_CACHE_PATHS" 1>&2
    TRAIN_FILES="${TRAIN_FILES%]},${SEARCH_TRAIN_PARQUET}]"
elif [[ "$INCLUDE_SEARCH_DATA" != "0" ]]; then
    echo "error: INCLUDE_SEARCH_DATA must be 0 or 1" 1>&2
    exit 2
fi

HYDRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    HYDRA_ARGS=(--cfg job)
fi

for required_path in "$PYTHON_BIN" "$REF_MODEL_PATH" "$PERCEPTION_TRAIN_PARQUET_1" "$PERCEPTION_TRAIN_PARQUET_2" "$PERCEPTION_TRAIN_PARQUET_3" "$PERCEPTION_TRAIN_PARQUET_4" "$PERCEPTION_TRAIN_PARQUET_5" "$REASON_TRAIN_PARQUET" "$VSTAR_TEST_PARQUET"; do
    [[ -e "$required_path" ]] || { echo "error: required path not found: $required_path" 1>&2; exit 1; }
done
mkdir -p "$LOG_DIR" "$SAVE_CHECKPOINT_DIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=[${VSTAR_TEST_PARQUET}] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=8192 \
    actor_rollout_ref.rollout.agent.max_turns=9 \
    actor_rollout_ref.rollout.agent.concurrent_workers=2 \
    actor_rollout_ref.rollout.agent.custom_stop=${CUSTOM_STOP} \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive_async \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=4 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 "${HYDRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${EXPERIMENT_NAME}.log"
