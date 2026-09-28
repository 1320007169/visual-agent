#!/usr/bin/env bash

# Source before starting services so every node uses the same run namespace.
if [[ "${VISUAL_AGENT_RUN_PATHS_READY:-0}" != "1" && "${RESUME_MODE:-disable}" == "disable" ]]; then
  run_token="${TRAIN_RUN_TOKEN:-${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-}}}}"
  if [[ -z "$run_token" ]]; then
    if [[ "${NNODES:-1}" != "1" && "${MULTITOOL_CONFIG_ONLY:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
      echo "error: set the same TRAIN_RUN_TOKEN on all nodes when the platform supplies no job ID" >&2
      exit 2
    fi
    run_token="$(date -u +%Y%m%dT%H%M%S%N)"
  fi
  [[ "$run_token" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
    echo "error: TRAIN_RUN_TOKEN/job ID must contain only letters, digits, underscores, dots or hyphens" >&2
    exit 2
  }
  run_base_id="${RUN_ID:-visual_agent_rl}"
  run_base_output="${RL_OUTPUT_DIR:-${OUTPUT_DIR:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/$run_base_id}}"
  run_base_logs="${RL_LOG_DIR:-${LOG_DIR:-$BASE/logs/visual-agent-zwz-rl}}"
  export RUN_ID="${run_base_id}_${run_token}"
  export RL_OUTPUT_DIR="${run_base_output}_${run_token}"
  export OUTPUT_DIR="$RL_OUTPUT_DIR"
  if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 ! -name '.launch-node-*' -print -quit)" ]]; then
    echo "error: fresh training output already contains files: $OUTPUT_DIR; use a new TRAIN_RUN_TOKEN or resume explicitly" >&2
    exit 2
  fi
  export RL_LOG_DIR="$run_base_logs/$RUN_ID"
  export LOG_DIR="$RL_LOG_DIR"
  export VALIDATION_DATA_DIR="$OUTPUT_DIR/validation"
  export BEST_HF_MODEL_DIR="$OUTPUT_DIR/best_huggingface"
  export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$BASE/rollouts/visual-agent-zwz-rl/$run_base_id}_${run_token}"
  export SYNC_DIR="${SYNC_DIR:-$BASE/tmp/visual-tool-rl-${NNODES:-1}node/$run_base_id}_${run_token}"
  for run_path_key in VTS_SERVICE_DIR VTS_TOOL_BRIDGE_ROOT RAY_STORAGE_ROOT WANDB_DIR; do
    if [[ -n "${!run_path_key:-}" ]]; then
      export "$run_path_key=${!run_path_key}_${run_token}"
    fi
  done
  for run_log_key in LOG_FILE TOOL_LOG_FILE GPU_MONITOR_LOG GPU_PROCESS_MONITOR_LOG; do
    if [[ -n "${!run_log_key:-}" ]]; then
      export "$run_log_key=$LOG_DIR/$(basename "${!run_log_key}")"
    fi
  done
  if [[ "${MULTITOOL_CONFIG_ONLY:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
    run_node_key="${NODE_RANK:-${MA_NODE_RANK:-${VC_TASK_INDEX:-${HOSTNAME:-node}}}}"
    mkdir -p "$OUTPUT_DIR"
    mkdir "$OUTPUT_DIR/.launch-node-$run_node_key" || {
      echo "error: this node already launched in $OUTPUT_DIR; use a new TRAIN_RUN_TOKEN or resume explicitly" >&2
      exit 2
    }
  fi
  export VISUAL_AGENT_RUN_PATHS_READY=1
fi
