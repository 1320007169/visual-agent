#!/usr/bin/env bash
set -Eeuo pipefail

# Run each prompt variant with the same VLOCR_MODEL_PATH and VLOCR_STEP.
export BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"
[[ -n "${VLOCR_MODEL_PATH:-}" && -n "${VLOCR_STEP:-}" ]] || {
    echo "error: set VLOCR_MODEL_PATH (global_step_*/actor/huggingface) and VLOCR_STEP" >&2
    exit 2
}
variant="${MME_PROMPT_VARIANT:-evidence}"
case "$variant" in
    baseline) export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_multitool_vlocr.txt" ;;
    evidence) export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_eval_mme_evidence.txt" ;;
    *) echo "error: MME_PROMPT_VARIANT must be baseline or evidence" >&2; exit 2 ;;
esac
[[ -s "$VISUAL_AGENT_SYSTEM_PROMPT_FILE" ]] || {
    echo "error: missing or empty agent prompt: $VISUAL_AGENT_SYSTEM_PROMPT_FILE" >&2
    exit 2
}
export EVAL_DATASETS="${EVAL_DATASETS:-MME-RealWorld-Lite}"
export RUN_ID="${RUN_ID:-mme_prompt_${variant}_step${VLOCR_STEP}_$(date +%Y%m%dT%H%M%S%N)_$$}"
export WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/outputs/vlmeval/mme_prompt_8gpu/$RUN_ID}"
export VLMEVAL_EVAL_ID="$RUN_ID"
export VISUAL_AGENT_FAILURE_TRACE_DIR="$WORK_ROOT/failure_traces"
# A prompt comparison requires fresh predictions, including previously successful rows.
unset VLOCR_REUSE_GROUP_ROOT VLOCR_RETRY_FAILED_ONLY
printf 'Variant: %s\nCheckpoint: %s\nPrompt: %s\nDatasets: %s\nResults: %s\n' \
    "$variant" "$VLOCR_MODEL_PATH" "$VISUAL_AGENT_SYSTEM_PROMPT_FILE" "$EVAL_DATASETS" "$WORK_ROOT"
if [[ "${CHECKPOINT_EVAL_CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi
exec bash "$REPO_ROOT/scripts/run_visual_agent_eval_multitool_vlocr_8gpu.sh" "$@"
