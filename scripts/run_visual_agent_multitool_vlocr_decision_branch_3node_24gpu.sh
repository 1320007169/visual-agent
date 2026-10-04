#!/usr/bin/env bash
set -Eeuo pipefail

# Five-tool VL-OCR RL with counterfactual first-decision branches and hierarchical
# advantages. Everything else matches run_visual_agent_multitool_vlocr_3node_24gpu.sh.
# Ablations: DECISION_BRANCH_ADV=grpo keeps the forced branches with plain GRPO;
# DECISION_BRANCH_FORCED_POSITIVE_WEIGHT=0 disables the off-policy update.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DECISION_BRANCH_ENABLE=True
export DECISION_BRANCH_ADV="${DECISION_BRANCH_ADV:-decision_branch}"
export DECISION_BRANCH_FORCED_PER_DECISION="${DECISION_BRANCH_FORCED_PER_DECISION:-1}"
export DECISION_BRANCH_FORCED_POSITIVE_WEIGHT="${DECISION_BRANCH_FORCED_POSITIVE_WEIGHT:-1.0}"
export DECISION_BRANCH_FORCED_NEGATIVE_WEIGHT="${DECISION_BRANCH_FORCED_NEGATIVE_WEIGHT:-0.0}"
export DECISION_BRANCH_DECISION_WEIGHT="${DECISION_BRANCH_DECISION_WEIGHT:-1.0}"
JOB_TOKEN="${MA_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-manual}}}"
export RUN_ID="${RUN_ID:-qwen3base_multitool_vlocr_decision_branch_${DECISION_BRANCH_ADV}_n16_3node_${JOB_TOKEN}}"

if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then
    for key in DECISION_BRANCH_ENABLE DECISION_BRANCH_ADV DECISION_BRANCH_FORCED_PER_DECISION \
        DECISION_BRANCH_FORCED_POSITIVE_WEIGHT DECISION_BRANCH_FORCED_NEGATIVE_WEIGHT DECISION_BRANCH_DECISION_WEIGHT RUN_ID; do
        printf '%s=%s\n' "$key" "${!key}"
    done
fi

exec bash "$SCRIPT_DIR/run_visual_agent_multitool_vlocr_3node_24gpu.sh" "$@"
