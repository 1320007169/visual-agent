#!/usr/bin/env bash
set -Eeuo pipefail

# One checkpoint x three modes x the same three benchmarks.
# This wrapper only sets existing launcher parameters; it does not change the
# shared inference loop, evaluator, or the four-checkpoint entrypoint.
#
# DIAGNOSTIC_CONFIG_ONLY=1 prints/validates the plan without starting services.
# DIAGNOSTIC_MODEL_PATH and DIAGNOSTIC_STEP select another checkpoint.
# DIAGNOSTIC_MODES can select a subset of: direct auto tool_first.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export BASE="${BASE:-$(dirname "$REPO_ROOT")}"
export CONFIG_PYTHON="${CONFIG_PYTHON:-python}"
DIAGNOSTIC_STEP="${DIAGNOSTIC_STEP:-120}"
DIAGNOSTIC_MODEL_PATH="${DIAGNOSTIC_MODEL_PATH:-$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_2node/global_step_120/actor/huggingface}"
DIAGNOSTIC_MODES="${DIAGNOSTIC_MODES:-direct auto tool_first}"
DIAGNOSTIC_CONFIG_ONLY="${DIAGNOSTIC_CONFIG_ONLY:-0}"
PROMPT_FILE="${DIAGNOSTIC_PROMPT_FILE:-$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt}"
GROUP_ID="${DIAGNOSTIC_RUN_ID:-tool_diagnostic_step${DIAGNOSTIC_STEP}_$(date +%Y%m%d_%H%M%S)}"
GROUP_ROOT="${DIAGNOSTIC_OUTPUT_ROOT:-$REPO_ROOT/outputs/vlmeval/tool_diagnostic}/$GROUP_ID"
export EVAL_DATASETS="${EVAL_DATASETS:-VStarBench HRBench4K HRBench8K}"
export VISUAL_AGENT_MAX_TURNS="${VISUAL_AGENT_MAX_TURNS:-6}"
export VISUAL_AGENT_MAX_TOKENS="${VISUAL_AGENT_MAX_TOKENS:-6144}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MODEL_SERVER_BACKEND=vllm
export ENV_DIR="${ENV_DIR:-/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/qwenvl3_xmx_vLLM}"
export VLLM_PYTHON="$ENV_DIR/bin/python"
export CUDA_HOME="${CUDA_HOME:-$BASE/conda_envs/spacetools-rl}"
export MODEL_CUDA_HOME="${MODEL_CUDA_HOME-$CUDA_HOME}"
export MODEL_CC="${MODEL_CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
export MODEL_CXX="${MODEL_CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
export TOOL_CUDA_HOME="${TOOL_CUDA_HOME-/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118}"
export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0}"
export VISUAL_TOOL_BACKEND=groundingdino SAM3_REPLICAS=0
export GROUNDING_DINO_REPLICAS="${GROUNDING_DINO_REPLICAS:-2}"
export VISUAL_AGENT_ALLOWED_TOOL_NAMES=grounding_detect,crop_zoom
export VLMEVAL_API_NPROC="${VLMEVAL_API_NPROC:-28}"
export VLMEVAL_FAILED_SAMPLE_RETRIES="${VLMEVAL_FAILED_SAMPLE_RETRIES:-5}"
export EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-0}"
export LOG_DIR="$GROUP_ROOT/logs"

"$CONFIG_PYTHON" - "$DIAGNOSTIC_MODEL_PATH" "$DIAGNOSTIC_STEP" "$DIAGNOSTIC_MODES" \
  "$GROUP_ROOT" "$GROUP_ID" "$PROMPT_FILE" "$DIAGNOSTIC_CONFIG_ONLY" "${NNODES:-1}" <<'PY'
import json
import os
from pathlib import Path
import re
import sys

model, step, mode_text, output, run_id, prompt_file, config_only, nodes = sys.argv[1:]
modes = mode_text.split()
if nodes != "1" or config_only not in {"0", "1"}:
    raise SystemExit("Use one node; DIAGNOSTIC_CONFIG_ONLY must be 0 or 1")
if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not step.isdigit():
    raise SystemExit("Invalid diagnostic run ID or checkpoint step")
if not modes or len(modes) != len(set(modes)) or set(modes) - {"direct", "auto", "tool_first"}:
    raise SystemExit("DIAGNOSTIC_MODES must contain unique entries from direct, auto, tool_first")
turns, tokens = int(os.environ["VISUAL_AGENT_MAX_TURNS"]), int(os.environ["VISUAL_AGENT_MAX_TOKENS"])
if turns < 1 or tokens < 1 or ("tool_first" in modes and turns < 2):
    raise SystemExit("Positive budgets required; tool_first needs at least two turns")
datasets = os.environ["EVAL_DATASETS"].split()
if not datasets or len(datasets) != len(set(datasets)) or set(datasets) - {"VStarBench", "HRBench4K", "HRBench8K"}:
    raise SystemExit("Select a nonempty subset of VStarBench, HRBench4K, HRBench8K")
path = Path(model).resolve()
json.loads((path / "config.json").read_text())
# Both reused launchers require an indexed HuggingFace checkpoint.
weights = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
if not weights or any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in weights):
    raise SystemExit(f"Missing/empty weight shards: {path}")
if path.name == "best_huggingface":
    metadata = json.loads((path.parent / "best_checkpoint.json").read_text())
    if metadata["step"] != int(step):
        raise SystemExit(f"Best checkpoint changed: expected {step}, got {metadata['step']}")
prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
if not prompt:
    raise SystemExit("Empty agent prompt")
prompt = re.sub(r"You have at most \d+ assistant turns, including the final answer\. Make at most \d+ tool calls", f"You have at most {turns} assistant turns, including the final answer. Make at most {turns - 1} tool calls", prompt)
required = prompt + "\n\nEvaluation instruction (takes priority over the optional-tool strategy above):\n" + (
    "Your first assistant response must call grounding_detect or crop_zoom on a relevant object or region, "
    "rather than give the final answer. Choose the query or crop yourself. Inspect the returned tool "
    "observation before answering. For spatial relations, localize the target and anchor separately "
    "when needed. Reserve the final assistant turn for the answer.\n"
)
protocol = {
    "checkpoint": str(path), "step": int(step), "modes": modes, "datasets": datasets,
    "max_turns": turns, "max_tokens_per_turn": tokens, "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "backend": "vllm", "model_gpus": os.environ["MODEL_CUDA_VISIBLE_DEVICES"],
    "direct": "Existing use_tools=false path: one model call and the existing concise-answer prompt",
    "auto": "Existing agent prompt and tool loop",
    "tool_first": "Same agent prompt plus an explicit first-tool instruction; no injected tool calls",
    "note": "All modes use the same benchmark questions. Prompt and turn budget differ between direct and agent modes.",
}
print(json.dumps(protocol, ensure_ascii=False, indent=2))
print(f"Results: {output}")
if config_only == "0":
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)  # Prevent reuse of another run's predictions.
    (root / "logs").mkdir()
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (root / "auto_prompt.txt").write_text(prompt + "\n")
    (root / "tool_first_prompt.txt").write_text(required)
PY

[[ "$DIAGNOSTIC_CONFIG_ONLY" == 1 ]] && exit 0
printf 'mode\texit_code\tseconds\tresult_dir\n' > "$GROUP_ROOT/status.tsv"
trap 'exit 130' INT
trap 'exit 143' TERM
failed=0
for mode in $DIAGNOSTIC_MODES; do
  export RUN_ID="${GROUP_ID}_${mode}"
  export VLMEVAL_EVAL_ID="T_${RUN_ID}"
  export WORK_ROOT="$GROUP_ROOT/$mode"
  if [[ "$mode" == direct ]]; then
    # This existing launcher already supports a true no-tool path. Its model
    # selector is named qwen3_base, but QWEN3_MODEL_PATH selects our checkpoint.
    export EVAL_MODELS=qwen3_base QWEN3_MODEL_PATH="$DIAGNOSTIC_MODEL_PATH"
    unset VISUAL_AGENT_SYSTEM_PROMPT_FILE
    entry="$REPO_ROOT/scripts/run_visual_agent_eval_qwen3_compare.sh"
    result_dir="$WORK_ROOT/qwen3_base"
  else
    export EVAL_MODELS=dino_latest
    export RL_DINO_LATEST_STEP="$DIAGNOSTIC_STEP"
    export RL_DINO_LATEST_MODEL_PATH="$DIAGNOSTIC_MODEL_PATH"
    export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$GROUP_ROOT/${mode}_prompt.txt"
    entry="$REPO_ROOT/scripts/run_visual_agent_eval_qwen3.sh"
    result_dir="$WORK_ROOT/dino_latest"
  fi
  echo "Starting $mode on $DIAGNOSTIC_MODEL_PATH"
  started=$SECONDS
  status=0
  bash "$entry" || status=$?
  printf '%s\t%s\t%s\t%s\n' "$mode" "$status" "$((SECONDS - started))" "$result_dir" >> "$GROUP_ROOT/status.tsv"
  if (( status != 0 )); then failed=1; fi
done
echo "Completed diagnostic attempts: $GROUP_ROOT/status.tsv"
exit "$failed"
