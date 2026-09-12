#!/usr/bin/env bash
# Perceive2Reason Quick Start
# 快速验证P2R实现是否正确的脚本

set -Eeuo pipefail

BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"
REPO_ROOT="$BASE/visual-agent"

echo "==================== Perceive2Reason Quick Validation ===================="
echo ""

# 1. 检查文件是否存在
echo "[1/5] Checking files..."
FILES=(
    "$REPO_ROOT/reinforcement_learning/verl/workers/rollout/sglang_rollout/perceive2reason_rollout.py"
    "$REPO_ROOT/reinforcement_learning/data_preprocessing/prepare_perceive2reason_data_from_train.py"
    "$REPO_ROOT/scripts/run_visual_agent_perceive2reason_2node_16gpu.sh"
    "$REPO_ROOT/PERCEIVE2REASON_README.md"
    "$REPO_ROOT/PERCEIVE2REASON_IMPLEMENTATION.md"
)

all_exist=true
for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $(basename "$file")"
    else
        echo "  ✗ $(basename "$file") NOT FOUND"
        all_exist=false
    fi
done

if [[ "$all_exist" == "false" ]]; then
    echo ""
    echo "Error: Some files are missing!"
    exit 1
fi

echo ""
echo "[2/5] Checking Python syntax..."
python -m py_compile "$REPO_ROOT/reinforcement_learning/verl/workers/rollout/sglang_rollout/perceive2reason_rollout.py" && echo "  ✓ perceive2reason_rollout.py" || echo "  ✗ Syntax error in perceive2reason_rollout.py"
python -m py_compile "$REPO_ROOT/reinforcement_learning/data_preprocessing/prepare_perceive2reason_data_from_train.py" && echo "  ✓ prepare_perceive2reason_data_from_train.py" || echo "  ✗ Syntax error in prepare_perceive2reason_data_from_train.py"

echo ""
echo "[3/5] Checking imports..."
cd "$REPO_ROOT/reinforcement_learning"
python -c "from verl.workers.rollout.sglang_rollout.perceive2reason_rollout import Perceive2ReasonRollout; print('  ✓ Perceive2ReasonRollout imports successfully')" 2>/dev/null || echo "  ⚠ Import failed (may need PYTHONPATH or dependencies)"

echo ""
echo "[4/5] Documentation summary..."
echo "  - README: $(wc -l < "$REPO_ROOT/PERCEIVE2REASON_README.md") lines"
echo "  - Implementation doc: $(wc -l < "$REPO_ROOT/PERCEIVE2REASON_IMPLEMENTATION.md") lines"
echo "  - Design doc: $(wc -l < "$REPO_ROOT/perceive2reason_rl_design.md") lines"

echo ""
echo "[5/5] Next steps..."
echo ""
echo "To use Perceive2Reason training:"
echo ""
echo "  Step 1: Generate training rollout data"
echo "    # TODO: Implement run_inference_on_train.py"
echo "    python run_inference_on_train.py \\"
echo "      --checkpoint saves/.../global_step_165 \\"
echo "      --data /path/to/train.jsonl \\"
echo "      --output outputs/train_rollout"
echo ""
echo "  Step 2: Extract P2R samples"
echo "    python reinforcement_learning/data_preprocessing/prepare_perceive2reason_data_from_train.py \\"
echo "      --rollout-dir outputs/train_rollout \\"
echo "      --output data/perceive2reason/p2r_train.jsonl"
echo ""
echo "  Step 3: Launch training"
echo "    export PERCEIVE2REASON_MODE=true"
echo "    export PERCEIVE2REASON_DATA_PATH=data/perceive2reason/p2r_train.jsonl"
echo "    bash scripts/run_visual_agent_perceive2reason_2node_16gpu.sh"
echo ""
echo "For detailed documentation, see:"
echo "  - PERCEIVE2REASON_README.md (usage guide)"
echo "  - PERCEIVE2REASON_IMPLEMENTATION.md (technical details)"
echo "  - perceive2reason_rl_design.md (design rationale)"
echo ""
echo "=========================================================================="
