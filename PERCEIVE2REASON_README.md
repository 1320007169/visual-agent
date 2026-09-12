# Perceive2Reason RL Training

Perceive2Reason (P2R) training for visual agent RL. This approach fixes correct perception prefixes and only trains the reasoning suffix, allowing focused optimization on reasoning while preserving learned perception skills.

## Motivation

Analysis of step165 checkpoint showed that **~90% of wrong answers had successful detection but incorrect reasoning**:
- VStarBench: 70% of errors had valid detection boxes but wrong answers
- HRBench4K: 96% of errors had valid detection boxes but wrong answers
- HRBench8K: 92% of errors had valid detection boxes but wrong answers

The bottleneck is not perception (detection works) but reasoning (using the detected information correctly).

## Key Concepts

**Standard RL Training:**
```
[system + image + question] → Model generates [tool calls + reasoning + answer]
                              ↑ entire sequence participates in rollout
```

**Perceive2Reason Training:**
```
[system + image + question] + [fixed tool call history] → Model generates [reasoning + answer]
                              ↑ from successful rollouts   ↑ only this part gets trained
```

**Benefits:**
- Reduces search space (don't re-explore tool calling)
- Prevents catastrophic forgetting of perception skills
- Focuses optimization on the actual bottleneck (reasoning)

## Code Structure

```
visual-agent/
├── reinforcement_learning/
│   ├── data_preprocessing/
│   │   ├── prepare_perceive2reason_data_from_train.py  # Extract P2R samples from training rollouts
│   │   └── prepare_perceive2reason_data.py             # (DEPRECATED: uses eval data)
│   └── verl/workers/rollout/sglang_rollout/
│       └── perceive2reason_rollout.py                  # P2R rollout worker
└── scripts/
    └── run_visual_agent_perceive2reason_2node_16gpu.sh  # Training launch script
```

## Usage

### Step 1: Prepare Training Data

**IMPORTANT:** Must use training set, not evaluation set!

First, run inference on your training data with the current checkpoint to get rollout traces:

```bash
# TODO: Implement this script
python run_inference_on_train.py \
  --checkpoint /path/to/step165 \
  --data /path/to/zwz_train.jsonl \
  --output train_rollout/
```

Then extract P2R samples (where detection succeeded but answer was wrong):

```bash
cd reinforcement_learning/data_preprocessing

python prepare_perceive2reason_data_from_train.py \
  --rollout-dir ../../outputs/train_rollout \
  --output ../../data/perceive2reason/p2r_train.jsonl
```

This will:
- Filter samples where tool calls succeeded but final answer was wrong
- Extract the perception prefix (tool call history)
- Save P2R training samples with fixed prefixes

Expected output:
```
Wrote 500 samples to data/perceive2reason/p2r_train.jsonl

Statistics:
  total: 5000
  correct: 4200
  wrong: 800
  no_tools: 50
  no_valid_detection: 250
  qualified: 500
```

### Step 2: Launch Training

```bash
cd visual-agent/scripts

# Set environment variables
export PERCEIVE2REASON_MODE=true
export PERCEIVE2REASON_DATA_PATH=/path/to/p2r_train.jsonl
export PERCEIVE2REASON_MIX_RATIO=0.3  # 30% P2R, 70% standard

# Launch training
bash run_visual_agent_perceive2reason_2node_16gpu.sh
```

### Step 3: Monitor Training

Training checkpoints will be saved to:
```
saves/visual_agent_perceive2reason/qwen3/perceive2reason_qwen3_base_2node_groundingdino_kl_<JOB_ID>/
```

Logs:
```
logs/visual-agent-p2r/
```

### Step 4: Evaluate

Evaluate checkpoints on benchmarks as usual:

```bash
python run_visual_agent_eval.py \
  --checkpoint saves/visual_agent_perceive2reason/qwen3/.../global_step_XX \
  --benchmarks VStarBench HRBench4K HRBench8K
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PERCEIVE2REASON_MODE` | `false` | Enable P2R mode |
| `PERCEIVE2REASON_DATA_PATH` | `data/perceive2reason/p2r_train.jsonl` | Path to P2R training data |
| `PERCEIVE2REASON_MIX_RATIO` | `0.3` | Fraction of P2R samples (rest are standard) |
| `REASONING_MAX_TOKENS` | `256` | Max tokens for reasoning generation |
| `ROLLOUT_WORKER_TYPE` | `perceive2reason` | Use P2R rollout worker |

### Hyperparameters

P2R training uses the same hyperparameters as successful groundingdino_kl_safe training:

```bash
ACTOR_LR=1e-6
ACTOR_KL_LOSS_COEF=0.001
ACTOR_KL_LOSS_TYPE=low_var_kl
ACTOR_ENTROPY_COEFF=0
```

## Implementation Details

### Perceive2ReasonRollout Worker

Located at: `verl/workers/rollout/sglang_rollout/perceive2reason_rollout.py`

**Key features:**
1. **Detects P2R samples**: Checks for `perception_prefix` in `non_tensor_batch`
2. **Injects fixed prefix**: Adds tool call history to messages before generation
3. **Prohibits tool calls**: Uses logit bias to block `<tool_call>` tokens during reasoning
4. **Falls back gracefully**: Standard samples use normal rollout

**Hook points:**
- `_preprocess_prompt_to_async_rollout_requests()`: Extract `perception_prefix` from batch
- `_async_rollout_a_request()`: Inject fixed prefix and set prohibition flag
- `_handle_engine_call()`: Add logit bias when `prohibit_tool_calls=True`

### Data Format

P2R training data format (JSONL):

```json
{
  "uid": "sample_123",
  "image": "/path/to/image.jpg",
  "question": "What color is the car?",
  "gt_answer": "red",
  "wrong_prediction": "blue",
  "perception_prefix": [
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"grounding_detect\", \"arguments\": {\"query\": \"car\"}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "<tool_response>{\"boxes\": [[100, 200, 300, 400]], \"confidence\": [0.95]}</tool_response>"
    }
  ],
  "num_tool_calls": 1,
  "task_type": "attribute"
}
```

The `perception_prefix` field contains the fixed tool call history that will be injected during rollout.

## Expected Results

Based on step165 error analysis, we expect P2R training to improve:

**Target improvements:**
- Cross-type relation errors: 30% → 20% (10 point improvement)
- Depth relation errors: Systematic failures → 50%+ correct
- Overall accuracy:
  - HRBench4K: 81.8% → 85%+
  - HRBench8K: 76.1% → 80%+

**Key metrics to monitor:**
1. **Reasoning-only accuracy**: On samples with fixed prefixes
2. **Tool-calling preservation**: Ensure standard samples don't degrade
3. **Cross-type vs single-type gap**: Should narrow

## Risks and Mitigations

### Risk 1: Catastrophic Forgetting
**Risk:** Model may lose ability to call tools
**Mitigation:** Mixed training (30% P2R, 70% standard) preserves tool-calling skills

### Risk 2: Overfitting
**Risk:** Small P2R dataset (~500 samples) may cause overfitting
**Mitigation:**
- KL divergence constraint (0.001 coefficient)
- More frequent checkpointing (every 5 steps)
- Early stopping if validation accuracy plateaus

### Risk 3: Suboptimal Prefix Quality
**Risk:** Not all "successful detections" are optimal
**Mitigation:**
- Manual inspection of extracted prefixes
- Filter out obviously bad detections (low confidence, wrong object)
- Future: human annotation of high-quality prefixes

## TODO

- [ ] Implement `run_inference_on_train.py` for training data rollout
- [ ] Add data loading logic to mix P2R and standard samples
- [ ] Complete `_call_sglang_generate()` integration in perceive2reason_rollout.py
- [ ] Add P2R-specific metrics logging
- [ ] Create visualization for P2R training progress
- [ ] Implement human annotation tool for prefix quality control

## References

- Design document: `perceive2reason_rl_design.md`
- Step165 error analysis: `outputs/vlmeval/step165_dino_kl_visual_analysis/`
- Standard GroundingDINO training: `run_visual_agent_zwz_rl_groundingdino_2node_16gpu.sh`
