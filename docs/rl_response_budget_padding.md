# Response budgets and common padding cropping

## Configuration

The chat scheduler now enforces `actor_rollout_ref.rollout.response_length`
(normally `${data.max_response_length}`) across the entire response, including
tool observations, image tokens and chat template tokens. It checks the real
processor encoding during rollout and again before constructing batch tensors.
Over-budget responses lose their last complete assistant/observation group;
image placeholders are never sliced. Image decoding and preprocessing are cached
per trajectory. Dropping a group removes its decoded images and processor
features immediately. A completed trajectory releases intermediate image
caches while retaining only final encoding tensors for batch postprocessing;
postprocessing then clears the budget's final encoding reference. Unknown tool
output sizes can only be checked after execution.

```bash
CROP_COMMON_PADDING=True OVERLONG_MASKING=True MAX_RESPONSE_LENGTH=2048 \
  bash scripts/run_visual_tool_rl_2node_16gpu.sh
```

For other launchers, add these Hydra overrides:

```text
actor_rollout_ref.model.crop_common_padding=True
actor_rollout_ref.rollout.multi_turn.overlong_masking=True
data.max_response_length=2048
```

`crop_common_padding` defaults to false and applies only when
`use_remove_padding=False`. It covers actor updates, old policy evaluation and
reference evaluation. It crops shared prompt-left and response-right padding,
keeps the prompt token needed for response prediction, preserves multimodal
features and positions, and pads output probabilities/entropy back to the
original response width. Existing loss aggregation weights stay unchanged.

`overlong_masking` defaults to true. Truncated rows contribute no policy, KL or
entropy loss and are excluded from GRPO group statistics. Empty groups produce
zero advantages. Masked rows still run forward/backward to preserve rank call
counts. When every rank has zero loss tokens, optimizer updates are skipped;
an update call with no optimizer steps also skips the LR scheduler step.

Reward scoring continues to decode the retained response. Empty responses have
a masked tensor placeholder and receive no token-level reward assignment.
`rollout/truncated_rate` and `rollout/empty_loss_rate` are reported every step.
Exported traces contain retained model/tool calls, retained turn counts and
explicitly marked `discarded_calls`. Execution wall-clock fields and existing
execution timing aggregates remain unchanged when calls are discarded.
`retained_model_cumulative_latency_ms`, `retained_tool_cumulative_latency_ms`
and `retained_call_cumulative_latency_ms` describe retained call costs. These
sums can exceed wall-clock time when tools run concurrently; they are not
subtracted from `active_latency_ms` or `total_latency_ms`.
Final truncation precedes trainer image-feature merging and mRoPE rebuilding.

## Verification

Use a Python environment with torch, transformers (Qwen3-VL support), pytest,
tensordict, omegaconf, ray and Pillow. All commands below use CPU:

```bash
OMP_NUM_THREADS=1 python3 scripts/verify_rl_padding.py --qwen3-vl
OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_rl_response_budget.py
python3 -m pytest -q tests/test_multitool_rl_fixes.py \
  tests/test_rollout_source_metadata.py tests/test_visual_agent_thyme_reward.py
```

Set `QWEN3_VL_PROCESSOR_PATH` to local Qwen3-VL tokenizer/processor files when
they are not at the default sibling DeepEyesV2 model location. The processor
tests explicitly skip when those assets are unavailable; model weights are not
needed. The numerical script uses a small, randomly initialized real
`Qwen3VLForConditionalGeneration` on CPU with eager attention and micro batch 1.
It executes the production actor forward and loss functions, comparing valid
token log probabilities, entropy, loss, gradients and one parameter update.
Cases include text, an original image, multiple crops, long tool text and empty
loss. Tensor tolerances are `atol=0.002, rtol=0.02` to accommodate BF16 execution.

Observed CPU Qwen3-VL results: log probability, entropy and loss maximum
differences were zero. The largest gradient difference was 0.000244140625 and
parameter-update difference was 0.000002440065, both within tolerance.

The focused tests additionally exercise whole-group truncation, retained image
features/token counts/mRoPE, preprocessing cache equivalence with the real
processor, trace pruning, GRPO masking, and two CPU ranks using Gloo/DDP. The
distributed test covers one masked rank alongside one valid rank and then all
ranks masked, including unchanged parameters despite existing AdamW state.
The focused suite passed all 13 tests with no skips. After the cache/timing
fixes, the scheduler/metadata regression selection passed 18 tests and 4
subtests; the earlier broader selection passed 47 tests and 83 subtests.
Additional checks use weak references to confirm discarded image tensors and
decoded images are released, retain feature object identity for surviving
images, exercise scheduler cleanup before batch completion, and preserve
execution timing fields despite concurrently discarded calls.
Reward coverage confirms masked rows
are scored from retained responses and empty responses receive no token reward.
AST loading isolates production methods from unavailable FlashAttention/FSDP
imports; control-flow fixtures are not evidence of model compatibility.

An optional small-model GPU comparison is available:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python3 scripts/verify_rl_padding.py \
  --qwen3-vl --device cuda:0
```

It uses the same five cases and limits the PyTorch CUDA allocator to 1% of the
selected GPU's total memory. CUDA context memory is outside that allocator
limit. All five GPU cases passed on physical GPU 1. Log probability and loss
differences were zero; maximum entropy, gradient and update differences were
4.76837158203125e-7, 8.940696716308594e-8 and 1.862645149230957e-9 respectively.
Full pretrained 8B inference, FlashAttention2 and actual FSDP
synchronization have not been validated. No end-to-end throughput improvement
is claimed from these checks. Run a short test on the target training stack
before a long 24/64-GPU run.

## Diagnostics

These metrics do not change training:

- `actor/entropy` is the old-policy entropy on loss tokens (tool observations
  excluded). `actor/entropy_tool_call` and `actor/entropy_non_tool_call` split it
  at `<tool_call>...</tool_call>` spans. Entropy is computed during the old log
  probability pass with row chunking; set `LOG_ENTROPY=False` to skip it.
- `rollout_consistency/*` compares the tokens vLLM sampled with the re-encoded
  training tokens. `turn_exact_match_rate` and `sampled_token_coverage` measure
  re-rendering drift; `logprob_abs_diff_*` compares vLLM and actor log
  probabilities on exactly matching turns. Requests set `logprobs=true` and
  `return_tokens_as_token_ids=true`; if vLLM ignores them, coverage is 0.
- `rollout/truncated_rate/{agent,native,single_turn,multi_turn}` and
  `rollout/row_share/*` split truncation by stream and by tool use.
  `actor/masked_micro_batch_rate` is the share of micro batches without loss
  tokens; these still count in gradient accumulation and shrink the update.

## GPU verification on Qwen3-VL-8B

`tests/test_qwen3vl_rope_index.py` compares the trainer's mRoPE rebuild
(`verl.models.transformers.qwen2_vl.get_rope_index`) with the Transformers
Qwen3-VL implementation. It needs `QWEN3_VL_PROCESSOR_PATH` with config and
processor files, but no weights and no GPU.

`scripts/verify_qwen3vl_actor_forward.py` runs the production actor forward on
the pretrained checkpoint (one GPU, about 40 GB with gradients):

```bash
python3 scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" --mode crop
python3 scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" --mode rmpad
```

`crop` compares `crop_common_padding` off/on. `rmpad` compares
`use_remove_padding` off/on with micro batch 1, then checks that a sample's
outputs do not change when a different sample is packed next to it. Each check
prints one JSON line with the names of failed differences; the exit code is
non-zero when any log-prob, entropy or gradient difference exceeds its limit
(`--max-abs`, `--mean-abs`, `--entropy-max-abs`, `--entropy-mean-abs`,
`--grad-rel`). With `--no-grad`, the padded path computes entropy in float32
while the packed path uses the configured `entropy_from_logits`, so an
`rmpad` failure on entropy alone may be a precision difference. Enable
`CROP_COMMON_PADDING=True` for long runs only after the `crop` mode passes, and
`use_remove_padding` only after the `rmpad` mode passes.

## Resuming checkpoints

Model and optimizer tensor shapes are unchanged. Resume using the existing
checkpoint flow and its topology requirements; no checkpoint conversion is
introduced. Apply identical overrides on every node. The hard response budget
and default masking alter retained trajectories and effective samples, so a
resumed run is not numerically identical to the previous training run. Keep the
budget and masking settings in the run configuration when comparing results.
Batch sizes, dynamic batching, learning rate configuration, NCCL settings and
`use_remove_padding` are not changed by this implementation.

## Implementation Files

- `reinforcement_learning/verl/workers/rollout/response_budget.py`
- `reinforcement_learning/verl/workers/rollout/chat_scheduler.py`
- `reinforcement_learning/verl/workers/actor/dp_actor.py`
- `reinforcement_learning/verl/workers/fsdp_workers.py`
- `reinforcement_learning/verl/trainer/ppo/core_algos.py`
- `reinforcement_learning/verl/trainer/ppo/ray_trainer.py`
- `reinforcement_learning/verl/trainer/ppo/metric_utils.py`
- `reinforcement_learning/verl/workers/reward_manager/naive_async.py`
- `reinforcement_learning/verl/trainer/config/ppo_trainer.yaml`
- `scripts/run_visual_tool_rl_2node_16gpu.sh`
- `scripts/verify_rl_padding.py`
- `tests/test_rl_response_budget.py`
