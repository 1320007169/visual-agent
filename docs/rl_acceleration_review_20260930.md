# RL acceleration review (2026-09-30)

Reviewed commit: `56e0fcac41b4c3efa71c02c851b2e2d65298b6ef`.
The checkout already contains the rollout diagnostics, full-model verification
script, and the entropy verification gate fix (all reported differences,
including entropy, have limits; NaN fails). No standalone `batch1_2` or
`verify-entropy-gate-fix` patch files were found, and no patch was applied twice.
Production code and launcher defaults have not been changed during this review.

## Findings

1. **P1: full pretrained 8B crop verification does not pass.** The existing
   production forward, BF16 and FlashAttention2 were tested on the actual
   Qwen3-VL-8B-Instruct checkpoint. Three of four cases exceed the committed
   numerical limits. This blocks enabling `CROP_COMMON_PADDING` by default.
   Integer mRoPE equality and small random-model tests do not establish full
   model numerical equivalence. The test establishes a mismatch, not its root
   cause; shape-dependent BF16/kernel behavior still needs to be distinguished
   from an implementation error. Do not relax the gates merely to pass.

   | Case | Max logprob difference | Mean logprob difference | Max entropy difference | Relative gradient difference | Result |
   | --- | ---: | ---: | ---: | ---: | --- |
   | text | 0.0887833 | 0.0189011 | 0.0577202 | 0.0441311 | fail |
   | original image | 0.124050 | 0.0326161 | 0.0347290 | 0.241771 | fail |
   | returned crop | 1.104372 | 0.149057 | 0.669022 | 0.518095 | fail |
   | long tool text | 0 | 0 | 0 | 0.0131242 | pass |

   Limits remain max/mean logprob and entropy differences 0.05/0.005,
   relative gradient difference 0.02. GPU environment: torch 2.8.0,
   transformers 4.57.1, flash-attn 2.8.3. Only physical GPU 1 was used.

   The optional `--mode rmpad --no-grad` check also failed: three of four
   numerical comparisons and all three sample-isolation comparisons exceeded
   the limits. Isolation max logprob differences were 1.12538 (long tool text
   neighbour), 1.10437 (text neighbour), and 1.39022 (image neighbour). This
   establishes failure of the isolation gate, not by itself proof of attention
   leaking across samples; shape-dependent numerical effects must also be
   investigated. No rmpad gradient verification was completed. Keep remove
   padding disabled.

2. **P2: entropy segmentation can include observation markup.** In
   `ray_trainer.py:385`, the tool-call cumulative delimiter count runs across
   all response tokens before the loss mask is applied. An unclosed
   `<tool_call>` quoted in an excluded tool observation classifies the next
   assistant answer as a tool call. A CPU reproduction produced
   `actor/entropy_tool_call=4.0` despite zero trainable tool-call tokens.
   Segment within assistant messages, reset state at message boundaries, and
   define precedence for quoted tool calls inside thought text. The current
   implementation provides only tool_call/non_tool_call, not separate
   think/tool_call/answer categories or their token counts.

3. **P2: truncation breakdown measures retained turns, not executed turns/tool
   usage.** `ray_trainer.py:428` uses `__num_turns__`, which postprocess updates
   after truncation. A trajectory with three executed turns trimmed to one is
   counted as single-turn; an empty retained trajectory is also single-turn.
   This hides the relation between tool usage and truncation. Preserve executed
   turn/tool counts separately and label the current retained-turn breakdown
   explicitly. `actor/masked_micro_batch_rate` is not the requested truncated
   loss-token share; both the actual contribution (expected zero under masking)
   and the excluded candidate token share are still missing.

4. **P2: missing sampling data is reported as token mismatch.**
   `rollout_consistency.py:57` increments the denominator before checking for
   missing logprobs. A response with unavailable sampling data produces exact
   match rate 0, even though no comparison is possible. Track availability
   separately and compute mismatch over observed turns. Therefore a low
   `turn_exact_match_rate` alone does not prove token drift. The logprob metric
   also needs an explicit temperature/logprobs-mode contract: the installed
   vLLM defaults to raw logprobs while actor logits are divided by temperature.
   Existing launchers use temperature 1, so that specific mismatch does not
   affect their current defaults. Prompt-token identity is not checked by this
   monitor, so exact assistant-token matches alone do not prove equal contexts.

5. **Shared-host launcher risk:** `run_visual_tool_rl_2node_16gpu.sh` calls
   `ray stop --force` during startup and cleanup (including lines 560 and 833).
   Its normal path is unsuitable for this shared-machine verification because
   it can stop unrelated Ray jobs. The attempted smoke uses its own local Ray
   instance, namespace and temp directory and calls only `ray.shutdown()` for
   its own instance. No unrelated process was stopped.

6. **MFU is not a useful speed measure for this checkpoint yet.** During the
   real worker startup, `verl/utils/flops_counter.py:68` reports that
   `qwen3_vl` is unsupported and MFU will always be zero. Use measured step and
   phase times; a zero MFU value here does not imply zero useful GPU work.

## Verified behavior and scope

- Real processor/config mRoPE test: **2 passed**, no skips. Text and original
  image plus two crops/tool text/left and right padding match HF Qwen3-VL at
  every valid position. No evidence from these cases requires replacing the
  trainer mRoPE implementation; this does not cover video.
- Budget, sampling alignment, verification gates and entropy gating: **29
  passed, 10 subtests passed**, including CPU two-rank masked updates and actual
  processor image-cache alignment.
- Existing rollout, metadata and launcher tests: **25 passed, 5 subtests
  passed**.
- Truncated rows remain excluded from GRPO statistics and policy/KL/entropy
  loss; all-rank empty loss skips optimizer updates while retaining collective
  forward/backward call counts. Actual multi-GPU FSDP was not rerun here.
- The new sampled logprobs are only consumed by diagnostic metrics; policy
  loss still uses recomputed old logprobs. Entropy logging does not change
  `entropy_coeff`, optimizer settings or the update loss, but adds computation
  and changes the reported entropy aggregation mask/precision.
- Hard response truncation and overlong masking themselves intentionally
  change retained training examples and effective loss weight. They are not
  equivalent to merely removing padding. Existing micro-batch accumulation
  denominators are preserved, so more masked rows can reduce update magnitude.
- Current default `crop_common_padding=False`, `use_remove_padding=False` and
  multimodal dynamic batching remain disabled. The latter is still unsafe to
  enable casually: `compute_log_prob` takes the multimodal branch but later
  accesses dynamic-batch `indices` which that branch did not initialize.

## Artifacts and reproduction

Committed results and reproduction commands are in
[verification/rl_acceleration_20260930](verification/rl_acceleration_20260930/README.md):

- [results.json](verification/rl_acceleration_20260930/results.json): complete
  numerical differences, pass limits, environment versions and CPU summaries.
- [metrics-reproductions.jsonl](verification/rl_acceleration_20260930/metrics-reproductions.jsonl):
  diagnostic counterexamples, generated by the adjacent portable
  `reproduce_metrics.py` script.
- [regression.txt](verification/rl_acceleration_20260930/regression.txt):
  captured regression test output.
- [smoke-failure.txt](verification/rl_acceleration_20260930/smoke-failure.txt):
  extracted errors from the isolated training attempt.

Original local run artifacts remain in `/tmp/visual-agent-review-20260930/`:

- `crop-training-env.log`: full 8B numerical results and failed gates.
- `rmpad-forward.log`: full 8B forward-only and sample-isolation results.
- `results.json`: machine-readable test outcomes; training metrics are empty
  because no training step completed.
- `crop.log`: initial attempt in the evaluation environment, blocked by its
  default system GCC missing `cc1`; rerun used the training environment and its
  available conda GCC without changing shared environments.
- `reproduce_metrics.py`, `metrics-reproductions.jsonl`: CPU diagnostic
  counterexamples.
- `regression.log`: regression test results.
- `smoke.py`, `smoke*.log`: isolated three-step smoke configuration and logs.

CPU processor path:
`QWEN3_VL_PROCESSOR_PATH=../DeepEyesV2/models/Qwen3-VL-8B-Instruct`.
GPU command uses `scripts/verify_qwen3vl_actor_forward.py --mode crop
--model-path ../DeepEyesV2/models/Qwen3-VL-8B-Instruct` with
`CUDA_VISIBLE_DEVICES=1` and the training environment Python.

The smoke uses synthetic images, local rule rewards, no external tool service,
one GPU, separate outputs and crop disabled. Even if it completes, it would
validate infrastructure only, not establish production 24/64-GPU throughput or
multi-tool performance. A comparable speed baseline requires the same model,
dataset, batch sizes, rollout settings, precision and logging settings before
and after the change.

The attempted smoke did **not** reach a training step. After FSDP loaded the
actual 8B model, vLLM 0.11.0 engine startup failed because OpenCV could not load
`libGL.so.1`. No system packages or shared environments were modified. The
isolated Ray cluster was shut down and GPU 1 memory was released; the original
GPU 0 process remained running. Consequently there are no real training values
for `rollout_consistency/*`, `actor/entropy*`, truncation metrics or `timing_s/*`
from this attempt, and no throughput baseline is claimed.
