# Verification-method correction

The original failure measurements are real, but they are insufficient to say
that crop is incorrect or that packed samples leak into each other. This audit
changes verification code and reports only; it does not change training code,
launch defaults, thresholds, batch sizes, optimizers or shared environments.

## Defects in the original test design

1. Crop off/on changes sequence shape. Without a padding-only control, numerical
   drift due to shape and reduced precision is confounded with crop logic.
2. Sample A alone versus A with a neighbour changes total length and batch
   shape. A difference cannot establish attention leakage. Isolation must also
   hold tensor shapes, grids, masks and positions fixed while B changes.
   Both directions must be tested: a missing causal boundary may leak A into B
   while leaving A unaffected by later tokens in B.
3. The gradient objective was NLL over all attended response tokens, including
   tool observations and image placeholders. Production loss excludes those
   tokens. Even the corrected assistant-only NLL objective is not a PPO,
   KL/entropy-loss or optimizer-update equivalence check.

## Actual full-checkpoint controls

All GPU experiments used the same pretrained Qwen3-VL-8B-Instruct checkpoint
and idle physical GPU 1 in the training environment recorded in `results.json`.
GPU 0's unrelated job was not stopped or used. BF16 controls use the actual
actor forward and FlashAttention2. The FP32 control uses direct HF forward with
SDPA and TF32 disabled: the actor forces BF16 autocast, so loading FP32 weights
alone would not provide an FP32 reference.

Initial four-case forward controls:

| Control | Result |
| --- | --- |
| Identical input repeated | All logprob/entropy differences zero |
| Crop disabled, only manual removal of common padding | Exactly reproduces previous crop forward differences |
| Crop enabled versus independently collated compact batch, crop disabled | All logprob/entropy differences zero |
| Fixed-shape B text/image change, padded and packed paths, crop neighbour | Sample A logprob/entropy differences zero |

| Case | BF16/FA2 padding-only max logprob difference | FP32/SDPA padding-only max logprob difference |
| --- | ---: | ---: |
| Text | 0.0887833 | 0.000047684 |
| Original image | 0.124050 | 0.000066280 |
| Returned crop | 1.104372 | 0.000204086 |
| Long tool text | 0 | 0.000146866 |

These controls support shape-sensitive reduced-precision/kernel effects as an
explanation of the original forward discrepancy. The FP32 test changes both
precision and attention backend; it does not isolate which operation causes
the BF16 difference. Large BF16 differences may still matter for optimization,
even when crop's indexing is correct. No tolerance was relaxed.

Repeated BF16 backward is also not exact in the tested runtime: with the same
crop input and crop disabled, the relative NLL gradient difference reached
0.0203879, exceeding the existing 0.02 gate. That gradient run is inconclusive
at the configured tolerance even before comparing implementations. A future
gradient study should first establish repeatability (the installed HF Flash
Attention wrapper supports `FLASH_ATTENTION_DETERMINISTIC=1`) and distinguish
any deterministic-backward reference from production kernel settings. No
deterministic-backward rerun is claimed in these results.

The corrected full crop run uses the scheduler's assistant loss mask:

| Case | Repeat relative NLL gradient difference | Crop versus manual compact relative NLL gradient difference |
| --- | ---: | ---: |
| Text | 0 | 0 |
| Original image | 0.00570493 | 0.00699138 |
| Returned crop | 0.02038788 (fails 0.02 limit) | 0.01528936 |
| Long tool text | 0.01346580 | 0.01425798 |

All four crop-versus-manual forward comparisons are exactly equal and their
NLL gradient comparisons pass. The original padded-versus-crop numerical gate
still fails for text, original image and returned crop. The process correctly
exits 1; these controls improve diagnosis, not the recorded pass status.

The corrected rmpad forward run passes all **16 fixed-shape isolation checks**
(two directions, four neighbour types, padded and packed paths): both logprob
and entropy differences are zero. All four packed-versus-manual-compact checks
pass, with zero logprob difference and at most 0.0000076294 entropy difference.
The original padded-versus-packed comparison still fails in three cases and
the process exits 1. There is no observed sample leakage in the tested cases;
there is still no rmpad gradient or optimizer-update result.

Raw numbers and CPU summaries are in
[method-audit-results.json](method-audit-results.json). Historical
[results.json](results.json) is preserved without rewriting its measurements.

## Script corrections

- Keep the original crop/rmpad numerical gates; add repeat, crop-disabled padding-only
  and independently compacted-input comparisons. Padding-only output is labeled
  diagnostic; failure attribution requires reading these controls together.
- Use the scheduler's real observation mask for assistant-only NLL gradients,
  while reporting logprob/entropy differences over all attended response tokens.
  Zero loss tokens yield zero NLL and zero gradients without a division by zero.
- Replace varying-shape isolation with fixed-shape text/image mutation in both directions;
  run both padded and packed controls with text, image, crop and long-tool-text
  neighbours. Preserve A even when image tensors share storage on the CPU.
- Reject packed checks with an attention backend other than FlashAttention2.
  Output explicitly says that optimizer updates have not been tested.
- Add CPU tests of observation exclusion with the actual processor, compacted
  mRoPE equality, empty-loss gradients and invariants of neighbour mutation.

## Reproduction and remaining scope

CPU regression: **34 tests and 14 subtests passed**, including real-processor
mRoPE and response-budget alignment checks, with no skipped processor tests.
The subsequently extended bidirectional mutation invariant test also passed.

Set `MODEL_PATH` to the full model directory, and choose an idle GPU explicitly.
Use the training environment for GPU commands and the evaluation environment
for pytest; the training environment has no pytest installed.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 QWEN3_VL_PROCESSOR_PATH="$MODEL_PATH" \
    python3 -m pytest -q -rA tests/test_verify_qwen3vl_actor_forward.py

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 \
    scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" --mode crop

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 \
    scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" --mode rmpad --no-grad

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 \
    docs/verification/rl_acceleration_20260930/verify_fp32_padding.py --model-path "$MODEL_PATH"
```

The tests still do not establish full PPO/KL/entropy loss and optimizer-update
equivalence, rmpad gradient equivalence, multi-GPU FSDP correctness, speedup, or
long-run training quality. Synthetic inputs include one returned crop; multiple
crops and response-budget boundaries have CPU coverage elsewhere, not a full
8B GPU equivalence check here. The earlier short training attempt completed
zero steps due to missing `libGL.so.1`, so no training timing baseline exists.
Default flags remain unchanged pending that evidence; this is not a conclusion
that the acceleration implementation is unusable.
