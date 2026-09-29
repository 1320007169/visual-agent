# RL verification results, 2026-09-30

**Read the [verification-method correction](method-audit.md) first.** The
historical failure flags below do not establish crop indexing errors or packed
sample leakage. Padding-only controls reproduce the differences; the old NLL
gradient mask included observations, and the old isolation test changed shape.

Tested commit: `56e0fcac41b4c3efa71c02c851b2e2d65298b6ef`.
See the [review report](../../rl_acceleration_review_20260930.md) for findings
and limitations. These artifacts record the existing tests without changing
their tolerances or the production training configuration.

## Results

- CPU: 56 tests and 15 subtests passed, including both real-processor mRoPE
  comparisons. None of the mRoPE cases was skipped.
- Full pretrained 8B crop: three of four comparisons failed. Gradients were
  included. Default crop remains disabled.
- Full pretrained 8B remove padding: three of four forward comparisons and
  all three varying-shape comparisons failed. Gradients were not checked.
  Remove padding remains disabled.
- Training smoke: zero steps completed. OpenCV could not load `libGL.so.1`
  during vLLM initialization. There is no training timing baseline.

`results.json` contains the measured GPU differences and limits; the CPU counts
are summaries of the captured test executions. `regression.txt` is the raw
output of the 25-test regression selection below. `smoke-failure.txt` contains
the relevant error excerpts rather than the full runtime/configuration log.

## Reproduction

Run from the repository root in the corresponding Python environment recorded
in `results.json`. Set `MODEL_PATH` to the local pretrained Qwen3-VL-8B-Instruct
directory, including weights, config and processor files.

```bash
export QWEN3_VL_PROCESSOR_PATH="$MODEL_PATH"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 -m pytest -q -rA \
    tests/test_qwen3vl_rope_index.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 -m pytest -q -rA \
    tests/test_rl_response_budget.py \
    tests/test_rollout_consistency.py \
    tests/test_verify_qwen3vl_actor_forward.py \
    tests/test_qwen3vl_actor_forward.py \
    reinforcement_learning/tests/workers/test_fsdp_worker_old_log_prob_entropy_on_cpu.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 -m pytest -q \
    tests/test_multitool_rl_fixes.py \
    tests/test_rollout_source_metadata.py \
    tests/test_modelarts_launcher_converter.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3 \
    docs/verification/rl_acceleration_20260930/reproduce_metrics.py
```

These are historical commands for commit `56e0fca`; the current script adds
controls and corrects the gradient mask and isolation design. New results are
linked from the method audit.

GPU 1 was verified idle before the GPU runs; select an available GPU on another
host. Use a working C compiler for Triton (the tested environment required its
conda GCC through `CC`). The following commands are expected to return exit
code 1 in the recorded environment because numerical gates failed.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 \
    scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" --mode crop

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 \
    scripts/verify_qwen3vl_actor_forward.py --model-path "$MODEL_PATH" \
    --mode rmpad --no-grad
```

The isolated smoke attempt used the real trainer with one GPU, a separate local
Ray instance, six synthetic images, train batch 2, three requested steps,
agent/native rollout counts 2 each, micro batch 1, prompt/response limits
1024/128, local rule rewards and an empty tool configuration. Crop and remove
padding were disabled. It reached FSDP model initialization and then failed
starting vLLM. This is not a production experiment or a multi-tool throughput
comparison; no `timing_s/*`, entropy or rollout-consistency training metrics
were produced.
