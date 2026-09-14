# SLIME Visual-Agent PoC

This is an isolated compatibility and correctness experiment. It does not
replace the existing VERL launchers, datasets, checkpoints, Ray cluster, or
visual-tool processes.

## Compatibility pin

The exact Qwen3-VL path uses SLIME's FSDP backend at commit
`0104a9e922ecd3cea768f7c2520e7d1a122afdc9`. SLIME removed FSDP in commit
`e4faf633095ed5a99cdb22858673b94235ea9c8c` on 2026-03-04. Current SLIME
uses Megatron only and does not include a native Qwen3-VL-8B VLM provider.
Do not move this PoC to another SLIME commit without rerunning the fixed
trajectory and GPU update tests.

## What the one-update run verifies

The launcher runs one complete synchronous update:

1. SGLang receives the original image and generates an XML tool call.
2. The adapter calls the existing visual-tool HTTP service.
3. Returned crop images are appended to the next Qwen3-VL turn.
4. Tool-observation text and image tokens are masked from the policy loss.
5. The deterministic ZWZ reward is computed.
6. FSDP applies one GRPO update and SLIME synchronizes the new weights back
   to SGLang. The PoC driver then compares the updated actor and SGLang
   weights and fails the run if they differ.
7. Rollout and train debug payloads plus the checkpoint are written only
   below `SLIME_POC_ROOT`.

The launcher does not run `pkill`, `ray stop`, or reuse the existing VERL
output directory. By default it also does not start Ray. Point it at a
dedicated Ray cluster, or explicitly set `START_LOCAL_RAY=1` on an unused
node.

## Run

Prepare a dedicated SLIME checkout and select the pinned commit:

```bash
git clone https://github.com/THUDM/slime.git /path/to/slime-poc
git -C /path/to/slime-poc checkout 0104a9e922ecd3cea768f7c2520e7d1a122afdc9
```

Run a configuration-only check first:

```bash
SLIME_ROOT=/path/to/slime-poc \
MODEL_PATH=/path/to/qwen3-vl-checkpoint \
VISUAL_TOOL_API_BASES=http://tool-host:9000 \
DRY_RUN=1 REQUIRE_IMAGES=0 \
bash scripts/run_visual_agent_slime_poc.sh
```

For a real run, use an unused Ray cluster:

```bash
SLIME_ROOT=/path/to/slime-poc \
MODEL_PATH=/path/to/qwen3-vl-checkpoint \
VISUAL_TOOL_API_BASES=http://tool-host:9000 \
RAY_DASHBOARD_ADDRESS=http://ray-head:8265 \
CONFIRM_DEDICATED_RESOURCES=1 \
bash scripts/run_visual_agent_slime_poc.sh
```

The explicit confirmation is intentional: a real PoC must not share Ray GPUs
or a visual-tool endpoint with an existing training run.

All topology values are parameters. `ACTOR_GPUS_PER_NODE`,
`ROLLOUT_NUM_GPUS`, `ROLLOUT_GPUS_PER_ENGINE`, and `COLOCATE` are intentionally
not fixed for the later layout comparison.

## CPU alignment tests

```bash
pytest -q tests/test_slime_visual_agent_fixed_trajectory.py \
  tests/test_prepare_visual_agent_slime_poc.py
```
