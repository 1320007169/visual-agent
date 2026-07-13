# Visual Tool SFT Pipeline

This repo-local pipeline converts naturalized visual-tool trajectories into an offline SFT dataset for the existing DeepEyesV2/LLaMA-Factory cold-start path.

## Inputs

- `/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_counting_sam3_dino_trajectories.natural.jsonl`
- `/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_spatial_sam3_trajectories.natural.jsonl`
- `/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_attribute_sam3_crop_zoom_trajectories.natural.jsonl`

The converter does not modify these source files.

## Outputs

- Full SFT data: `data/visual_tool_sft_v0.jsonl`
- Smoke SFT data: `data/visual_tool_sft_v0.smoke.jsonl`
- Smoke config: `configs/train_visual_tool_sft_smoke.yaml`
- Full config: `configs/train_visual_tool_sft_full.yaml`

The training rows use LLaMA-Factory ShareGPT formatting with `messages` and `images`. Every training row is normalized to the strict `user, assistant, user, assistant` role pattern expected by LLaMA-Factory SFT. Tool observations are represented as user-side `<tool_response>...</tool_response>` content because the current cold-start `dataset_info.json` only declares `user`, `assistant`, and `system` roles. The original roles are preserved in `messages_raw` metadata.

## Supported Visual Tools

- `sam3_segment_multi`
- `grounding_detect`
- `sam3_crop_zoom`
- `sam3_crop_zoom_multi`

The first stage is offline SFT only. The visual tools are registered as schema-only tools in `reinforcement_learning/verl/tools/visual_tool.py` and `reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_config.yaml`; they intentionally do not call SAM3 or GroundingDINO.

## Commands

Generate full and smoke datasets:

```bash
python scripts/convert_visual_tool_sft.py
```

Validate full data:

```bash
python scripts/validate_visual_tool_sft.py --input data/visual_tool_sft_v0.jsonl
```

Validate smoke data:

```bash
python scripts/validate_visual_tool_sft.py --input data/visual_tool_sft_v0.smoke.jsonl
```

Run smoke training:

```bash
bash scripts/train_visual_tool_sft_smoke.sh
```

Use a specific LLaMA-Factory CLI, for example from a prepared conda environment:

```bash
LLAMAFACTORY_CLI=/path/to/bin/llamafactory-cli bash scripts/train_visual_tool_sft_smoke.sh
```

Override the base model path:

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct bash scripts/train_visual_tool_sft_smoke.sh
```

## Distributed Full SFT

The production config follows the original DeepEyesV2 cold-start strategy: full-parameter language-model SFT while the vision tower and multimodal projector remain frozen. It uses BF16, DeepSpeed ZeRO-3, gradient checkpointing, per-device batch size 1, learning rate `1e-5`, and global batch size 32.

For 8, 16, and 32 GPUs, the launcher automatically selects gradient accumulation 4, 2, and 1 respectively.

### Huawei platform: current 8-GPU entry point

The default persistent work root is:

```text
/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx
```

Like the supplied Huawei platform script, the entry point derives absolute paths from that base directory. By default it expects:

- checkout: `$PLATFORM_WORK_ROOT/visual-agent`
- model: `$PLATFORM_WORK_ROOT/models/Qwen3-VL-8B-Instruct`
- Conda environment: `$PLATFORM_WORK_ROOT/envs/llamafactory`
- logs: `$PLATFORM_WORK_ROOT/logs/visual-tool-sft`

The shell creates the same missing Huawei dataset, `synaflow_wl` algorithm, and model-storage symlinks used by the supplied platform reference, initializes the platform CUDA 11.8 installation, activates the Conda environment, prints hardware and topology information, and starts eight processes:

```bash
bash scripts/run_visual_tool_sft.sh
```

The entry script can be copied to any directory and run directly because it does not derive the checkout from its own location:

```bash
bash /other/directory/run_visual_tool_sft.sh
```

If the checkout is not at the default location, set `ROOT_DIR` to its absolute path.

Paths can be overridden without editing the script:

```bash
MODEL_NAME_OR_PATH=/shared/models/Qwen3-VL-8B-Instruct \
CONDA_ENV_PATH=/shared/envs/llamafactory \
bash scripts/run_visual_tool_sft.sh
```

For a five-step cluster smoke run, use a separate output directory:

```bash
MAX_STEPS=5 \
OUTPUT_DIR=saves/visual_tool_sft_v0/smoke/full-sft \
bash scripts/run_visual_tool_sft.sh
```

Set `PLATFORM_SETUP=0` to skip Huawei symlink and CUDA initialization. Set `CONDA_ENV_PATH=` explicitly to use the already-active shell environment. Set `OFFLINE_MODE=1` when the model, dataset, and caches are all local.

The repository does not pin a LLaMA-Factory version. The config explicitly sets `disable_gradient_checkpointing: false` so full SFT keeps gradient checkpointing enabled. If the installed platform CLI reports this as an unknown argument, remove that one line from `configs/train_visual_tool_sft_full.yaml`; the original cold-start config already relies on LLaMA-Factory's enabled-by-default behavior.

### Direct launcher and dry run

Inside an existing eight-GPU allocation:

```bash
NNODES=1 NPROC_PER_NODE=8 \
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-8B-Instruct \
bash scripts/train_visual_tool_sft_full.sh
```

Inspect the resolved topology and generated YAML without starting workers:

```bash
DRY_RUN=1 NNODES=1 NPROC_PER_NODE=8 \
MODEL_NAME_OR_PATH=/path/to/Qwen3-VL-8B-Instruct \
bash scripts/train_visual_tool_sft_full.sh
```

### Slurm scaling

The Slurm wrapper requests one task and eight GPUs per node. It uses `SLURM_SUBMIT_DIR` to locate the checkout, so run these commands from the repository root. Site-specific partition and account values should be passed to `sbatch`:

```bash
# 1 node x 8 GPUs
sbatch scripts/slurm/train_visual_tool_sft.sbatch

# Future 2 nodes x 8 GPUs and 4 nodes x 8 GPUs
sbatch --nodes=2 scripts/slurm/train_visual_tool_sft.sbatch
sbatch --nodes=4 scripts/slurm/train_visual_tool_sft.sbatch
```

All nodes must see the repository, model, dataset JSONL, referenced images, and output directory at identical absolute paths. The generated `data/visual_tool_sft_v0*.jsonl` files are gitignored and must be generated or copied onto the platform. Their `images` fields may still contain `/data_cinema/...` paths; those paths must exist on every training node, or the JSONL files must be rewritten consistently after upload.

Useful runtime overrides are `MODEL_NAME_OR_PATH`, `OUTPUT_DIR`, `LLAMAFACTORY_CLI`, `CONFIG_PATH`, `DEEPSPEED_CONFIG`, `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `TARGET_GLOBAL_BATCH_SIZE`, `RESUME_FROM_CHECKPOINT`, `MAX_STEPS`, and `DRY_RUN`.

## Current Validation Summary

`data/visual_tool_sft_v0.jsonl`:

- total: 2187
- by task: counting 1789, spatial 265, attribute 133
- by tool: grounding_detect 811, sam3_segment_multi 1243, sam3_crop_zoom 129, sam3_crop_zoom_multi 4
- multi-image count: 133
- invalid count: 0

One source uid appears in both spatial and attribute data. The converter keeps both rows, preserves the original value in `source_uid`, and prefixes the second training `uid` to keep the output unique.
