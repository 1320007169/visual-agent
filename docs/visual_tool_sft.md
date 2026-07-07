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

Prepare full training without starting it automatically:
Do not run this on a machine without enough GPU memory.

```bash
bash scripts/train_visual_tool_sft_full.sh
```

## Current Validation Summary

`data/visual_tool_sft_v0.jsonl`:

- total: 2187
- by task: counting 1789, spatial 265, attribute 133
- by tool: grounding_detect 811, sam3_segment_multi 1243, sam3_crop_zoom 129, sam3_crop_zoom_multi 4
- multi-image count: 133
- invalid count: 0

One source uid appears in both spatial and attribute data. The converter keeps both rows, preserves the original value in `source_uid`, and prefixes the second training `uid` to keep the output unique.
