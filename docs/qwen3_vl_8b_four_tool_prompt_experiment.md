# Qwen3-VL-8B Five-Tool Prompt Experiment

The historical filename retains `four_tool` for launcher compatibility. The current experiment exposes five model-visible tools:

- `crop_zoom`: inspect a selected image region at higher effective resolution.
- `grounding_detect`: detect and localize concrete objects.
- `object_count`: count repeated concrete objects.
- `ocr_read`: recognize visible text.
- `ground_depth`: locate an object and estimate its median metric depth.

`depth_measure` and all SAM tools are excluded. `ground_depth` performs localization inside the server, so the model does not need a separate box-depth API.

## Comparison

The entrypoint runs the five-tool treatment on the original Qwen3-VL-8B-Instruct checkpoint. Existing direct-answer Qwen3-VL-8B results provide the low-cost baseline. This avoids rerunning an already evaluated checkpoint, but it is not a perfectly synchronized A/B if evaluator code, cached samples, or failure rates differ. A publication-quality comparison should rerun the direct baseline with the same commit, dataset cache, and retry settings.

The treatment uses temperature 0, at most six assistant turns, and 6,144 generated tokens per turn. Default datasets are `VStarBench`, `HRBench4K`, `HRBench8K`, `OCRBench`, `MME-RealWorld-Lite`, and `MME-RealWorld-CN`. They cover general perception, localization, counting, text, and real-world depth cues. `ChartQA_TEST` remains optional because its normal scoring path needs a judge API key.

## Resources

The ModelArts wrapper starts every service on one eight-GPU node:

| GPU | Service |
|---:|---|
| 0 | GroundingDINO and the Visual-Agent bridge |
| 1 | PaddleOCR-VL and Depth Anything 3 |
| 2 | CountGD++ |
| 3-7 | Five Qwen3-VL-8B vLLM replicas |

The wrapper reads environment and model locations from `groundingdino_offline_pipeline/.env`, starts the three isolated VTS services, waits for their health endpoints, then starts the existing Visual-Agent evaluator. The bridge directory is shared locally through `VTS_TOOL_BRIDGE_ROOT`.

## Run

Validate the checkpoint, prompt, environment, and dataset paths without allocating models:

```bash
FIVE_TOOL_CONFIG_ONLY=1 \
bash scripts/run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh
```

Run the treatment:

```bash
bash scripts/run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh
```

The older `FOUR_TOOL_*` environment names remain accepted as aliases, but new jobs should use `FIVE_TOOL_MODEL_PATH`, `FIVE_TOOL_PROMPT_FILE`, `FIVE_TOOL_RUN_ID`, and `FIVE_TOOL_OUTPUT_ROOT`.

## Readout

VLMEval writes each benchmark's normal score artifact. The wrapper additionally writes:

- `protocol.json`: checkpoint, prompt hash, five-tool allowlist, datasets, and budgets.
- `status.tsv`: exit code, elapsed time, and result directory.
- `behavior_summary.json`: API failures, empty predictions, tool-use rate, per-tool calls, success rate, mean turns, and unexpected tool names.
- `tool_services/`: VTS logs and PID records for diagnosis; these remain under ignored experiment outputs.

Compare benchmark scores with the recorded direct baseline only after confirming equal sample counts and reporting API failures and empty predictions. Interpret score changes with tool behavior: near-zero tool use indicates prompt/context effects, while frequent calls with low success indicate a service or routing problem.
