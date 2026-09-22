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
| 1 | PaddleX PP-OCRv5 Server and Depth Anything 3 |
| 2 | CountGD++ |
| 3-7 | Five Qwen3-VL-8B vLLM replicas |

The wrapper reads environment and model locations from `groundingdino_offline_pipeline/.env`, starts the three isolated VTS services, waits for their health endpoints, then starts the existing Visual-Agent evaluator. The bridge directory is shared locally through `VTS_TOOL_BRIDGE_ROOT`.
OCR uses `PP-OCRv5_server_det`, `PP-OCRv5_server_rec`, and
`PP-LCNet_x1_0_textline_ori` as configured in
`groundingdino_offline_pipeline/configs/tools/paddlex_ocrv5_server.yaml`.

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

## Results (2026-09-21)

The completed run is
`outputs/vlmeval/five_tool_prompt/qwen3_vl_8b_five_tool_prompt_20260921_200256/`.
Its `status.tsv` reports exit code 0 after 9,897 seconds; `protocol.json` records
the Qwen3-VL-8B-Instruct checkpoint, prompt hash, six datasets, and six-turn
budget. Scores are from the run's own score CSV/JSON files. VStar uses `Overall`,
HRBench uses `Average / all`, and OCRBench reports the score out of 1,000.

| Benchmark | Five-tool prompt | Samples | API failed | Tool-use rate |
|---|---:|---:|---:|---:|
| VStarBench | 73.30% | 191 | 0 | 40.8% |
| HRBench4K | 76.88% | 800 | 0 | 54.4% |
| HRBench8K | 70.00% | 800 | 0 | 51.9% |
| OCRBench | 835/1000 (83.50%) | 1,000 | 0 | 19.5% |
| MME-RealWorld-Lite | 44.29% | 1,919 | 0 | 12.0% |
| MME-RealWorld-CN | 58.68% | 5,917 | 1 | 61.0% |

The six distinct workbooks contain 2,489 `grounding_detect`, 2,403
`crop_zoom`, 2,205 `ocr_read`, 582 `object_count`, and 12 `ground_depth` calls.
There were 60 recoverable OCR `no_text` responses. Another 42 tool requests
failed, all on MME-RealWorld-CN: 41 out-of-range `target_image` references and
one unsupported `slack_ratio` argument to `grounding_detect`. The one CN
`api_failed` prediction is separate from those tool errors. The generated
`behavior_summary.json` counts duplicate copies of some workbooks (including
`_score` variants), so its `mode_totals` should not be treated as a count of
distinct calls or samples; the counts above use one workbook per dataset.

For context, the previously recorded direct-answer Qwen3 base scored
84.29% / 76.38% / 70.62% on VStar / HR4K / HR8K. The five-tool prompt is
10.99 percentage points lower on VStar, 0.50 higher on HR4K, and 0.62 lower
on HR8K. This is an indicative prompt-and-tool comparison, not an isolated
measurement of tool utility: the prompt, turns, and evaluator run differ.
