# Multi-tool RL training

The ModelArts entrypoint is
`scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh`.
Run it once on each training node with the same run token and node count.

## Corrected defaults

- `TOTAL_TRAINING_STEPS=null` derives the limit from the training dataloader
  and `TOTAL_EPOCHS`. The default 30,808-row dataset, batch 112, and one epoch
  produce 275 updates because the training loader drops the incomplete batch.
  An explicit step limit is capped by the available batches so final validation
  and checkpoint saving run on the last update.
- `VAL_BEFORE_TRAIN=True` records an initial validation baseline.
- `BEST_METRIC=val-core/visual-agent/acc/macro_mean` averages the per-source
  accuracy of the visual validation datasets with equal weight per source.
  This includes HRBench4K, raw depth, and TallyQA in the default validation set.
- Tool schemas retain nested array lengths and coordinate limits. Invalid
  arguments return a recoverable observation before any HTTP request.
  Deterministic HTTP errors such as 422 are not retried; 408, 429, and server
  errors retain retries. Business errors inside a successful HTTP response
  are recorded as failed tool calls.
- Checkpoint rotation protects the checkpoint just saved, even when the
  output directory contains a checkpoint with a higher step number.

## Monitor tool selection

Training and validation logs now include, for each data source:

- `train-tools/<source>/acc_mean` and `error_trajectory_rate`
- `train-tools/<source>/<tool>/trajectory_rate`, `calls`, and `errors`
- `train-tools/<source>/<tool>/success_rate` when at least one call occurred

Validation uses the `val-tools` prefix and saves tool traces alongside answers.
For counting, watch
`train-tools/visual-agent-tallyqa/object_count/trajectory_rate`.
A zero call rate is logged explicitly; no success rate is invented for zero calls.

These fixes do not restore tool selection in an already collapsed policy.
Reward still measures answer accuracy and format; calling a tool alone earns
no extra reward. Compare the initial and subsequent validation results, and
use an earlier checkpoint with useful count-tool behavior or the base model
for a recovery experiment rather than assuming a late collapsed checkpoint
will recover automatically.

## Launch and resume

For a fresh two-node run, execute this on both nodes:

```bash
TRAIN_RUN_TOKEN=multitool_schema_fix_v1 NNODES=2 \
  bash scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh
```

Use a new token for every fresh run. For an existing run, retain its output
paths, topology, and resume settings. If its `best_checkpoint.json` tracks
HRBench4K, also retain
`BEST_METRIC=val-core/visual-agent-hrbench4k/acc/mean@1`; the trainer rejects
changing the selection metric while resuming its saved best-model history.

The RL environment needs `jsonschema>=4.23.0`, declared in the RL requirements.
Schema changes alter the model prompt, so newly generated trajectories are
needed to measure their effect. Existing rollout files are historical results.

## OCR and Chart data version record (2026-10-01)

### Continuing the original 24-GPU run on 16 GPUs

Use `scripts/run_visual_agent_multitool_vlocr_2node_16gpu_modelarts.sh` on
both eight-GPU nodes. Its default source is the complete `global_step_60`
checkpoint in
`qwen3base_multitool_vlocr_n16_3node_resume_step44_20260930T190545597224_5a18e952`.
It resumes at step 61 with 14 training GPUs and two tool GPUs, using a new
output directory. The original data, model path (including KL reference),
tool schemas, prompt, global batch 126, PPO mini-batch 42 and rollout 16 stay
the same. Each training GPU processes 50% more trajectories; this is not a
promise of equal step time or memory use. Checkpoints are saved every ten
steps; after ten hours the current step is saved before exiting.

`ALLOW_FSDP_WORLD_SIZE_CHANGE=True` opts this entrypoint into loading the
21-rank DTensor model shards and raw Adam moment shards on 14 ranks. The
loader reads the source without modifying it, copies only each destination
rank's tensor intervals, and removes old flat-parameter padding before adding
the new padding. Model precision, Adam moments/counters, scheduler, global
step and dataloader state are retained. Destination rank N restores source
rank N's RNG state; changed rank assignment and reduction order mean the
continued run is not bitwise identical to a 24-GPU run. The next saved
checkpoint uses 14 ranks and can use the ordinary same-world-size loader.

This path supports FSDP1 with `use_orig_params=False`, a one-dimensional
`Shard(0)` model mesh, unchanged model/wrap policy, Adam/AdamW without AMSGrad,
and a reduction in rank count. It is disabled by default; existing launchers
retain their ordinary loading path. Increasing rank count, FSDP2, hybrid
sharding and changing the model/wrap policy are outside this conversion.

Validation uses real CPU FSDP collectives: three ranks save, two ranks restore,
and the next model/Adam update is compared with continuation on three ranks.
The default same-rank loader and saving/reloading the reduced-rank checkpoint
are also exercised in `tests/test_fsdp_reshard.py`. The actual Qwen3-VL step-60
checkpoint has 750 matching model entries and 64 optimizer parameter groups;
all 21 optimizer layouts matched the model's wrapping, with Adam step 180 and
scheduler step 60 at learning rate 1e-6. These checks do not constitute a
16-GPU Qwen3-VL forward/backward or end-to-end training validation.

### Dataset changes

This records the previous and new datasets for the eight-node VL-OCR run.
The new version is prepared for training; no new training or benchmark score
is reported here. The two-node entrypoint described above retains its old data.

| Item | Previous version | New version |
|---|---|---|
| Data directory under `data/` | `zwz_deepeyesv2_depth_tallyqa5k_multitool_20260924` | `zwz_deepeyesv2_depth_tallyqa5k_ocr_chart_multitool_20261001` |
| Training rows | 30,808 | 33,688 (+2,880) |
| Validation rows | 992 | 1,312 (+320) |
| Total rows | 31,800 | 35,000 (+3,200) |
| Run ID before job token | `qwen3base_multitool_vlocr_n16_8node` | `qwen3base_multitool_vlocr_ocr_chart1600_n16_8node` |
| Validation sources in macro accuracy | HRBench4K, raw depth, TallyQA | Previous three plus OCR and ChartQA |
| Updates at batch 336, one epoch, default step limit | 91 | 100 |

### Source distribution

| Source / ability | Previous train | Previous val | New train | New val |
|---|---:|---:|---:|---:|
| ZWZ spatial relation | 18,518 | 0 | 18,518 | 0 |
| DeepEyesV2 text reading | 1,000 | 0 | 1,000 | 0 |
| DeepEyesV2 attribute recognition | 2,000 | 0 | 2,000 | 0 |
| Raw depth / distance | 4,398 | 84 | 4,398 | 84 |
| TallyQA counting | 4,892 | 108 | 4,892 | 108 |
| HRBench4K single image | 0 | 400 | 0 | 400 |
| HRBench4K cross image | 0 | 400 | 0 | 400 |
| Added OCR (`visual-agent-ocr`) | 0 | 0 | 1,440 | 160 |
| Added Chart (`visual-agent-chartqa`) | 0 | 0 | 1,440 | 160 |
| **Total** | **30,808** | **992** | **33,688** | **1,312** |

The added samples come from `data/ocr_chart_rl_3200_20261001`, with the
following source quotas. All are original-image QA; neither source CoT nor
precomputed tool trajectories are used. New OCR/Chart rows have empty boxes.
Chart images can include the source dataset's rotation and mirror transforms.

| Added original source | Train | Val | Total |
|---|---:|---:|---:|
| TextVQA | 576 | 64 | 640 |
| DocVQA | 576 | 64 | 640 |
| SROIE | 144 | 16 | 160 |
| InfoVQA (`infographicsvqa`) | 144 | 16 | 160 |
| CodeVision-RL Chart (`train_group_1.parquet`, source split `train`, rule QA) | 1,440 | 160 | 1,600 |

OCR annotations originate from `min/Visual-CoT/metadata/*_cot_train.jsonl`;
their images reference the existing Visual-CoT image tree. Chart images are
stored in the selected 3,200-row dataset's `images/` directory. The merged
dataset preserves those paths, so both image locations must remain available
on every training node. Original mixed rows and their splits are retained,
including 46 existing duplicate image-path/question pairs in the old training
set. Added samples keep their 90%/10% split and image groups; only the merged
training order is shuffled, using seed `20261001`.
Both splits use the same Parquet schema, including the new `answer_aliases`
and `transform` columns. Empty validation boxes retain their empty values but
use the training split's numeric list type, allowing both files to be loaded
together by the dataset reader.

### Registration and scoring

- `ZwzDeepEyesV2Dataset` accepts `visual-agent-ocr` and `visual-agent-chartqa`.
  It passes their source, UID, split, and answer aliases to reward metadata.
- The reward dispatcher registers both sources with `visual_agent_thyme`.
  OCR accepts a normalized exact match against the canonical answer or aliases.
  Normalization ignores case, repeated whitespace, and boundary punctuation.
  This training rule does not implement TextVQA consensus accuracy or DocVQA /
  InfoVQA ANLS, and must not be reported as those benchmark scores.
- Chart accepts case-insensitive exact text or a numeric answer within 5% of
  the gold value. Percentages are converted to fractions; a zero gold value
  requires numeric equality. Non-finite numbers are rejected. Neither new
  source falls back to an LLM judge.
- Existing sources retain their scoring. The shared reward remains
  `0.9 * accuracy + 0.1 * format - grounding query penalty`, floored at zero.
  Calling a tool alone earns no extra reward.
- Both the repository eight-node ModelArts script and its copy at
  `/home/ma-user/work/algorithm/codebkp/run_visual_agent/run_visual_agent_multitool_vlocr_8node_64gpu_modelarts.sh`
  default to the new directory and run ID. Explicit `TRAIN_FILES`, `VAL_FILES`,
  and `MULTITOOL_DATA_DIR` overrides remain supported.

The base model, eight-node layout, batch 336, PPO mini-batch 112, rollout 16,
tool configuration, and system prompt retain the previous VL-OCR settings.
OCR and Chart tools were already enabled in that version; enabling a tool
does not itself add questions to the dataset. More data changes the number
of updates under the default one-epoch schedule, even with the same batch.

The eight-node VL-OCR launchers default to `SAVE_FREQ=10` and
`TRAINER_STOP_AFTER_SECONDS=36000`. The timer starts after initial validation.
Once ten hours have elapsed, the current training step finishes, its complete
checkpoint is saved even outside the regular ten-step interval, and training
exits normally. A checkpoint write that crosses the deadline also triggers
this exit. The final training step can end the run earlier; this is not a
hard job timeout. Model loading and initial validation are outside the timer.

Signal interruption is handled separately: SIGINT exits with code 130 and
SIGTERM with code 143. The common launcher runs cleanup once through its EXIT
trap and writes the same status to the node-0 completion file, so worker nodes
recognize interruption as a failure. Signal handling does not guarantee saving
an in-progress training step; the ten-hour trainer deadline uses the checkpoint
and normal-exit path described above.

### Checks and comparison limits

Dataset validation counts, image overlap checks, source totals, parent paths,
and output SHA-256 values are recorded in the new dataset's `summary.json`.
All 23,282 unique image paths were decoded successfully. Training and
validation share zero image paths and zero exact decoded RGB images.
The selected OCR samples were filtered against OCRBench by exact decoded RGB
pixels; Chart samples were filtered against ChartQA_TEST including rotations
and flips. That benchmark filtering covers the added samples only. It does
not establish that the original mixed data is benchmark-disjoint, and does
not detect recompressed or cropped near-duplicates.

The 160-row validation set per new task is a training holdout, not OCRBench
or ChartQA_TEST. The overall validation macro mean changes from three to
five equally weighted source groups, so old and new overall values are not
directly comparable. Compare unchanged sources separately and evaluate both
checkpoints on the same independent OCRBench / ChartQA protocol.

Registration and reward tests pass: 34 tests and 108 subtests. Configuration-only
execution resolves the requested external entrypoint to the new data and run
name. Parquet and JSONL rows match, and all parent rows are retained exactly,
including their duplicate multiplicities. The real VERL dataset reader loads
all 35,000 rows with the local Qwen tokenizer and VL-OCR tool schemas; sampling
and reward dispatch pass for all seven registered sources. That smoke check
uses tokenizer-only preprocessing and does not exercise image processing or
a full GPU rollout or training run.
