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
