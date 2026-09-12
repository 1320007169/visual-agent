# Spatial Relation RL Recovery

## Current Scope

This workflow prepares existing RL trajectories for review and possible SFT.
It does not generate new answers, call a judge model, or start training.

Answer status is separate from semantic correctness. `label_match` means a
complete answer agrees with the dataset label after explicit alias normalization.
`different_label` and `unmapped_answer` require visual review; neither proves a
semantic error. Bare `on`, `in`, and `over` are not automatically mapped, rewritten
to the gold answer, or classified as semantically wrong. All `semantic_verdict`
fields remain null until review.

An earlier local reward-code change also introduced deterministic matching for
ZWZ relation training. That code has not been used in a new training run or
validated for semantic performance. This offline workflow only shares its label
normalizer; it does not establish that replacing the training judge is beneficial.

The rollout exporter now records stable source identity in `source_metadata`.
Older dumps without an ID are accepted by the SFT converter only when the
question identifies exactly one source row before looking at the prediction.

## Generated Artifacts

The recovery run writes a new directory, leaving the old distillation data intact:

- `outputs/rl_rollout_recovery_20260912_v2/audit.json`: label agreement totals, per-step
  counts, rejection reasons and example locations, selected labels and review flags.
- `outputs/rl_rollout_recovery_20260912_v2/candidates.jsonl`: one selected candidate
  per source, with the historical system prompt, canonical tool calls and answer,
  original image and reconstructed crop paths, and provenance in `metadata`.
- `outputs/rl_rollout_recovery_20260912_v2/candidates_crops/`: reconstructed crop images.
- `outputs/rl_rollout_recovery_20260912_v2/semantic_review.jsonl`: representative
  raw trajectories deduplicated by source ID and raw answer, with occurrence counts.
- `outputs/rl_rollout_recovery_20260912_v2/semantic_review_sample_200.jsonl`: up to
  200 different sources, stratified by gold label, answer status, `on` versus
  other answers, and early/late steps. This is a diagnostic sample, not a random
  estimate of semantic accuracy.
- `outputs/rl_rollout_recovery_20260912_v2/validation.json`: structural and source
  integrity checks, including isolation from the held-out validation images.

The older `outputs/rl_rollout_recovery_20260912/` directory is an interrupted,
incomplete export. Use the v2 artifacts only after both reports confirm completion.

The completed v2 run scanned 295,680 trajectories from 165 steps. It exported
9,153 candidates from 5,925 original images and reconstructed 1,648 crop images.
Four additional selected candidates were excluded because reconstructed crop
dimensions differed from the recorded dimensions by one pixel. The semantic
review queue contains 12,556 distinct source-answer pairs and a 200-source sample.
All candidates passed structural/source/validation-isolation checks, all crop
files passed readability/dimension checks, and 20 sampled reconstructions
(including nine nested crops) matched the generated JPEG bytes exactly.

This pool remains imbalanced: 71.3% of answers are left/right relations and only
69 candidates answer directly without tools. These counts describe candidates,
not a finalized SFT mixture or measured model-quality improvement.

These are automatically filtered candidates, not visually verified teachers.
Empty detections followed by a crop may be valid recovery. Multiple returned
boxes require checking that the intended object was selected. The converter does
not prove tool necessity, target identity, visual label correctness, or causal
use of observations. It preserves the original final explanation in metadata
for review while exporting only the final answer required by the system prompt.

Run from the repository root. The new export records absolute crop-image paths.

```bash
../conda_envs/deepeyes-rl-conda/bin/python scripts/convert_rl_rollouts_to_sft.py \
  --audit-only \
  --report-json outputs/rl_rollout_recovery_next/audit.json \
  --semantic-review-jsonl outputs/rl_rollout_recovery_next/semantic_review.jsonl \
  --review-sample-jsonl outputs/rl_rollout_recovery_next/semantic_review_sample_200.jsonl
```

To create another candidate dataset, remove `--audit-only` and add
`--output outputs/rl_rollout_recovery_next/candidates.jsonl`. Existing output,
crop, and report files are not overwritten. `--min-step` and `--max-step` can
limit the source checkpoints. Rejection counts report the first failing check
for each trajectory, so they are mutually exclusive, not all detected faults.

## Next Experiment

1. Review a stratified sample of candidates using the original image and every
   crop. Include all relation labels, direct answers, multiple-box cases, and
   empty-detection recovery. Check object identity, direction, evidence and the
   original final explanation. Reject unsupported or contradictory trajectories.
2. Evaluate the original Instruct model and the preserved RL steps 130 and 165
   under identical prompts, image resolution, decoding and tool settings. Use
   label-agreement scoring on `data/zwz_rl_vqa/rl_original_relation/val.parquet`,
   and report separately reviewed semantic correctness for disputed expressions.
   Compare tools enabled and disabled; report accuracy, macro accuracy, calls per
   question and invalid-call rate. The existing validation set has 378 questions
   from 189 images and is image-disjoint from training, but rare labels have few
   examples. Training-batch scores do not select the best checkpoint.
3. Train a pilot from the original Instruct model on about 2,000 reviewed samples.
   Preserve reliable direct-answer examples and image-disjoint validation. A
   starting setting is frozen vision/projector, learning rate `2e-6`, global
   batch 32, at most one epoch, with midpoint and final evaluation. This is an
   experimental setting, not a validated optimum. Do not add low-quality direct
   examples merely to reach a target ratio.
4. Expand SFT or resume RL only after the pilot improves the fixed evaluation.
   Further RL should validate tool schemas at execution and evaluate periodically;
   label matching alone does not enforce useful queries or tool necessity.

## Verification

```bash
../conda_envs/deepeyes-rl-conda/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_convert_rl_rollouts_to_sft.py \
  tests/test_validate_rl_sft_candidates.py \
  tests/test_visual_agent_thyme_reward.py \
  tests/test_rollout_source_metadata.py \
  reinforcement_learning/tests/utils/reward_score/test_visual_agent_thyme_on_cpu.py
```

```bash
../conda_envs/deepeyes-rl-conda/bin/python scripts/validate_rl_sft_candidates.py \
  --input outputs/rl_rollout_recovery_20260912_v2/candidates.jsonl \
  --train-parquet data/zwz_rl_vqa/rl_original_relation/train.parquet \
  --val-parquet data/zwz_rl_vqa/rl_original_relation/val.parquet \
  --report-json outputs/rl_rollout_recovery_20260912_v2/validation.json
```

The recovery work does not launch model inference, distributed RL or SFT training.
