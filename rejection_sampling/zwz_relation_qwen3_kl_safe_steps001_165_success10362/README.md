# DeepSeek V4.1 rejection sampling

This batch is the 10,362 successful SFT trajectories converted from the complete
Qwen3 GroundingDINO KL-safe RL run (`1.jsonl` through `165.jsonl`).

Source rollout directory:

```text
/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts/visual-agent-zwz-rl/zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual
```

Input dataset:

```text
/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/data/rl_distill/qwen3_groundingdino_kl_safe_success_sft.jsonl
```

The judge uses the official DeepSeek API (`https://api.deepseek.com`) and the
official V4.1 Flash model alias (`deepseek-flash`). It sends the trajectory
images with detector/crop boxes drawn on them. The source JSONL is never
modified.

Validate one request locally without using the API:

```bash
bash run_deepseek_v41_judge.sh --dry-run
```

Run the default 300-sample calibration batch:

```bash
bash run_deepseek_v41_judge.sh
```

Resume and judge all remaining samples after calibration:

```bash
LIMIT=0 bash run_deepseek_v41_judge.sh
```

Results are written under `outputs/deepseek_v41/`:

- `decisions.jsonl`: append-only structured judge decisions
- `accepted.jsonl`: accepted SFT samples
- `review.jsonl`: uncertain samples requiring review
- `rejected.jsonl`: rejected samples
- `errors.jsonl`: retryable request or parsing failures
- `summary.json`: current totals

Existing decisions are detected by a stable sample ID, so rerunning the same
command resumes instead of submitting duplicate API requests.
