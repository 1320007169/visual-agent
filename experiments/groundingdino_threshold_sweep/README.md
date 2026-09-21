# GroundingDINO empty-detection threshold sweep

This experiment extracts successful-but-empty `grounding_detect` calls from a
VStarBench evaluation result, runs one GroundingDINO forward pass per call, and
reapplies post-processing over a grid of box and text thresholds.

It writes an immutable manifest, detailed JSONL, summary CSV/JSON, threshold
contact sheets, and `review.html`. An output directory must not already exist.

```bash
/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/conda_envs/visual-tools/bin/python \
  run_vstar_empty_sweep.py \
  --result-jsonl /path/to/VStarBench_exact_matching_result.jsonl \
  --model /home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-tools/grounding-dino-base-transformers \
  --output-dir outputs/vstar_empty_YYYYMMDD_HHMMSS \
  --device cuda:0
```
