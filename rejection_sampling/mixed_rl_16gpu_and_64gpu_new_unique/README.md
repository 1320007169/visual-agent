# Mixed RL new-trajectory rejection sampling

This directory judges newly successful tool trajectories from these two RL runs:

- `zwz_deepeyesv2_3k_nocount_v1_n8_2node`
- `zwz_deepeyesv2_3k_nocount_v1_n8_b224_8node_scratch`

`prepare_candidates.py` keeps rows with `acc=1`, `format=1`, and a successful tool trajectory, deduplicates them by `(data_source, source_image, source_index)`, and excludes relation samples already accepted by the earlier DeepSeek judge. It preserves original assistant turns. Only the original source image is included for judging; crop images are neither reconstructed nor uploaded.

The exact rollout files and step ranges used are frozen in `data/prepare_report.json`. The judge is resumable and writes accepted, rejected, and review partitions under `outputs/deepseek_v41`.

```bash
python prepare_candidates.py
bash run_deepseek_v41_judge.sh --dry-run
nohup bash run_deepseek_v41_judge.sh > outputs/deepseek_v41/full_run.log 2>&1 &
```
