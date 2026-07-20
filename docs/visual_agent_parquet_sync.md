# Visual Agent parquet synchronization

The cumulative dataset is `albert13200/Visual_Agent_Parquet`. Download it through
the mirror and keep the snapshot separate from the existing JSONL data:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  conda run -n base python scripts/update_visual_agent_parquet.py \
  --output-dir data/visual_agent_parquet_snapshot
```

The snapshot contains separate `samples/*.parquet` and `images/*.parquet` files;
they must not be concatenated because their schemas are different. The script
writes `sync_summary.json` with the overlap and newly added UID counts. Re-run the
same command whenever the dataset is updated. For the current snapshot, 1,492 UIDs
are new relative to `data/visual_tool_sft_v0.jsonl`; 2,186 are shared and one old
local UID is absent from the remote snapshot.
