#!/usr/bin/env python3
"""Update the cumulative Visual Agent parquet snapshot and report UID changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ID = "albert13200/Visual_Agent_Parquet"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def jsonl_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {str(row["uid"]) for line in handle if line.strip() for row in [json.loads(line)]}


def parquet_uids(root: Path) -> tuple[set[str], int]:
    import pyarrow.parquet as pq

    uids: set[str] = set()
    rows = 0
    for path in sorted((root / "samples").rglob("*.parquet")):
        table = pq.read_table(path, columns=["uid"])
        values = [str(value) for value in table.column("uid").to_pylist()]
        rows += len(values)
        uids.update(values)
    return uids, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/visual_agent_parquet"))
    parser.add_argument("--existing-jsonl", type=Path, default=Path("data/visual_tool_sft_v0.jsonl"))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=["*.parquet"],
        max_workers=args.max_workers,
    ))
    local = jsonl_uids(args.existing_jsonl)
    remote, remote_rows = parquet_uids(snapshot)
    summary = {
        "repo_id": REPO_ID,
        "endpoint": args.endpoint,
        "snapshot_dir": str(snapshot),
        "local_jsonl_rows": len(local),
        "remote_sample_rows": remote_rows,
        "remote_unique_uids": len(remote),
        "overlap_uids": len(local & remote),
        "new_remote_uids": len(remote - local),
        "local_only_uids": len(local - remote),
        "local_only_uid_values": sorted(local - remote),
    }
    (snapshot / "sync_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
