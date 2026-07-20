#!/usr/bin/env python3
"""Convert Visual_Agent_Parquet sample shards to JSONL without deduplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from convert_visual_tool_sft import convert_item


def find_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {path}")
    return files


def iter_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        if "record_json" not in parquet.schema_arrow.names:
            raise ValueError(f"{path}: missing required record_json column")
        for batch in parquet.iter_batches(columns=["record_json"]):
            for raw_record in batch.column(0).to_pylist():
                if not raw_record:
                    raise ValueError(f"{path}: empty record_json value")
                record = json.loads(raw_record)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: record_json is not a JSON object")
                task_type = str(record.get("task_type") or record.get("data_type") or "unknown")
                yield convert_item(record, task_type=task_type, uid=record.get("uid"))


def convert(input_path: Path, output_path: Path) -> int:
    sources = find_parquet_files(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_records(sources):
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/visual_agent_parquet_snapshot/samples"),
        help="Sample parquet file or directory (not the images parquet directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visual_agent_parquet.jsonl"),
    )
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": count}, indent=2))


if __name__ == "__main__":
    main()
