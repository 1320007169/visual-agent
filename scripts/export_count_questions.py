#!/usr/bin/env python3
"""Export the questions that invoked object_count from a VLMEval score file.

Reads a ``VisualAgent-vllm_<Dataset>_score.jsonl`` produced by the prompted-tool
evaluation and writes one record per question whose trace contains at least one
``object_count`` call. The last call of each question is treated as the tool
answer, matching the counting diagnostics reported for the five-tool run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def numeric_count(value: Any) -> int | None:
    if value is None:
        return None
    match = re.match(r"^-?\d+(?:\.\d+)?", str(value).strip())
    if match is None:
        return None
    number = float(match.group())
    return int(number) if number.is_integer() else None


def object_count_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        trace = json.loads(str(row.get("raw_response") or "{}"), strict=False)
    except (TypeError, ValueError):
        return []
    calls = trace.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict) and call.get("name") == "object_count"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_jsonl", help="Path to the VisualAgent score.jsonl")
    parser.add_argument("output_jsonl", help="Path to write the exported questions")
    args = parser.parse_args()

    source = Path(args.score_jsonl)
    destination = Path(args.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line, strict=False)
            calls = object_count_calls(row)
            if not calls:
                continue
            answer = row.get("answer")
            last = calls[-1]
            arguments = last.get("arguments") or {}
            result = last.get("result") or {}
            record = {
                "index": row.get("index"),
                "question": row.get("question"),
                "category": row.get("category"),
                "l2_category": row.get("l2-category"),
                "image_path": row.get("image_path"),
                "gt_count": numeric_count(row.get(answer)),
                "gt_answer": answer,
                "baseline_prediction": row.get("prediction"),
                "baseline_correct": row.get("prediction") == answer,
                "query": arguments.get("query"),
                "tool_count": result.get("count"),
                "queries": [(call.get("arguments") or {}).get("query") for call in calls],
                "tool_counts": [(call.get("result") or {}).get("count") for call in calls],
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    print(f"Exported {exported} object_count questions to {destination}")


if __name__ == "__main__":
    main()
