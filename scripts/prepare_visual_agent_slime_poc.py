#!/usr/bin/env python3
"""Convert existing Visual-Agent ZWZ parquet rows to SLIME VLM JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def convert_row(
    row: dict[str, Any],
    system_prompt: str,
    index: int,
    image_max_pixels: int | None = None,
) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    raw_solution = str(row.get("solution") or "").strip()
    data_source = str(row.get("data_source") or "visual-agent-zwz-relation").strip()
    solution = raw_solution.lower() if data_source == "visual-agent-zwz-relation" else raw_solution
    images = [str(value) for value in (row.get("images") or [])]
    if not question:
        raise ValueError(f"row {index} has no question")
    if not raw_solution:
        raise ValueError(f"row {index} has no solution")
    if not images:
        raise ValueError(f"row {index} has no images")
    image_content = []
    for image in images:
        image_spec: dict[str, Any] = {"type": "image", "image": image}
        if image_max_pixels is not None:
            image_spec["max_pixels"] = image_max_pixels
        image_content.append(image_spec)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [*image_content, {"type": "text", "text": question}],
            },
        ],
        "images": images,
        "solution": solution,
        "metadata": {
            "source": (
                "zwz_rl_vqa/original_images"
                if data_source == "visual-agent-zwz-relation"
                else data_source
            ),
            "source_name": data_source,
            "data_source": data_source,
            "ability": row.get("ability"),
            "split": row.get("split", "train"),
            "question": question,
            "answer": solution,
            "source_index": row.get("source_index", index),
            "source_image": row.get("source_image"),
            "bbox": row.get("bbox"),
        },
    }


def convert_rows(
    rows: Iterable[dict[str, Any]],
    system_prompt: str,
    image_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    return [
        convert_row(row, system_prompt, index, image_max_pixels)
        for index, row in enumerate(rows)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing ZWZ parquet file")
    parser.add_argument("--output", type=Path, required=True, help="New SLIME JSONL path")
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("prompts/visual_agent_rl_system.txt"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input parquet not found: {args.input}")
    if not args.system_prompt.is_file():
        raise SystemExit(f"system prompt not found: {args.system_prompt}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists (pass --force to replace it): {args.output}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.image_max_pixels is not None and args.image_max_pixels < 1:
        raise SystemExit("--image-max-pixels must be positive")

    import pyarrow.parquet as pq

    rows = pq.read_table(args.input).to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    converted = convert_rows(rows, system_prompt, args.image_max_pixels)
    if args.require_images:
        missing = [image for row in converted for image in row["images"] if not Path(image).is_file()]
        if missing:
            raise SystemExit(f"missing {len(missing)} images; first: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in converted:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": len(converted)}, indent=2))


if __name__ == "__main__":
    main()
