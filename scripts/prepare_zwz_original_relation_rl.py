#!/usr/bin/env python3
"""Prepare original-image spatial-relation RL data from zwz_rl_vqa."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ALLOWED_ANSWERS = {
    "left of",
    "right of",
    "above",
    "below",
    "overlap",
    "inside",
    "contain",
    "in front of",
    "behind",
    "next to",
}


def normalize_question(value: object) -> str:
    text = str(value or "").replace("<image>", "", 1).strip()
    return re.sub(r"[ \t]+", " ", text)


def image_split(image_path: str, val_percent: int) -> str:
    bucket = int(hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/zwz_rl_vqa/train1.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/zwz_rl_vqa/rl_original_relation"))
    parser.add_argument("--val-percent", type=int, default=2)
    parser.add_argument("--smoke-train", type=int, default=14)
    parser.add_argument("--smoke-val", type=int, default=2)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.val_percent <= 50:
        raise SystemExit("--val-percent must be between 1 and 50")
    if not args.input.is_file():
        raise SystemExit(f"input parquet not found: {args.input}")

    source_root = args.input.parent.resolve()
    rows = pq.read_table(args.input).to_pylist()
    prepared: list[dict] = []
    seen: set[tuple] = set()
    missing_images: set[str] = set()
    conflicts: dict[tuple, set[str]] = {}

    for source_index, row in enumerate(rows):
        answer = str(row.get("answer") or "").strip().lower()
        if answer not in ALLOWED_ANSWERS:
            continue

        original_images = row.get("original_images") or []
        bbox = row.get("bbox") or []
        question = normalize_question(row.get("problem"))
        if len(original_images) != 1 or len(bbox) != 4 or not question:
            continue

        image_path = (source_root / str(original_images[0])).resolve()
        if not image_path.is_file():
            missing_images.add(str(image_path))
        if args.require_images and not image_path.is_file():
            continue

        bbox_key = tuple(float(value) for value in bbox)
        example_key = (str(image_path), bbox_key, question, answer)
        if example_key in seen:
            continue
        seen.add(example_key)

        conflict_key = (str(image_path), bbox_key, question)
        conflicts.setdefault(conflict_key, set()).add(answer)
        prepared.append(
            {
                "images": [str(image_path)],
                "question": question,
                "solution": answer,
                "bbox": list(bbox_key),
                "source_index": source_index,
                "source_image": str(original_images[0]),
                "data_source": "visual-agent-zwz-relation",
                "ability": "spatial_relation",
            }
        )

    conflicting_keys = {key for key, answers in conflicts.items() if len(answers) > 1}
    if conflicting_keys:
        prepared = [
            row
            for row in prepared
            if (row["images"][0], tuple(row["bbox"]), row["question"]) not in conflicting_keys
        ]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in prepared:
        target = val_rows if image_split(row["images"][0], args.val_percent) == "val" else train_rows
        target.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, values: list[dict]) -> None:
        if not values:
            raise SystemExit(f"no rows available for {name}")
        pq.write_table(pa.Table.from_pylist(values), args.output_dir / name, compression="zstd")

    write("train.parquet", train_rows)
    write("val.parquet", val_rows)
    write("smoke_train.parquet", train_rows[: args.smoke_train])
    write("smoke_val.parquet", val_rows[: args.smoke_val])

    summary = {
        "source": str(args.input.resolve()),
        "allowed_answers": sorted(ALLOWED_ANSWERS),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unique_images": len({row["images"][0] for row in prepared}),
        "conflicting_keys_removed": len(conflicting_keys),
        "missing_image_files": len(missing_images),
        "images_required": args.require_images,
        "smoke_train_rows": min(args.smoke_train, len(train_rows)),
        "smoke_val_rows": min(args.smoke_val, len(val_rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
