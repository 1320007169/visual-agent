#!/usr/bin/env python3
"""Curate camera-relative depth questions from the raw CA-VQA RL seed."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_QUESTION = re.compile(
    r"^Which object is closer to the camera taking this photo, the "
    r"(?P<object_a>.+?) \[(?P<box_a>\d+, \d+, \d+, \d+)\] or the "
    r"(?P<object_b>.+?) \[(?P<box_b>\d+, \d+, \d+, \d+)\]\?$"
)


def image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected PNG reference image: {path}")
    return struct.unpack(">II", header[16:24])


def relative_box(raw: str, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = (int(value) for value in raw.split(", "))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("invalid_box")
    if min((x2 - x1) / width, (y2 - y1) / height) < 0.03:
        raise ValueError("small_box")
    if (x2 - x1) * (y2 - y1) / (width * height) < 0.0025:
        raise ValueError("small_box")
    return [round(1000 * x1 / width), round(1000 * y1 / height),
            round(1000 * x2 / width), round(1000 * y2 / height)]


def curate(row: dict) -> dict:
    question = str(row["question"]).splitlines()[0]
    match = CAMERA_QUESTION.fullmatch(question)
    if row.get("source") != "ca_vqa_multichoice" or match is None:
        raise ValueError("not_camera_comparison")
    if row.get("solution") not in {"A", "B"}:
        raise ValueError("invalid_answer")
    images = row.get("images") or []
    if not images:
        raise ValueError("missing_image")
    reference = Path(images[-1])
    if not reference.is_file():
        raise ValueError("missing_image")
    width, height = image_size(reference)
    box_a = relative_box(match["box_a"], width, height)
    box_b = relative_box(match["box_b"], width, height)
    object_a, object_b = match["object_a"], match["object_b"]
    return {
        "images": [str(reference)],
        "question": (
            "Which object is closer to the camera in image 0?\n"
            f"A. {object_a}\nB. {object_b}\n"
            "Answer with A or B."
        ),
        "solution": row["solution"],
        "uid": row["uid"],
        "source": row["source"],
        "data_source": "visual-agent-depth-camera",
        "ability": "camera_depth_compare",
        "answer_type": "multiple_choice",
        "object_a": object_a,
        "object_b": object_b,
        "bbox_a": box_a,
        "bbox_b": box_b,
        "reference_image": str(reference),
        "source_question": row["question"],
        "source_image_count": len(images),
    }


def split_for(reference_image: str, val_percent: int) -> str:
    bucket = int(hashlib.sha256(reference_image.encode()).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def prepare(input_path: Path, output_dir: Path, val_percent: int = 10) -> dict:
    if not 1 <= val_percent <= 50:
        raise ValueError("val_percent must be between 1 and 50")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    excluded = Counter()
    seen_uids = set()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            try:
                item = curate(row)
            except ValueError as exc:
                excluded[str(exc)] += 1
                continue
            if item["uid"] in seen_uids:
                raise ValueError(f"Duplicate UID: {item['uid']}")
            seen_uids.add(item["uid"])
            split = split_for(item["reference_image"], val_percent)
            item["split"] = split
            splits[split].append(item)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both training and validation splits must contain examples")
    output_dir.mkdir(parents=True)
    for name, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{name}.parquet", compression="zstd")
    summary = {
        "input": str(input_path.resolve()),
        "selection": "CA-VQA camera-relative A/B; source boxes are hidden audit fields",
        "val_percent": val_percent,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "train_scenes": len({r["reference_image"] for r in splits["train"]}),
        "val_scenes": len({r["reference_image"] for r in splits["val"]}),
        "answers": dict(Counter(r["solution"] for rows in splits.values() for r in rows)),
        "excluded": dict(excluded),
        "images_per_sample": 1,
        "contains_tool_trajectories": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-percent", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.val_percent), indent=2))


if __name__ == "__main__":
    main()
