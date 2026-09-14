#!/usr/bin/env python3
"""Prepare original-image multiple-choice RL data from Vision-OPD-6K."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE = "visual-agent-vision-opd"
ABILITY = "fine_grained_vqa"
OPTION_PATTERN = re.compile(r"^([A-D])\.\s*(.+?)\s*$", re.MULTILINE)


def image_split(source_image: str, val_percent: int) -> str:
    bucket = int(hashlib.sha256(source_image.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def _one_path(row: dict[str, Any], key: str, line_number: int) -> str:
    values = row.get(key)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"line {line_number}: {key} must contain exactly one string path")
    return values[0]


def _resolve_file(dataset_root: Path, relative_path: str, key: str, line_number: int) -> Path:
    path = (dataset_root / relative_path).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: {key} escapes the dataset root") from exc
    if not path.is_file():
        raise ValueError(f"line {line_number}: missing {key}: {path}")
    return path


def convert_row(row: dict[str, Any], dataset_root: Path, source_index: int) -> dict[str, Any]:
    line_number = source_index + 1
    original_relative = _one_path(row, "original_images", line_number)
    teacher_relative = _one_path(row, "teacher_images", line_number)
    boxed_relative = _one_path(row, "images", line_number)
    original_path = _resolve_file(dataset_root, original_relative, "original image", line_number)
    teacher_path = _resolve_file(dataset_root, teacher_relative, "teacher image", line_number)
    boxed_path = _resolve_file(dataset_root, boxed_relative, "boxed image", line_number)

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"line {line_number}: extra_info must be an object")
    question = str(extra_info.get("question") or "").strip()
    if not question:
        raise ValueError(f"line {line_number}: clean question is empty")
    if "<image>" in question or "red bounding box" in question.lower():
        raise ValueError(f"line {line_number}: clean question leaks image or bounding-box markup")

    answer = str(row.get("answer") or "").strip().upper()
    extra_answer = str(extra_info.get("answer") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"line {line_number}: answer must be one of A, B, C, D")
    if extra_answer != answer:
        raise ValueError(f"line {line_number}: answer disagrees with extra_info.answer")
    options = dict(OPTION_PATTERN.findall(question))
    if set(options) != {"A", "B", "C", "D"}:
        raise ValueError(f"line {line_number}: question must contain exactly options A-D")

    bbox = row.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        raise ValueError(f"line {line_number}: bbox must be valid [x1, y1, x2, y2]")

    return {
        # The RL loader consumes this column. It intentionally points to the
        # unmarked original, never the boxed student image.
        "images": [str(original_path)],
        "question": question,
        "solution": answer,
        "answer_text": options[answer],
        "options": options,
        # These fields are provenance and offline diagnostics only.
        "bbox": [float(value) for value in bbox],
        "source_index": source_index,
        "source_image": original_relative,
        "teacher_image": str(teacher_path),
        "boxed_image": str(boxed_path),
        "data_source": DATA_SOURCE,
        "ability": ABILITY,
    }


def load_and_validate(input_path: Path) -> list[dict[str, Any]]:
    dataset_root = input_path.parent.resolve()
    prepared: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"line {source_index + 1}: blank lines are not allowed")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {source_index + 1}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {source_index + 1}: row must be an object")
            prepared.append(convert_row(row, dataset_root, source_index))
    if not prepared:
        raise ValueError(f"input contains no rows: {input_path}")
    return prepared


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty split: {path.name}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    val_percent: int = 0,
    smoke_train: int = 14,
    smoke_val: int = 2,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not 0 <= val_percent <= 50:
        raise ValueError("val_percent must be between 0 and 50")
    if smoke_train < 1 or smoke_val < 1:
        raise ValueError("smoke split sizes must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    prepared = load_and_validate(input_path)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in prepared:
        if val_percent and image_split(row["source_image"], val_percent) == "val":
            val_rows.append(row)
        else:
            train_rows.append(row)
    if len(train_rows) < smoke_train:
        raise ValueError(f"train split has {len(train_rows)} rows, fewer than smoke_train={smoke_train}")
    if val_percent and len(val_rows) < smoke_val:
        raise ValueError(f"val split has {len(val_rows)} rows, fewer than smoke_val={smoke_val}")

    answer_counts = dict(sorted(collections.Counter(row["solution"] for row in prepared).items()))
    summary = {
        "source": str(input_path),
        "data_source": DATA_SOURCE,
        "ability": ABILITY,
        "total_rows": len(prepared),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_percent": val_percent,
        "smoke_train_rows": smoke_train,
        "smoke_val_rows": smoke_val if val_percent else 0,
        "unique_original_images": len({row["images"][0] for row in prepared}),
        "answer_counts": answer_counts,
        "policy_image_field": "images (unmarked original image)",
        "policy_question_field": "question (extra_info.question)",
        "bbox_exposed_to_policy": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_parquet(train_rows, output_dir / "train.parquet")
    _write_parquet(train_rows[:smoke_train], output_dir / "smoke_train.parquet")
    if val_percent:
        _write_parquet(val_rows, output_dir / "val.parquet")
        _write_parquet(val_rows[:smoke_val], output_dir / "smoke_val.parquet")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root.parent / "datasets" / "vision-opd"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=dataset_root / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=dataset_root / "rl")
    parser.add_argument(
        "--val-percent",
        type=int,
        default=0,
        help="Optional hash-based validation percentage; 0 keeps every row in train",
    )
    parser.add_argument("--smoke-train", type=int, default=14)
    parser.add_argument("--smoke-val", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        args.input,
        args.output_dir,
        val_percent=args.val_percent,
        smoke_train=args.smoke_train,
        smoke_val=args.smoke_val,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
