#!/usr/bin/env python3
"""Make Visual_Agent JSONL image fields portable within the dataset directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PAIRS = {
    "counting/final_counting_sam3_dino_trajectories.natural.jsonl":
        "counting/final_counting_sam3_dino_trajectories.natural_with_images.jsonl",
    "spatial/final_spatial_sam3_trajectories.natural.jsonl":
        "spatial/final_spatial_sam3_trajectories.natural_with_images.jsonl",
    "attribute/final_attribute_sam3_crop_zoom_trajectories.natural.jsonl":
        "attribute/final_attribute_sam3_crop_zoom_trajectories.natural_with_images.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def normalize_image_meta(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(meta, dict) for meta in value):
        return value
    raise ValueError("image_meta must be an object, an array of objects, or null")


def normalize_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["image_meta"] = normalize_image_meta(row.get("image_meta"))


def sync_portable_fields(root: Path) -> set[Path]:
    changed: set[Path] = set()
    for natural_name, portable_name in PAIRS.items():
        natural_path = root / natural_name
        portable_path = root / portable_name
        natural_rows = read_jsonl(natural_path)
        portable_rows = read_jsonl(portable_path)
        portable_by_uid = {str(row.get("uid")): row for row in portable_rows}
        if len(portable_by_uid) != len(portable_rows):
            raise ValueError(f"duplicate uid in {portable_path}")

        for row in natural_rows:
            uid = str(row.get("uid"))
            if uid not in portable_by_uid:
                raise ValueError(f"uid {uid!r} from {natural_path} is missing in {portable_path}")
            portable = portable_by_uid[uid]
            row["image"] = portable.get("image")
            row["images"] = portable.get("images") or []
            row["image_meta"] = normalize_image_meta(portable.get("image_meta"))

        normalize_rows(portable_rows)
        write_jsonl_atomic(natural_path, natural_rows)
        write_jsonl_atomic(portable_path, portable_rows)
        changed.update((natural_path, portable_path))
    return changed


def validate_portable_paths(root: Path, paths: set[Path]) -> tuple[int, int]:
    row_count = 0
    image_references = 0
    for path in sorted(paths):
        for line_no, row in enumerate(read_jsonl(path), start=1):
            row_count += 1
            image_meta = row.get("image_meta")
            if not isinstance(image_meta, list) or not all(isinstance(meta, dict) for meta in image_meta):
                raise ValueError(f"{path}:{line_no}: image_meta is not an array of objects")

            references: list[str] = []
            if row.get("image"):
                references.append(str(row["image"]))
            references.extend(str(image) for image in row.get("images") or [])
            references.extend(str(meta["path"]) for meta in image_meta if meta.get("path"))
            for reference in references:
                image_references += 1
                if Path(reference).is_absolute():
                    raise ValueError(f"{path}:{line_no}: absolute operational path: {reference}")
                if not (root / reference).is_file():
                    raise ValueError(f"{path}:{line_no}: missing dataset image: {reference}")
    return row_count, image_references


def normalize_dataset(root: Path) -> dict[str, int]:
    root = root.resolve()
    changed = sync_portable_fields(root)
    combined = root / "all_training_trajectories_with_images.jsonl"
    combined_rows = read_jsonl(combined)
    normalize_rows(combined_rows)
    write_jsonl_atomic(combined, combined_rows)
    changed.add(combined)
    rows, references = validate_portable_paths(root, changed)
    return {"files": len(changed), "rows": rows, "image_references_checked": references}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the training_trajectories_natural directory",
    )
    args = parser.parse_args()
    print(json.dumps(normalize_dataset(args.dataset_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
