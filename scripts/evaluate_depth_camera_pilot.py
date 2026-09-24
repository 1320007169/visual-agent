#!/usr/bin/env python3
"""Run a staged GroundingDINO + DA3 check on curated camera-depth questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iou(a: list[float], b: list[float]) -> float:
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def pixel_box(box: list[float], width: int, height: int) -> list[float]:
    return [box[0] * width / 1000, box[1] * height / 1000,
            box[2] * width / 1000, box[3] * height / 1000]


def detect(args: argparse.Namespace) -> None:
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from visual_tool_server import GroundingDinoBackend

    rows = read_jsonl(args.input)[:args.limit]
    done = {row["uid"] for row in read_jsonl(args.output)}
    backend = GroundingDinoBackend(args.model, "cuda:0", 0.25, 0.25)
    for index, row in enumerate(rows, 1):
        if row["uid"] in done:
            continue
        result = {key: row[key] for key in (
            "uid", "images", "solution", "object_a", "object_b", "bbox_a", "bbox_b"
        )}
        try:
            with Image.open(row["images"][0]) as source:
                image = source.convert("RGB")
            width, height = image.size
            result["image_size"] = [width, height]
            for side in ("a", "b"):
                detected = backend.detect(image, row[f"object_{side}"])
                boxes = detected["boxes"]
                oracle = pixel_box(row[f"bbox_{side}"], width, height)
                result[f"detect_{side}"] = detected
                result[f"top_iou_{side}"] = iou(boxes[0], oracle) if boxes else 0.0
                result[f"best_iou_{side}"] = max((iou(box, oracle) for box in boxes), default=0.0)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(args.output, result)
        print(f"detect {index}/{len(rows)} {row['uid']}", flush=True)


def median_depth(depth, box: list[float], width: int, height: int) -> float | None:
    import numpy as np

    scale_x, scale_y = depth.shape[1] / width, depth.shape[0] / height
    x1 = max(0, round(box[0] * scale_x))
    y1 = max(0, round(box[1] * scale_y))
    x2 = min(depth.shape[1], round(box[2] * scale_x))
    y2 = min(depth.shape[0], round(box[3] * scale_y))
    valid = depth[y1:y2, x1:x2]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    return float(np.median(valid)) if valid.size else None


def depth(args: argparse.Namespace) -> None:
    import numpy as np
    from depth_anything_3.api import DepthAnything3

    rows = read_jsonl(args.input)[:args.limit]
    done = {row["uid"] for row in read_jsonl(args.output)}
    model = DepthAnything3.from_pretrained(args.model).to("cuda:0").eval()
    cache = {}
    for index, row in enumerate(rows, 1):
        if row["uid"] in done:
            continue
        result = {key: row[key] for key in (
            "uid", "images", "solution", "object_a", "object_b",
            "top_iou_a", "top_iou_b", "best_iou_a", "best_iou_b"
        ) if key in row}
        try:
            path = row["images"][0]
            if path not in cache:
                prediction = model.inference([path], process_res=504, export_format="mini_npz")
                cache[path] = np.asarray(prediction.depth[0], dtype=np.float32)
            estimated = cache[path]
            width, height = row["image_size"]
            for side in ("a", "b"):
                oracle = pixel_box(row[f"bbox_{side}"], width, height)
                result[f"oracle_depth_{side}"] = median_depth(estimated, oracle, width, height)
                detected = row[f"detect_{side}"]["boxes"]
                result[f"detected_depth_{side}"] = (
                    median_depth(estimated, detected[0], width, height) if detected else None
                )
            for kind in ("oracle", "detected"):
                left, right = result[f"{kind}_depth_a"], result[f"{kind}_depth_b"]
                result[f"{kind}_answer"] = (
                    "A" if left < right else "B" if right < left else None
                ) if left is not None and right is not None else None
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(args.output, result)
        print(f"depth {index}/{len(rows)} {row['uid']}", flush=True)


def report(args: argparse.Namespace) -> None:
    detected = read_jsonl(args.detected)
    depth_rows = read_jsonl(args.depth)
    def rate(rows: list[dict], predicate) -> str:
        return f"{sum(bool(predicate(row)) for row in rows)}/{len(rows)}" if rows else "0/0"
    summary = {
        "detected_rows": len(detected),
        "both_found": rate(detected, lambda r: bool(r.get("detect_a", {}).get("boxes")) and bool(r.get("detect_b", {}).get("boxes"))),
        "both_top_iou_0.5": rate(detected, lambda r: r.get("top_iou_a", 0) >= 0.5 and r.get("top_iou_b", 0) >= 0.5),
        "both_best_iou_0.5": rate(detected, lambda r: r.get("best_iou_a", 0) >= 0.5 and r.get("best_iou_b", 0) >= 0.5),
        "depth_rows": len(depth_rows),
        "oracle_correct": rate(depth_rows, lambda r: r.get("oracle_answer") == r["solution"]),
        "detected_correct": rate(depth_rows, lambda r: r.get("detected_answer") == r["solution"]),
        "detected_valid": rate(depth_rows, lambda r: r.get("detected_answer") is not None),
        "depth_errors": [r["uid"] for r in depth_rows if "error" in r],
        "localization_failures": [r["uid"] for r in detected if r.get("top_iou_a", 0) < 0.5 or r.get("top_iou_b", 0) < 0.5],
    }
    print(json.dumps(summary, indent=2))


def export(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    rows = pq.read_table(args.input).to_pylist()[:args.limit]
    if args.output.exists():
        raise FileExistsError(args.output)
    for row in rows:
        append_jsonl(args.output, row)
    print(f"exported {len(rows)} rows to {args.output}")


def locate_requests(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)[:args.limit]
    if args.output.exists():
        raise FileExistsError(args.output)
    for row in rows:
        for side in ("a", "b"):
            append_jsonl(args.output, {"image": row["images"][0], "query": row[f"object_{side}"]})
    print(f"exported {2 * len(rows)} LocateAnything requests to {args.output}")


def locate_convert(args: argparse.Namespace) -> None:
    from PIL import Image

    rows = read_jsonl(args.input)[:args.limit]
    responses = read_jsonl(args.responses)
    if len(responses) != 2 * len(rows):
        raise ValueError(f"Expected {2 * len(rows)} responses, got {len(responses)}")
    if args.output.exists():
        raise FileExistsError(args.output)
    pattern = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
    for index, row in enumerate(rows):
        result = {key: row[key] for key in (
            "uid", "images", "solution", "object_a", "object_b", "bbox_a", "bbox_b"
        )}
        with Image.open(row["images"][0]) as image:
            width, height = image.size
        result["image_size"] = [width, height]
        for side_index, side in enumerate(("a", "b")):
            response = responses[2 * index + side_index]
            if response["image"] != row["images"][0] or response["query"] != row[f"object_{side}"]:
                raise ValueError(f"Response order mismatch for {row['uid']} {side}")
            boxes = []
            for match in pattern.finditer(response["raw_response"]):
                box = pixel_box([int(value) for value in match.groups()], width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
            oracle = pixel_box(row[f"bbox_{side}"], width, height)
            result[f"detect_{side}"] = {"boxes": boxes, "raw_response": response["raw_response"]}
            result[f"top_iou_{side}"] = iou(boxes[0], oracle) if boxes else 0.0
            result[f"best_iou_{side}"] = max((iou(box, oracle) for box in boxes), default=0.0)
        append_jsonl(args.output, result)
    print(f"converted {len(rows)} LocateAnything rows to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("export", "detect", "depth"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name != "export":
            command.add_argument("--model", required=True)
        command.add_argument("--limit", type=int, default=44)
    command = sub.add_parser("report")
    command.add_argument("--detected", type=Path, required=True)
    command.add_argument("--depth", type=Path, required=True)
    command = sub.add_parser("locate_requests")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--limit", type=int, default=44)
    command = sub.add_parser("locate_convert")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--limit", type=int, default=44)
    args = parser.parse_args()
    {"export": export, "detect": detect, "depth": depth, "report": report,
     "locate_requests": locate_requests, "locate_convert": locate_convert}[args.stage](args)


if __name__ == "__main__":
    main()
