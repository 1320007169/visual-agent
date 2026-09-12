#!/usr/bin/env python3
"""Convert GroundingDINO VisReason trajectories to visual-agent SFT JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from convert_visual_tool_sft import convert_item
from validate_visual_tool_sft import validate_item


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    Path(
        "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/"
        "groundingdino_offline_pipeline/outputs/"
        "gdino_visreason_general_qwen80b_20260813_211307/"
        "03_accepted_trajectories.jsonl"
    ),
    Path(
        "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/"
        "groundingdino_offline_pipeline/outputs/"
        "gdino_visreason_general_qwen80b_offset30000_limit30000/"
        "03_accepted_trajectories.jsonl"
    ),
]
DEFAULT_OUTPUT = ROOT / "data/groundingdino_visreason_sft_55980.jsonl"
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def read_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: row is not a JSON object")
                yield path, line_no, row


def relative_bbox(bbox: Any, image_size: tuple[int, int]) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"invalid absolute bbox: {bbox!r}")
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    result = [
        max(0.0, min(1000.0, x1 / width * 1000)),
        max(0.0, min(1000.0, y1 / height * 1000)),
        max(0.0, min(1000.0, x2 / width * 1000)),
        max(0.0, min(1000.0, y2 / height * 1000)),
    ]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"bbox does not define a valid region: {bbox!r}")
    return [round(value, 3) for value in result]


def get_image_size(
    images: list[str], target_image: int, cache: dict[int, tuple[int, int]]
) -> tuple[int, int]:
    if target_image < 0 or target_image >= len(images):
        raise ValueError(f"target_image {target_image} outside images[0:{len(images)}]")
    if target_image not in cache:
        with Image.open(images[target_image]) as image:
            cache[target_image] = image.size
    return cache[target_image]


def normalize_grounding_response(
    observation: dict[str, Any], image_size: tuple[int, int]
) -> dict[str, Any]:
    normalized = dict(observation)
    normalized["boxes"] = [
        relative_bbox(box, image_size) for box in observation.get("boxes") or []
    ]
    normalized.pop("target_image", None)
    normalized["coordinate_space"] = "relative_0_1000"
    return normalized


def normalize_crop_response(
    observation: dict[str, Any], arguments: dict[str, Any], image_size: tuple[int, int]
) -> dict[str, Any]:
    selected_pixel_box = [float(value) for value in arguments["bbox_2d"]]
    selected_box = relative_bbox(selected_pixel_box, image_size)
    crop_box = relative_bbox(observation["bbox_2d"], image_size)
    crop_observation = {
        "target_image": observation["target_image"],
        "crop_path": observation["crop_path"],
        "requested_bbox_2d": selected_box,
        "bbox_2d": crop_box,
        "coordinate_space": "relative_0_1000",
        "slack_ratio": float(arguments["slack_ratio"]),
        "image_outputs": observation["image_outputs"],
        "text_summary": "Cropped the specified image region at its native resolution.",
    }
    return {
        "bbox_2d": selected_box,
        "pixel_bbox_2d": selected_pixel_box,
        "label": arguments.get("label"),
        "target_image": arguments.get("target_image", 0),
        "crop_zoom": crop_observation,
        "image_outputs": observation["image_outputs"],
        "coordinate_space": "relative_0_1000",
        "source": "crop_zoom",
    }


def normalize_messages(row: dict[str, Any]) -> None:
    images = [str(image) for image in row.get("images") or []]
    if not images and row.get("image"):
        images = [str(row["image"])]
    size_cache: dict[int, tuple[int, int]] = {}
    pending_call: dict[str, Any] | None = None

    for message in row.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "")
        matches = list(TOOL_CALL_RE.finditer(content))

        if matches:
            if role != "assistant" or len(matches) != 1 or pending_call is not None:
                raise ValueError("each assistant turn must contain at most one tool call")
            match = matches[0]
            original_call = json.loads(match.group(1))
            normalized_call = json.loads(json.dumps(original_call))
            if normalized_call.get("name") == "crop_zoom":
                arguments = normalized_call.get("arguments") or {}
                target_image = int(arguments.get("target_image", 0))
                arguments["bbox_2d"] = relative_bbox(
                    arguments.get("bbox_2d"),
                    get_image_size(images, target_image, size_cache),
                )
            payload = json.dumps(normalized_call, ensure_ascii=False, separators=(",", ":"))
            message["content"] = (
                content[: match.start(1)] + payload + content[match.end(1) :]
            )
            pending_call = original_call
            continue

        if role == "tool":
            if pending_call is None:
                raise ValueError("tool response has no preceding tool call")
            observation = json.loads(content)
            arguments = pending_call.get("arguments") or {}
            target_image = int(arguments.get("target_image", 0))
            image_size = get_image_size(images, target_image, size_cache)
            if pending_call.get("name") == "grounding_detect":
                observation = normalize_grounding_response(observation, image_size)
            elif pending_call.get("name") == "crop_zoom":
                observation = normalize_crop_response(observation, arguments, image_size)
            else:
                raise ValueError(f"unsupported tool: {pending_call.get('name')}")
            message["content"] = json.dumps(
                observation, ensure_ascii=False, separators=(",", ":")
            )
            pending_call = None
        elif pending_call is not None:
            raise ValueError("tool call is not followed by a tool response")

    if pending_call is not None:
        raise ValueError("final tool call has no response")


def convert(inputs: list[Path], output: Path) -> int:
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input JSONL does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_uids: set[str] = set()
    count = 0
    with output.open("x", encoding="utf-8") as handle:
        for path, line_no, row in read_jsonl(inputs):
            try:
                normalize_messages(row)
                converted = convert_item(
                    row,
                    task_type=str(row.get("data_type") or "unknown"),
                    uid=str(row.get("uid") or ""),
                )
                validation = validate_item(converted, seen_uids)
                if validation.errors:
                    raise ValueError("; ".join(validation.errors))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            handle.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        help="Input JSONL; repeat for multiple batches. Defaults to the two current batches.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or DEFAULT_INPUTS
    count = convert(inputs, args.output)
    print(json.dumps({"inputs": [str(path) for path in inputs], "output": str(args.output), "rows": count}, indent=2))


if __name__ == "__main__":
    main()
