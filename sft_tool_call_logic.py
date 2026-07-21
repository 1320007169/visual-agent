#!/usr/bin/env python3
"""SFT-compatible visual tool call logic.

This file intentionally keeps the runtime simple. It does not load SAM3,
SAM3.1, GroundingDINO, or start a server. Instead, callers pass in small
backend objects with these methods:

    sam3_backend.segment(image, query) -> {
        "boxes": [[x1, y1, x2, y2], ...],
        "confidence": [score, ...],
        "mask_area_px": [area, ...],
    }

    grounding_backend.detect(image, query) -> {
        "boxes": [[x1, y1, x2, y2], ...],
        "labels": ["person", ...],
        "confidence": [score, ...],
    }

The returned dictionaries match the tool response shapes used by the current
Visual Agent SFT data for:

    - crop_zoom
    - grounding_detect
    - sam3_segment_multi
    - sam3_crop_zoom
    - sam3_crop_zoom_multi
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from PIL import Image


Json = dict[str, Any]
CropWriter = Callable[[Image.Image, str, int], str]


def crop_zoom(
    images: list[Image.Image],
    bbox_2d: list[float],
    *,
    crop_dir: str | Path,
    uid: str = "sample",
    target_image: int = 0,
    label: str | None = None,
    slack_ratio: float = 0.0,
    min_crop_side: int = 96,
    crop_writer: CropWriter | None = None,
) -> Json:
    """Crop a Qwen3-VL relative [0, 1000] bbox without resampling it."""

    image = _get_image(images, target_image)
    relative_box, selected_box = _relative_bbox_to_absolute(bbox_2d, image.size)
    if label is not None:
        label = _require_text(label, "label")
    try:
        slack_ratio = float(slack_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("slack_ratio must be a non-negative number") from exc
    if not math.isfinite(slack_ratio) or slack_ratio < 0:
        raise ValueError("slack_ratio must be a non-negative number")

    crop_obs = _make_crop_zoom(
        image=image,
        selected_box=selected_box,
        query=label or "specified region",
        crop_dir=Path(crop_dir),
        filename=f"{uid}_crop_zoom.jpg",
        crop_target_image=len(images),
        slack_ratio=slack_ratio,
        min_crop_side=min_crop_side,
        output_side=None,
        crop_writer=crop_writer,
        text_summary="Cropped the specified image region at its native resolution.",
    )
    return {
        "bbox_2d": relative_box,
        "pixel_bbox_2d": selected_box,
        "label": label,
        "target_image": target_image,
        "crop_zoom": crop_obs,
        "image_outputs": crop_obs["image_outputs"],
        "coordinate_space": "relative_0_1000",
        "source": "crop_zoom",
    }


def grounding_detect(
    images: list[Image.Image],
    query: str,
    *,
    grounding_backend: Any,
    target_image: int = 0,
) -> Json:
    """Run GroundingDINO-style detection and return the SFT tool schema."""

    image = _get_image(images, target_image)
    query = _require_text(query, "query")
    detected = grounding_backend.detect(image, query)
    pixel_boxes = _round_boxes(detected.get("boxes") or [])
    boxes = [_absolute_bbox_to_relative(box, image.size) for box in pixel_boxes]
    labels = [str(v) for v in detected.get("labels") or [query] * len(boxes)]
    confidence = _round_floats(detected.get("confidence") or [])
    return {
        "query": query,
        "boxes": boxes,
        "labels": labels,
        "confidence": confidence,
        "count": len(boxes),
        "coordinate_space": "relative_0_1000",
        "source": "groundingdino",
    }


def sam3_segment_multi(
    images: list[Image.Image],
    queries: list[Json],
    *,
    sam3_backend: Any,
    target_image: int = 0,
) -> Json:
    """Run SAM3/SAM3.1 text-prompt segmentation for one or more queries."""

    image = _get_image(images, target_image)
    normalized_queries = _require_queries(queries)
    groups = []
    for item in normalized_queries:
        segmented = sam3_backend.segment(image, item["query"])
        group, _ = _segment_group(
            item, segmented, include_selection=False, image_size=image.size
        )
        groups.append(group)
    return {"queries": groups, "coordinate_space": "relative_0_1000", "source": "sam3"}


def sam3_crop_zoom(
    images: list[Image.Image],
    query: str,
    *,
    sam3_backend: Any,
    crop_dir: str | Path,
    uid: str = "sample",
    target_image: int = 0,
    slack_ratio: float = 0.35,
    min_crop_side: int = 96,
    output_side: int = 336,
    crop_writer: CropWriter | None = None,
) -> Json:
    """Localize one target with SAM3 and write one crop image."""

    image = _get_image(images, target_image)
    query = _require_text(query, "query")
    segmented = sam3_backend.segment(image, query)
    target, selected_pixel_box = _segment_group(
        {"role": "target", "query": query},
        segmented,
        include_selection=True,
        image_size=image.size,
    )
    crop_obs = _make_crop_zoom(
        image=image,
        selected_box=selected_pixel_box,
        query=query,
        crop_dir=Path(crop_dir),
        filename=f"{uid}_crop_zoom.jpg",
        crop_target_image=len(images),
        slack_ratio=slack_ratio,
        min_crop_side=min_crop_side,
        output_side=output_side,
        crop_writer=crop_writer,
        text_summary="Localized the target and cropped the region for closer visual inspection.",
    )
    return {
        "query": query,
        "target_image": target_image,
        "target": target,
        "crop_zoom": crop_obs,
        "image_outputs": crop_obs["image_outputs"],
        "coordinate_space": "relative_0_1000",
        "source": "sam3_crop_zoom",
    }


def sam3_crop_zoom_multi(
    images: list[Image.Image],
    queries: list[Json],
    *,
    sam3_backend: Any,
    crop_dir: str | Path,
    uid: str = "sample",
    target_image: int = 0,
    slack_ratio: float = 0.35,
    min_crop_side: int = 96,
    output_side: int = 336,
    crop_writer: CropWriter | None = None,
) -> Json:
    """Localize multiple targets with SAM3 and write one crop per query."""

    image = _get_image(images, target_image)
    normalized_queries = _require_queries(queries)
    crop_dir = Path(crop_dir)
    groups = []
    image_outputs = []
    for index, item in enumerate(normalized_queries, start=1):
        segmented = sam3_backend.segment(image, item["query"])
        group, selected_pixel_box = _segment_group(
            item, segmented, include_selection=True, image_size=image.size
        )
        crop_obs = _make_crop_zoom(
            image=image,
            selected_box=selected_pixel_box,
            query=item["query"],
            crop_dir=crop_dir,
            filename=f"{uid}_crop_zoom_t{index}.jpg",
            crop_target_image=len(images) + index - 1,
            slack_ratio=slack_ratio,
            min_crop_side=min_crop_side,
            output_side=output_side,
            crop_writer=crop_writer,
            text_summary=f"Cropped the localized {item['query']} region for closer visual inspection.",
        )
        group["crop_zoom"] = crop_obs
        groups.append(group)
        image_outputs.extend(crop_obs["image_outputs"])
    return {
        "queries": groups,
        "image_outputs": image_outputs,
        "coordinate_space": "relative_0_1000",
        "source": "sam3_crop_zoom_multi",
    }


def execute_tool_call(
    tool_call: Json,
    *,
    images: list[Image.Image],
    sam3_backend: Any | None = None,
    grounding_backend: Any | None = None,
    crop_dir: str | Path = "tool_crops",
    uid: str = "sample",
    crop_writer: CropWriter | None = None,
    min_crop_side: int = 96,
    output_side: int = 336,
) -> Json:
    """Dispatch one parsed SFT tool call by name."""

    name = tool_call.get("name")
    arguments = tool_call.get("arguments") or {}
    if name == "crop_zoom":
        return crop_zoom(
            images,
            crop_dir=crop_dir,
            uid=uid,
            crop_writer=crop_writer,
            min_crop_side=min_crop_side,
            **arguments,
        )
    if name == "grounding_detect":
        if grounding_backend is None:
            raise ValueError("grounding_backend is required for grounding_detect")
        return grounding_detect(images, grounding_backend=grounding_backend, **arguments)
    if name == "sam3_segment_multi":
        if sam3_backend is None:
            raise ValueError("sam3_backend is required for sam3_segment_multi")
        return sam3_segment_multi(images, sam3_backend=sam3_backend, **arguments)
    if name == "sam3_crop_zoom":
        if sam3_backend is None:
            raise ValueError("sam3_backend is required for sam3_crop_zoom")
        return sam3_crop_zoom(
            images,
            sam3_backend=sam3_backend,
            crop_dir=crop_dir,
            uid=uid,
            crop_writer=crop_writer,
            min_crop_side=min_crop_side,
            output_side=output_side,
            **arguments,
        )
    if name == "sam3_crop_zoom_multi":
        if sam3_backend is None:
            raise ValueError("sam3_backend is required for sam3_crop_zoom_multi")
        return sam3_crop_zoom_multi(
            images,
            sam3_backend=sam3_backend,
            crop_dir=crop_dir,
            uid=uid,
            crop_writer=crop_writer,
            min_crop_side=min_crop_side,
            output_side=output_side,
            **arguments,
        )
    raise ValueError(f"unsupported tool call: {name!r}")


def _segment_group(
    query_item: Json,
    segmented: Json,
    *,
    include_selection: bool,
    image_size: tuple[int, int],
) -> tuple[Json, list[float] | None]:
    pixel_boxes = _round_boxes(segmented.get("boxes") or [])
    boxes = [_absolute_bbox_to_relative(box, image_size) for box in pixel_boxes]
    confidence = _round_floats(segmented.get("confidence") or [])
    mask_area_px = [int(v) for v in segmented.get("mask_area_px") or []]
    group = {
        "role": query_item.get("role", "target"),
        "query": query_item["query"],
        "boxes": boxes,
        "confidence": confidence,
        "mask_area_px": mask_area_px,
        "count": len(boxes),
    }
    if include_selection:
        group.update(
            {
                "selected_box": boxes[0] if boxes else None,
                "selected_confidence": confidence[0] if confidence else None,
                "selected_mask_area_px": mask_area_px[0] if mask_area_px else None,
            }
        )
    return group, pixel_boxes[0] if pixel_boxes else None


def _make_crop_zoom(
    *,
    image: Image.Image,
    selected_box: list[float] | None,
    query: str,
    crop_dir: Path,
    filename: str,
    crop_target_image: int,
    slack_ratio: float,
    min_crop_side: int,
    output_side: int | None,
    crop_writer: CropWriter | None,
    text_summary: str,
) -> Json:
    if selected_box is None:
        raise ValueError(f"SAM3 returned no selected box for query {query!r}")
    crop_bbox = _expanded_square_bbox(selected_box, image.size, slack_ratio, min_crop_side)
    rect = _integer_crop_rect(crop_bbox, image.size)
    crop = image.crop(rect)
    if output_side is not None:
        crop = crop.resize((output_side, output_side), Image.Resampling.LANCZOS)
    if crop_writer is None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = str(crop_dir / filename)
        crop.save(crop_path, quality=95)
    else:
        crop_path = crop_writer(crop, filename, crop_target_image)
    image_output = {
        "target_image": crop_target_image,
        "path": str(crop_path),
        "width": crop.width,
        "height": crop.height,
        "source": "crop_zoom",
    }
    return {
        "target_image": crop_target_image,
        "crop_path": str(crop_path),
        "requested_bbox_2d": _absolute_bbox_to_relative(selected_box, image.size),
        "bbox_2d": _absolute_bbox_to_relative(crop_bbox, image.size),
        "coordinate_space": "relative_0_1000",
        "slack_ratio": float(slack_ratio),
        "image_outputs": [image_output],
        "text_summary": text_summary,
    }


def _expanded_square_bbox(
    box: list[float],
    image_size: tuple[int, int],
    slack_ratio: float,
    min_crop_side: int,
) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw * (1.0 + 2.0 * slack_ratio), bh * (1.0 + 2.0 * slack_ratio), float(min_crop_side))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = cx - side / 2.0
    top = cy - side / 2.0
    right = cx + side / 2.0
    bottom = cy + side / 2.0

    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > width:
        left -= right - width
        right = float(width)
    if bottom > height:
        top -= bottom - height
        bottom = float(height)

    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    return _round_box([left, top, right, bottom])


def _integer_crop_rect(box: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    rect = (
        max(0, min(width - 1, round(left))),
        max(0, min(height - 1, round(top))),
        max(1, min(width, round(right))),
        max(1, min(height, round(bottom))),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError(f"invalid crop bbox: {box}")
    return rect


def _get_image(images: list[Image.Image], target_image: int) -> Image.Image:
    if not isinstance(target_image, int):
        raise ValueError("target_image must be an integer")
    if target_image < 0 or target_image >= len(images):
        raise ValueError(f"target_image {target_image} outside images[0:{len(images)}]")
    return images[target_image].convert("RGB")


def _relative_bbox_to_absolute(
    value: Any, image_size: tuple[int, int]
) -> tuple[list[float], list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_2d must be relative [x1, y1, x2, y2] coordinates from 0 to 1000")
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox_2d coordinates must be finite numbers") from exc
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("bbox_2d coordinates must be finite numbers")

    relative_box = [
        max(0.0, x1),
        max(0.0, y1),
        min(1000.0, x2),
        min(1000.0, y2),
    ]
    if relative_box[0] >= relative_box[2] or relative_box[1] >= relative_box[3]:
        raise ValueError(f"bbox_2d does not define a valid relative region: {value}")

    width, height = image_size
    absolute_box = [
        relative_box[0] / 1000 * width,
        relative_box[1] / 1000 * height,
        relative_box[2] / 1000 * width,
        relative_box[3] / 1000 * height,
    ]
    return _round_box(relative_box), _round_box(absolute_box)


def _absolute_bbox_to_relative(
    value: list[float], image_size: tuple[int, int]
) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in value]
    relative_box = [
        max(0.0, min(1000.0, x1 / width * 1000)),
        max(0.0, min(1000.0, y1 / height * 1000)),
        max(0.0, min(1000.0, x2 / width * 1000)),
        max(0.0, min(1000.0, y2 / height * 1000)),
    ]
    return _round_box(relative_box)


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_queries(value: Any) -> list[Json]:
    if not isinstance(value, list) or not value:
        raise ValueError("queries must be a non-empty list")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each query entry must be an object")
        normalized.append(
            {
                "role": str(item.get("role", "target")),
                "query": _require_text(item.get("query"), "query"),
            }
        )
    return normalized


def _round_floats(values: Any, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _round_box(box: list[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in box]


def _round_boxes(boxes: Any, digits: int = 4) -> list[list[float]]:
    return [_round_box(list(box), digits) for box in boxes]
