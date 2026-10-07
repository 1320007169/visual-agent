"""Counterfactual faults for model-visible visual tool observations.

Shared by RL rollouts and evaluation. A fault keeps the observation schema but
makes its content wrong, so the policy can only recover by checking the image
or another tool. This module has no third-party dependencies so evaluation can
load it by file path without importing the verl package.
"""

from __future__ import annotations

import random
import re
import string
from typing import Any

FAULT_TOOLS = ("object_count", "grounding_detect", "depth_measure", "ocr_read")


def _integer(value: Any) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _count_fault(observation: dict, rng: random.Random, ground_truth: Any) -> dict | None:
    count = observation.get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        return None
    if count > 0:
        candidates = {round(count * factor) for factor in (0.5, 0.67, 1.5, 2.0)} | {count - 2, count + 2, count + 3}
    else:
        candidates = {1, 2, 3}
    excluded = {count, _integer(ground_truth)}
    candidates = sorted(value for value in candidates if value >= 0 and value not in excluded)
    return {**observation, "count": rng.choice(candidates)} if candidates else None


def _like(value: float, reference: Any) -> int | float:
    return int(value) if isinstance(reference, int) else round(float(value), 1)


def _detection_fault(observation: dict, rng: random.Random, ground_truth: Any) -> dict | None:
    boxes = observation.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        return None
    if rng.random() < 0.5:
        # A missed detection, the most common real failure of the detector.
        return {**observation, "boxes": [], "confidence": [], "labels": []}
    shifted = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            return None
        x1, y1, x2, y2 = box
        width, height = max(1, int(x2 - x1)), max(1, int(y2 - y1))
        left = rng.randint(0, max(0, 1000 - width))
        top_edge = rng.randint(0, max(0, 1000 - height))
        shifted.append([_like(left, x1), _like(top_edge, y1), _like(left + width, x2), _like(top_edge + height, y2)])
    return {**observation, "boxes": shifted}


def _depth_fault(observation: dict, rng: random.Random, ground_truth: Any) -> dict | None:
    regions = observation.get("regions")
    if not isinstance(regions, list) or not regions:
        return None
    depths = [region.get("depth_m") if isinstance(region, dict) else None for region in regions]
    if not all(isinstance(depth, (int, float)) and not isinstance(depth, bool) for depth in depths):
        return None
    faulty = depths[::-1]
    if faulty == depths:
        factor = rng.choice((0.5, 2.0))
        faulty = [round(depth * factor, 3) for depth in depths]
        if faulty == depths:
            return None
    return {**observation, "regions": [{**region, "depth_m": depth} for region, depth in zip(regions, faulty)]}


def _ocr_fault(observation: dict, rng: random.Random, ground_truth: Any) -> dict | None:
    text = observation.get("text")
    if not isinstance(text, str):
        return None
    positions = [index for index, char in enumerate(text) if char.isascii() and char.isalnum()]
    answer = str(ground_truth).strip() if isinstance(ground_truth, (str, int, float)) else ""
    start = text.lower().find(answer.lower()) if answer else -1
    if start >= 0:
        # Corrupt the span that carries the answer so the fault is load-bearing.
        positions = [index for index in positions if start <= index < start + len(answer)] or positions
    if not positions:
        return None
    chars = list(text)
    for index in rng.sample(positions, min(len(positions), rng.randint(1, 3))):
        char = chars[index]
        pool = string.digits if char.isdigit() else string.ascii_uppercase if char.isupper() else string.ascii_lowercase
        chars[index] = rng.choice(pool.replace(char, ""))
    faulty = "".join(chars)
    return None if faulty == text else {**observation, "text": faulty}


def _ocr_confusable_fault(observation: dict, rng: random.Random, ground_truth: Any) -> dict | None:
    text = observation.get("text")
    if not isinstance(text, str):
        return None
    commands = list(re.finditer(r"\\[a-zA-Z]+", text))
    matches = [match for match in re.finditer(r"rn|[0O1lm]", text)
               if not any(command.start() <= match.start() < command.end() for command in commands)]
    if not matches:
        return None
    match = rng.choice(matches)
    replacement = {"0": "O", "O": "0", "1": "l", "l": "1", "rn": "m", "m": "rn"}[match.group()]
    return {**observation, "text": text[:match.start()] + replacement + text[match.end():]}


_FAULTS = {
    "object_count": _count_fault,
    "grounding_detect": _detection_fault,
    "depth_measure": _depth_fault,
    "ocr_read": _ocr_fault,
}


def inject_fault(tool_name: str, observation: Any, rng: random.Random, ground_truth: Any = None,
                 *, variant: str = "training") -> dict | None:
    """Return a wrong but well-formed observation, or None when it cannot be faulted.

    ``ground_truth`` is optional; when given, count faults avoid the true count
    and OCR faults corrupt the span containing the answer.
    """
    if variant not in {"training", "ocr_confusable"}:
        raise ValueError(f"Unknown fault variant: {variant}")
    if not isinstance(observation, dict) or observation.get("status") in {"error", "failed"}:
        return None
    fault = _FAULTS.get(tool_name) if variant == "training" else (
        _ocr_confusable_fault if tool_name == "ocr_read" else None
    )
    return fault(observation, rng, ground_truth) if fault else None


def fault_probability(profile: dict | None, data_source: str | None, tool_name: str) -> float:
    """Probability of faulting a tool on a data source; a missing profile always faults."""
    if not profile:
        return 1.0
    by_source = profile.get("by_source", {}).get(data_source or "", {})
    return float(by_source.get(tool_name, profile.get("by_tool", {}).get(tool_name, profile.get("default", 1.0))))
