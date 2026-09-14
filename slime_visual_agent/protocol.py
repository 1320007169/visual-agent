"""Pure trajectory helpers shared by the SLIME rollout and CPU tests."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
DEFAULT_TOOL_NAMES = frozenset(
    {
        "crop_zoom",
        "grounding_detect",
        "sam3_segment_multi",
        "sam3_crop_zoom",
        "sam3_crop_zoom_multi",
    }
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def extract_latest_tool_call(text: str, allowed_names: Iterable[str] = DEFAULT_TOOL_NAMES) -> ToolCall | None:
    """Return the last well-formed XML-wrapped tool call in one model turn."""
    allowed = set(allowed_names)
    for raw_call in reversed(TOOL_CALL_RE.findall(text or "")):
        try:
            payload = json.loads(raw_call)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(name, str) and name in allowed and isinstance(arguments, dict):
            return ToolCall(name=name, arguments=arguments)
    return None


def tool_observation_message(output: str, returned_images: list[str]) -> dict[str, Any]:
    """Build the Qwen3-VL user observation used after an XML tool call."""
    text = f"<tool_response>\n{output}\n</tool_response>"
    if not returned_images:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            *[{"type": "image", "image": image} for image in returned_images],
        ],
    }


def append_response_segment(
    sample: Any,
    response_tokens: list[int],
    token_ids: Iterable[int],
    *,
    log_probs: Iterable[float] | None,
    trainable: bool,
) -> None:
    """Append one aligned generated or environment-observation segment."""
    tokens = [int(token) for token in token_ids]
    probabilities = [float(value) for value in log_probs] if log_probs is not None else [0.0] * len(tokens)
    if len(tokens) != len(probabilities):
        raise ValueError(f"token/log-prob length mismatch: {len(tokens)} != {len(probabilities)}")

    if sample.loss_mask is None:
        sample.loss_mask = []
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.loss_mask.extend([1 if trainable else 0] * len(tokens))
    sample.rollout_log_probs.extend(probabilities)
    sample.response_length = len(response_tokens)


def validate_sample_alignment(sample: Any, prompt_length: int) -> None:
    """Fail fast before SLIME converts a multimodal trajectory to train data."""
    if prompt_length < 0 or prompt_length > len(sample.tokens):
        raise ValueError(f"invalid prompt length {prompt_length} for {len(sample.tokens)} tokens")
    response_length = len(sample.tokens) - prompt_length
    if sample.response_length != response_length:
        raise ValueError(f"response length mismatch: {sample.response_length} != {response_length}")
    if sample.loss_mask is None or len(sample.loss_mask) != response_length:
        actual = None if sample.loss_mask is None else len(sample.loss_mask)
        raise ValueError(f"loss-mask length mismatch: {actual} != {response_length}")
    if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != response_length:
        actual = None if sample.rollout_log_probs is None else len(sample.rollout_log_probs)
        raise ValueError(f"rollout-log-prob length mismatch: {actual} != {response_length}")
    if not any(sample.loss_mask):
        raise ValueError("trajectory has no trainable response tokens")


def slime_grpo_advantages(rewards: Iterable[float], epsilon: float = 1e-6) -> list[float]:
    """Mirror SLIME/VERL group normalization, including sample standard deviation."""
    values = [float(value) for value in rewards]
    if len(values) < 2:
        raise ValueError("GRPO requires at least two samples per prompt")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return [(value - mean) / (std + epsilon) for value in values]
