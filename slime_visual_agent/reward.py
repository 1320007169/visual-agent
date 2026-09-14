"""SLIME reward adapter matching the deterministic ZWZ VERL reward."""

from __future__ import annotations

import re
from typing import Any


RELATION_LABELS = frozenset(
    {
        "above",
        "below",
        "left of",
        "right of",
        "in front of",
        "behind",
        "inside",
        "contain",
        "overlap",
        "next to",
    }
)
RELATION_ALIASES = {
    "left": "left of",
    "to the left of": "left of",
    "on the left of": "left of",
    "right": "right of",
    "to the right of": "right of",
    "on the right of": "right of",
    "under": "below",
    "beneath": "below",
    "inside of": "inside",
    "within": "inside",
    "contains": "contain",
    "containing": "contain",
    "overlaps": "overlap",
    "overlapping": "overlap",
    "beside": "next to",
    "adjacent to": "next to",
}


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized.strip(" \t\r\n.,;:!?")


def _normalize_relation(text: str) -> str | None:
    normalized = _normalize(text)
    normalized = RELATION_ALIASES.get(normalized, normalized)
    return normalized if normalized in RELATION_LABELS else None


def has_strict_answer_format(text: str) -> bool:
    if len(re.findall(r"<answer>", text, flags=re.IGNORECASE)) != 1:
        return False
    if len(re.findall(r"</answer>", text, flags=re.IGNORECASE)) != 1:
        return False
    final_message = text
    if re.search(r"<tool_response>", text, flags=re.IGNORECASE):
        assistant_markers = list(
            re.finditer(r"(?:<\|im_start\|>|^|\s)assistant\s*", text, flags=re.IGNORECASE)
        )
        if not assistant_markers:
            return False
        final_message = text[assistant_markers[-1].end() :]
    cleaned = re.sub(
        r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|</s>))+\s*$",
        "",
        final_message,
        flags=re.IGNORECASE,
    )
    return re.search(
        r"<answer>\s*[^<>]+?\s*</answer>\s*$",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ) is not None


def score_response(response: str, ground_truth: str) -> dict[str, float]:
    """Score a closed-label spatial-relation response without an API judge."""
    answer = extract_answer(response)
    format_reward = 1.0 if has_strict_answer_format(response) else 0.0
    if not answer:
        return {"score": 0.0, "acc": 0.0, "format": 0.0, "tool_used": 0.0}
    prediction = _normalize_relation(answer)
    expected = _normalize_relation(str(ground_truth))
    accuracy = 1.0 if prediction is not None and prediction == expected else 0.0
    return {
        "score": 0.9 * accuracy + 0.1 * format_reward,
        "acc": accuracy,
        "format": format_reward,
        "tool_used": 1.0 if "<tool_call>" in response else 0.0,
    }


async def compute_reward(args: Any, sample: Any) -> dict[str, float]:
    """SLIME ``--custom-rm-path`` entry point."""
    del args
    result = score_response(sample.response, str(sample.label or ""))
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    metadata["visual_agent_reward"] = result
    sample.metadata = metadata
    return result
