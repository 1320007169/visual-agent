"""SLIME reward adapter matching Visual-Agent's source-routed VERL reward."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
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


def _verl_score_response(
    response: str,
    ground_truth: str,
    metadata: dict[str, Any],
) -> dict[str, float]:
    reinforcement_learning = Path(__file__).resolve().parents[1] / "reinforcement_learning"
    if str(reinforcement_learning) not in sys.path:
        sys.path.insert(0, str(reinforcement_learning))
    from verl.utils.reward_score.visual_agent_thyme import compute_score

    return compute_score(response, ground_truth, metadata)


def score_response(
    response: str,
    ground_truth: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Route relation, free-form, and benchmark samples like the VERL run."""
    metadata = dict(metadata or {})
    metadata.setdefault("data_source", "visual-agent-zwz-relation")
    if metadata["data_source"] == "visual-agent-zwz-relation":
        metadata.setdefault("source", "zwz_rl_vqa/original_images")
    return _verl_score_response(response, str(ground_truth), metadata)


async def compute_reward(args: Any, sample: Any) -> dict[str, float]:
    """SLIME ``--custom-rm-path`` entry point."""
    del args
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if metadata.get("data_source") == "visual-agent-deepeyesv2":
        # This branch may call an external semantic judge. Keep that blocking
        # SDK request outside SLIME's rollout event loop.
        result = await asyncio.to_thread(
            score_response,
            sample.response,
            str(sample.label or ""),
            metadata,
        )
    else:
        result = score_response(sample.response, str(sample.label or ""), metadata)
    metadata["visual_agent_reward"] = result
    sample.metadata = metadata
    return result
