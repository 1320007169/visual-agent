#!/usr/bin/env python3
"""Extract old accepted trajectories with relation-bearing tool queries."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OLD_BATCH = SCRIPT_DIR.parent / "zwz_relation_qwen3_kl_safe_steps001_165_success10362"
DEFAULT_INPUT = OLD_BATCH / "outputs/deepseek_v41/accepted.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "data/suspicious834_original_only.jsonl"
DEFAULT_REPORT = SCRIPT_DIR / "data/prepare_report.json"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TOOL_RESPONSE_WITH_SUFFIX_RE = re.compile(
    r"(<tool_response>\s*.*?\s*</tool_response>).*$", re.DOTALL
)
RELATION_LABELS = (
    "in front of",
    "right of",
    "left of",
    "next to",
    "above",
    "below",
    "behind",
    "inside",
    "contain",
    "overlap",
)


def phrase_in(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text.lower()) is not None


def final_answer(item: dict[str, Any]) -> str:
    for message in reversed(item.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        matches = ANSWER_RE.findall(str(message.get("content") or ""))
        if matches:
            return matches[-1].strip().lower()
    raise ValueError("missing_final_answer")


def suspicious_hits(item: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    hits = []
    for turn_index, message in enumerate(item.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        for raw_call in TOOL_CALL_RE.findall(str(message.get("content") or "")):
            call = json.loads(raw_call)
            arguments = call.get("arguments") or {}
            for field in ("query", "label"):
                value = arguments.get(field)
                if not isinstance(value, str):
                    continue
                relations = [label for label in RELATION_LABELS if phrase_in(value, label)]
                if relations:
                    hits.append({
                        "turn_index": turn_index,
                        "tool": call.get("name"),
                        "field": field,
                        "value": value,
                        "relation_phrases": relations,
                        "contains_final_answer": phrase_in(value, answer),
                    })
    return hits


def original_only_copy(item: dict[str, Any], line_number: int, answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(json.dumps(item))
    images = result.get("images") or []
    if not images or not Path(str(images[0])).is_file():
        raise ValueError(f"missing original image at accepted line {line_number}")
    result["images"] = [images[0]]
    for message in result.get("messages") or []:
        if message.get("role") == "user" and "<tool_response>" in str(message.get("content") or ""):
            message["content"] = TOOL_RESPONSE_WITH_SUFFIX_RE.sub(r"\1", message["content"])
    result["metadata"] = {
        "ability": "spatial_relation",
        "reference_answer": answer,
        "old_accepted_line": line_number,
        "suspicious_tool_fields": hits,
        "rejudge_scope": "old DeepSeek-accepted rows with a relation phrase in grounding query or crop label",
        "image_policy": "original_image_only",
    }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output or report already exists")
    selected = []
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counts["input_rows"] += 1
            item = json.loads(line)
            answer = final_answer(item)
            hits = suspicious_hits(item, answer)
            if not hits:
                continue
            selected.append(original_only_copy(item, line_number, answer, hits))
            counts["selected_rows"] += 1
            counts["rows_with_final_answer_hit"] += int(
                any(hit["contains_final_answer"] for hit in hits)
            )
            counts["suspicious_tool_fields"] += len(hits)
            counts["final_answer_tool_fields"] += sum(
                hit["contains_final_answer"] for hit in hits
            )
    if counts["selected_rows"] != 834:
        raise ValueError(f"expected 834 suspicious rows, got {counts['selected_rows']}")
    with args.output.open("x", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "selection": "relation phrase in grounding_detect.query or crop_zoom.label",
        "relation_phrases": list(RELATION_LABELS),
        "image_policy": "original_image_only",
        "counts": dict(counts),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
