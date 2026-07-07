#!/usr/bin/env python
"""Validate converted visual-tool SFT JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .visual_tools import VISUAL_TOOL_NAMES
except ImportError:  # pragma: no cover - used when invoked as a script
    from visual_tools import VISUAL_TOOL_NAMES


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


@dataclass
class ItemValidation:
    uid: str
    task_type: str
    tools: list[str] = field(default_factory=list)
    image_count: int = 0
    errors: list[str] = field(default_factory=list)


def _json_loads_fragment(fragment: str, label: str, errors: list[str]) -> Any:
    try:
        return json.loads(fragment.strip())
    except json.JSONDecodeError as exc:
        errors.append(f"{label} JSON decode failed: {exc}")
        return None


def _validate_image_paths(item: dict[str, Any], result: ItemValidation) -> None:
    images = item.get("images") or []
    if item.get("image") and item.get("image") not in images:
        result.errors.append("image is not included in images")
    if not images:
        result.errors.append("images is empty")
    for image in images:
        if not Path(str(image)).exists():
            result.errors.append(f"image path does not exist: {image}")
    result.image_count = len(images)


def validate_item(item: dict[str, Any], seen_uids: set[str]) -> ItemValidation:
    uid = str(item.get("uid") or "")
    result = ItemValidation(uid=uid, task_type=str(item.get("task_type") or "unknown"))

    if not uid:
        result.errors.append("uid is empty")
    elif uid in seen_uids:
        result.errors.append(f"duplicate uid: {uid}")
    seen_uids.add(uid)

    _validate_image_paths(item, result)

    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        result.errors.append("messages is empty or not a list")
        return result

    if messages[0].get("role") != "user":
        result.errors.append("first message is not user")

    for idx, message in enumerate(messages):
        expected = "user" if idx % 2 == 0 else "assistant"
        if message.get("role") != expected:
            result.errors.append(f"message {idx} role is {message.get('role')}, expected {expected}")

    if len(messages) % 2 != 0:
        result.errors.append("messages must end with an assistant response")

    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    if not any("<tool_call>" in str(m.get("content") or "") for m in assistant_messages):
        result.errors.append("no assistant message contains <tool_call>")

    for message in assistant_messages:
        for fragment in TOOL_CALL_RE.findall(str(message.get("content") or "")):
            call = _json_loads_fragment(fragment, "tool_call", result.errors)
            if not isinstance(call, dict):
                result.errors.append("tool_call is not a JSON object")
                continue
            name = call.get("name")
            if name not in VISUAL_TOOL_NAMES:
                result.errors.append(f"unsupported tool: {name}")
            else:
                result.tools.append(str(name))
            if not isinstance(call.get("arguments"), dict):
                result.errors.append(f"tool_call arguments is not an object for {name}")

    for message in messages:
        content = str(message.get("content") or "")
        for fragment in TOOL_RESPONSE_RE.findall(content):
            response = _json_loads_fragment(fragment, "tool_response", result.errors)
            if response is not None and not isinstance(response, (dict, list)):
                result.errors.append("tool_response is not a JSON object or array")

    final_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if final_assistant is None:
        result.errors.append("no assistant message")
    elif not ANSWER_RE.search(str(final_assistant.get("content") or "")):
        result.errors.append("final assistant message does not contain <answer>...</answer>")

    image_tags = sum(str(m.get("content") or "").count("<image>") for m in messages)
    if result.image_count > 1 and image_tags != result.image_count:
        result.errors.append(f"multi-image <image> count mismatch: tags={image_tags}, images={result.image_count}")
    elif result.image_count == 1 and image_tags < 1:
        result.errors.append("single-image sample has no <image> placeholder")

    return result


def validate_file(path: Path, max_invalid_uids: int = 20) -> dict[str, Any]:
    seen_uids: set[str] = set()
    by_task: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    invalid_uids: list[str] = []
    total = 0
    invalid = 0
    multi_image_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                if len(invalid_uids) < max_invalid_uids:
                    invalid_uids.append(f"line:{line_no}:json:{exc}")
                continue

            result = validate_item(item, seen_uids)
            by_task[result.task_type] += 1
            by_tool.update(result.tools)
            if result.image_count > 1:
                multi_image_count += 1
            if result.errors:
                invalid += 1
                if len(invalid_uids) < max_invalid_uids:
                    invalid_uids.append(f"{result.uid or 'line:' + str(line_no)}: {'; '.join(result.errors)}")

    return {
        "total": total,
        "by_task": dict(sorted(by_task.items())),
        "by_tool": dict(sorted(by_tool.items())),
        "multi_image_count": multi_image_count,
        "invalid_count": invalid,
        "invalid_sample_uid": invalid_uids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/visual_tool_sft_v0.jsonl")
    parser.add_argument("--max-invalid-uids", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_file(Path(args.input), args.max_invalid_uids)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    if summary["invalid_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
