#!/usr/bin/env python
"""Convert naturalized visual-tool trajectories into ShareGPT SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .visual_tools import get_visual_tool_schemas
except ImportError:  # pragma: no cover - used when invoked as a script
    from visual_tools import get_visual_tool_schemas


DEFAULT_INPUTS = {
    "counting": "/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_counting_sam3_dino_trajectories.natural.jsonl",
    "spatial": "/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_spatial_sam3_trajectories.natural.jsonl",
    "attribute": "/data_cinema/gx/twi_pilot_data/outputs/gemini-api/final_attribute_sam3_crop_zoom_trajectories.natural.jsonl",
}

DEFAULT_OUTPUT = "data/visual_tool_sft_v0.jsonl"
DEFAULT_SMOKE_OUTPUT = "data/visual_tool_sft_v0.smoke.jsonl"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def _normalize_images(item: dict[str, Any]) -> list[str]:
    images = item.get("images") or []
    if images:
        return [str(image) for image in images]
    image = item.get("image")
    return [str(image)] if image else []


def _ensure_image_placeholders(messages: list[dict[str, str]], image_count: int) -> list[dict[str, str]]:
    if image_count == 0:
        return messages

    current = sum(message["content"].count("<image>") for message in messages)
    if current >= image_count:
        return messages

    normalized = [dict(message) for message in messages]
    missing = image_count - current
    normalized[0]["content"] = "<image>\n" * missing + normalized[0]["content"]
    return normalized


def _merge_adjacent_same_role(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            previous = merged[-1]["content"].rstrip()
            current = message["content"].lstrip()
            merged[-1]["content"] = f"{previous}\n{current}" if previous and current else previous or current
        else:
            merged.append(dict(message))
    return merged


def _to_train_messages(raw_messages: list[dict[str, Any]], image_count: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            stripped = content.strip()
            if stripped.startswith("<tool_response>") and stripped.endswith("</tool_response>"):
                messages.append({"role": "user", "content": stripped})
            else:
                messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
        elif role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    messages = _merge_adjacent_same_role(messages)
    return _ensure_image_placeholders(messages, image_count)


def convert_item(item: dict[str, Any], task_type: str, uid: str | None = None) -> dict[str, Any]:
    images = _normalize_images(item)
    raw_messages = [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in item.get("messages") or []]
    converted = {
        "uid": uid or item.get("uid"),
        "source_uid": item.get("uid"),
        "task_type": task_type,
        "image": item.get("image"),
        "images": images,
        "question": item.get("question"),
        "answer": item.get("answer"),
        "required_tools": item.get("required_tools") or [],
        "messages": _to_train_messages(raw_messages, len(images)),
        "messages_raw": raw_messages,
        "tools": get_visual_tool_schemas(),
    }
    if "image_meta" in item:
        converted["image_meta"] = item["image_meta"]
    if "data_type" in item:
        converted["source_data_type"] = item["data_type"]
    return converted


def convert_files(inputs: dict[str, Path], output: Path, smoke_output: Path | None = None, smoke_samples: int = 100) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if smoke_output is not None:
        smoke_output.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    smoke_written = 0
    total = 0
    seen_uids: set[str] = set()
    with output.open("w", encoding="utf-8") as out_handle:
        smoke_handle = smoke_output.open("w", encoding="utf-8") if smoke_output is not None else None
        try:
            for task_type, path in inputs.items():
                counts[task_type] = 0
                for item in _read_jsonl(path):
                    source_uid = str(item.get("uid") or f"{task_type}:{counts[task_type]}")
                    uid = source_uid if source_uid not in seen_uids else f"{task_type}:{source_uid}"
                    seen_uids.add(uid)
                    converted = convert_item(item, task_type=task_type, uid=uid)
                    line = json.dumps(converted, ensure_ascii=False)
                    out_handle.write(line + "\n")
                    total += 1
                    counts[task_type] += 1
                    if smoke_handle is not None and smoke_written < smoke_samples:
                        smoke_handle.write(line + "\n")
                        smoke_written += 1
        finally:
            if smoke_handle is not None:
                smoke_handle.close()

    counts["total"] = total
    if smoke_output is not None:
        counts["smoke"] = smoke_written
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counting", default=DEFAULT_INPUTS["counting"])
    parser.add_argument("--spatial", default=DEFAULT_INPUTS["spatial"])
    parser.add_argument("--attribute", default=DEFAULT_INPUTS["attribute"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output", default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--smoke-samples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "counting": Path(args.counting),
        "spatial": Path(args.spatial),
        "attribute": Path(args.attribute),
    }
    counts = convert_files(inputs, Path(args.output), Path(args.smoke_output), args.smoke_samples)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
