#!/usr/bin/env python3
"""Export DeepSeek-accepted trajectories as image-complete SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ModuleNotFoundError:
    modelarts_sdk = Path("/opt/huawei/modelarts-dev/modelarts-sdk")
    if not modelarts_sdk.is_dir():
        raise
    sys.path.insert(0, str(modelarts_sdk))
    from PIL import Image


BASE = Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx")
REPO_ROOT = BASE / "visual-agent"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACCEPTED = SCRIPT_DIR / "outputs/deepseek_v41/accepted.jsonl"
DEFAULT_DECISIONS = SCRIPT_DIR / "outputs/deepseek_v41/decisions.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/rl_distill/qwen3_mixed_rl_deepseek_accepted_2394_sft.jsonl"
DEFAULT_CROP_DIR = REPO_ROOT / "data/rl_distill/qwen3_mixed_rl_deepseek_accepted_2394_crops"
DEFAULT_REPORT = REPO_ROOT / "data/rl_distill/qwen3_mixed_rl_deepseek_accepted_2394_report.json"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>.*", re.DOTALL)
SUPPORTED_TOOLS = {"grounding_detect", "crop_zoom"}
QUALITY_STATUS = "deepseek_v41_visual_judge_accepted_not_human_verified"


def validate_box(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("invalid_bbox")
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in value):
        raise ValueError("invalid_bbox")
    if any(number < 0 or number > 1000 for number in value):
        raise ValueError("invalid_bbox")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("invalid_bbox")


def relative_crop_rect(
    bbox: Any,
    image_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    validate_box(bbox)
    width, height = image_size
    left, top, right, bottom = [float(number) for number in bbox]
    pixel_left = max(0, min(width - 1, round(left / 1000 * width)))
    pixel_top = max(0, min(height - 1, round(top / 1000 * height)))
    expected_width, expected_height = expected_size
    if expected_width > width or expected_height > height:
        raise ValueError(f"crop_larger_than_source:{expected_size}:{image_size}")
    # Recorded relative coordinates are rounded to four decimals. Recover the
    # exact integer crop extent from the tool's recorded output dimensions.
    pixel_left = min(pixel_left, width - expected_width)
    pixel_top = min(pixel_top, height - expected_height)
    rect = (
        pixel_left,
        pixel_top,
        pixel_left + expected_width,
        pixel_top + expected_height,
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError(f"invalid_crop_rectangle:{rect}")
    return rect


def jpeg_roundtrip(image: Image.Image) -> tuple[bytes, Image.Image]:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    payload = buffer.getvalue()
    return payload, Image.open(io.BytesIO(payload)).convert("RGB")


def rewrite_crop_paths(value: Any, uri: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"path", "crop_path"} and isinstance(child, str) and child.startswith("tool://"):
                value[key] = uri
            else:
                rewrite_crop_paths(child, uri)
    elif isinstance(value, list):
        for child in value:
            rewrite_crop_paths(child, uri)


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            decision = json.loads(line)
            sample_id = decision.get("sample_id")
            if not isinstance(sample_id, str) or "-" not in sample_id:
                raise ValueError(f"invalid decision sample_id at line {line_number}")
            digest = sample_id.rsplit("-", 1)[-1]
            if digest in decisions:
                raise ValueError(f"duplicate decision digest: {digest}")
            decisions[digest] = decision
    return decisions


def compact_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        key: decision.get(key)
        for key in (
            "sample_id",
            "model",
            "prompt_version",
            "verdict",
            "provider_verdict",
            "confidence",
            "entity_queries_valid",
            "detections_visually_correct",
            "tool_sequence_reasonable",
            "evidence_supports_answer",
            "final_explanation_consistent",
            "reason_codes",
            "brief_reason",
        )
    }


def convert_item(
    item: dict[str, Any],
    decision: dict[str, Any],
    crop_dir: Path,
) -> tuple[dict[str, Any], Counter[str], list[tuple[Path, bytes]]]:
    if decision.get("verdict") != "accept":
        raise ValueError("non_accepted_decision")
    if "answer_leakage_in_query" in (decision.get("reason_codes") or []):
        raise ValueError("accepted_answer_leakage")
    messages = item.get("messages")
    source_images = item.get("images")
    metadata = item.get("metadata")
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError("invalid_messages")
    if not isinstance(source_images, list) or len(source_images) != 1:
        raise ValueError("expected_one_judge_image")
    if not isinstance(metadata, dict):
        raise ValueError("missing_metadata")
    original_path = Path(str(source_images[0]))
    if not original_path.is_file():
        raise FileNotFoundError(original_path)
    expected_roles = ["system"] + ["user" if index % 2 else "assistant" for index in range(1, len(messages))]
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError("invalid_message_roles")

    source_key = (
        f"{metadata.get('data_source')}:{metadata.get('source_image')}:"
        f"{metadata.get('source_index')}"
    )
    uid = hashlib.sha256(source_key.encode()).hexdigest()[:20]
    output_messages = [json.loads(json.dumps(messages[0])), json.loads(json.dumps(messages[1]))]
    images = [str(original_path)]
    image_objects: list[Image.Image | None] = [None]
    pending_crops: list[tuple[Path, bytes]] = []
    counts: Counter[str] = Counter()

    for index in range(2, len(messages) - 1, 2):
        assistant = messages[index]
        observation = messages[index + 1]
        call_match = TOOL_CALL_RE.search(str(assistant.get("content") or ""))
        response_match = TOOL_RESPONSE_RE.fullmatch(str(observation.get("content") or ""))
        if call_match is None or response_match is None:
            raise ValueError(f"invalid_tool_pair:{index}")
        call = json.loads(call_match.group(1))
        response = json.loads(response_match.group(1))
        if not isinstance(call, dict) or not isinstance(response, dict):
            raise ValueError("invalid_tool_json")
        name = call.get("name")
        arguments = call.get("arguments")
        if name not in SUPPORTED_TOOLS or not isinstance(arguments, dict):
            raise ValueError("unsupported_tool")
        target_image = arguments.get("target_image", 0)
        if type(target_image) is not int or not 0 <= target_image < len(images):
            raise ValueError("invalid_target_image")
        output_messages.append(json.loads(json.dumps(assistant)))
        counts[f"tool:{name}"] += 1

        suffix = ""
        if name == "crop_zoom":
            crop_info = response.get("crop_zoom")
            if not isinstance(crop_info, dict):
                raise ValueError("invalid_crop_response")
            crop_target = crop_info.get("target_image")
            if type(crop_target) is not int or crop_target != len(images):
                raise ValueError("invalid_crop_target")
            if image_objects[target_image] is None:
                with Image.open(images[target_image]) as opened:
                    image_objects[target_image] = opened.convert("RGB")
            outputs = crop_info.get("image_outputs")
            if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
                raise ValueError("invalid_crop_outputs")
            reported_size = (outputs[0].get("width"), outputs[0].get("height"))
            if not all(type(number) is int and number > 0 for number in reported_size):
                raise ValueError("invalid_crop_size")
            crop = image_objects[target_image].crop(
                relative_crop_rect(
                    crop_info.get("bbox_2d"),
                    image_objects[target_image].size,
                    reported_size,
                )
            )
            payload, decoded_crop = jpeg_roundtrip(crop)
            if crop.size != reported_size:
                raise ValueError(f"crop_size_mismatch:{crop.size}:{reported_size}")
            crop_path = crop_dir / f"{uid}_image{crop_target}.jpg"
            uri = f"tool://images/{crop_target}/{crop_path.name}"
            rewrite_crop_paths(response, uri)
            pending_crops.append((crop_path, payload))
            images.append(str(crop_path))
            image_objects.append(decoded_crop)
            suffix = "\ncrop_zoom returned a crop: <image>"
            counts["crop_images"] += 1

        response_text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        output_messages.append({
            "role": "user",
            "content": f"<tool_response>\n{response_text}\n</tool_response>{suffix}",
        })

    output_messages.append(json.loads(json.dumps(messages[-1])))
    placeholders = sum(str(message["content"]).count("<image>") for message in output_messages)
    if placeholders != len(images):
        raise ValueError(f"image_placeholder_mismatch:{placeholders}:{len(images)}")
    if len(set(images)) != len(images):
        raise ValueError("duplicate_image_path")

    output_metadata = json.loads(json.dumps(metadata))
    output_metadata["quality_status"] = QUALITY_STATUS
    output_metadata["visual_judge"] = compact_decision(decision)
    return {
        "messages": output_messages,
        "images": images,
        "metadata": output_metadata,
    }, counts, pending_crops


def export(args: argparse.Namespace) -> dict[str, Any]:
    for required in (args.accepted, args.decisions):
        if not required.is_file():
            raise FileNotFoundError(required)
    for target in (args.output, args.report, args.crop_dir):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)

    decisions = load_decisions(args.decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.crop_dir.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    pending_crops: list[tuple[Path, bytes]] = []
    seen_sources: set[tuple[Any, ...]] = set()

    with args.accepted.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            digest = hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()[:16]
            decision = decisions.get(digest)
            if decision is None:
                raise ValueError(f"missing decision for accepted line {line_number}")
            item = json.loads(line)
            metadata = item.get("metadata") or {}
            source_key = (
                metadata.get("data_source"),
                metadata.get("source_image"),
                metadata.get("source_index"),
            )
            if source_key in seen_sources:
                raise ValueError(f"duplicate source: {source_key}")
            seen_sources.add(source_key)
            converted, item_counts, item_crops = convert_item(item, decision, args.crop_dir)
            rows.append(converted)
            pending_crops.extend(item_crops)
            counts.update(item_counts)
            counts["rows"] += 1
            counts[f"ability:{metadata.get('ability')}"] += 1
            counts[f"run:{metadata.get('rollout_run')}"] += 1

    if counts["rows"] != 2394:
        raise ValueError(f"expected 2394 accepted rows, got {counts['rows']}")
    if counts["crop_images"] != len(pending_crops):
        raise ValueError("crop accounting mismatch")

    for path, payload in pending_crops:
        path.write_bytes(payload)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "passed",
        "input_accepted": str(args.accepted.resolve()),
        "input_decisions": str(args.decisions.resolve()),
        "output": str(args.output.resolve()),
        "crop_dir": str(args.crop_dir.resolve()),
        "quality_status": QUALITY_STATUS,
        "counts": dict(sorted(counts.items())),
        "unique_sources": len(seen_sources),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
