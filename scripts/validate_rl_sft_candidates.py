#!/usr/bin/env python3
"""Validate offline RL-derived SFT candidates without changing their contents."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
REWARD_SPEC = importlib.util.spec_from_file_location("candidate_validation_reward", REWARD_PATH)
reward = importlib.util.module_from_spec(REWARD_SPEC)
assert REWARD_SPEC.loader is not None
REWARD_SPEC.loader.exec_module(reward)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>(.*)", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>([^<>]+)</answer>")


class InvalidCandidate(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InvalidCandidate(code, detail)


@lru_cache(maxsize=65536)
def _resolved_image_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def image_path(value: Any) -> str:
    require(isinstance(value, str) and bool(value), "invalid_image_path")
    require("://" not in value and not value.startswith("data:"), "non_filesystem_image_path")
    return _resolved_image_path(value)


class ImageChecks:
    def __init__(self):
        self.exists: dict[str, bool] = {}
        self.crop_sizes: dict[str, tuple[int, int]] = {}

    def check_exists(self, path: str) -> None:
        if path not in self.exists:
            self.exists[path] = Path(path).is_file()
        require(self.exists[path], "missing_image", path)

    def crop_size(self, path: str) -> tuple[int, int]:
        self.check_exists(path)
        if path not in self.crop_sizes:
            try:
                with Image.open(path) as image:
                    size = image.size
                    image.verify()
            except (OSError, ValueError) as exc:
                raise InvalidCandidate("invalid_crop_image", f"{path}: {exc}") from exc
            self.crop_sizes[path] = size
        return self.crop_sizes[path]


def load_sources(train_path: Path, val_path: Path):
    columns = ["source_index", "source_image", "images", "question", "solution"]
    sources: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(train_path, columns=columns).to_pylist():
        sources[row["source_index"]].append(row)
    val_rows = pq.read_table(val_path, columns=["source_index", "source_image", "images"]).to_pylist()
    val_indices = {row["source_index"] for row in val_rows}
    val_source_images = {row["source_image"] for row in val_rows if row["source_image"]}
    val_images = {image_path(path) for row in val_rows for path in row["images"] or []}
    return sources, val_indices, val_source_images, val_images


def valid_box(box: Any) -> bool:
    return (
        isinstance(box, list) and len(box) == 4
        and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000 for value in box)
        and box[0] < box[2] and box[1] < box[3]
    )


def parse_json_object(text: str, code: str) -> dict[str, Any]:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidCandidate(code, str(exc)) from exc
    require(isinstance(result, dict), code)
    return result


def validate_candidate(row, sources, val_indices, val_source_images, val_images, seen_indices, images):
    require(isinstance(row, dict), "invalid_candidate_object")
    metadata = row.get("metadata")
    require(isinstance(metadata, dict), "missing_metadata")
    source_index = metadata.get("source_index")
    require(type(source_index) is int and source_index >= 0, "invalid_source_index")
    require(source_index not in seen_indices, "duplicate_source_index", str(source_index))
    seen_indices.add(source_index)
    require(source_index not in val_indices, "validation_source_index_overlap", str(source_index))
    matches = sources.get(source_index, [])
    require(len(matches) == 1, "non_unique_training_source", str(source_index))
    source = matches[0]
    require(isinstance(metadata.get("source_image"), str) and bool(metadata["source_image"]), "missing_source_image")
    require(metadata["source_image"] == source["source_image"], "source_image_mismatch")
    require(metadata["source_image"] not in val_source_images, "validation_source_image_overlap")
    for key in ("rollout_file", "original_final", "quality_status"):
        require(isinstance(metadata.get(key), str) and bool(metadata[key]), "missing_provenance", key)
    for key, minimum in (("rollout_line", 1), ("rollout_step", 0)):
        require(type(metadata.get(key)) is int and metadata[key] >= minimum, "missing_provenance", key)
    flags = metadata.get("review_flags")
    require(isinstance(flags, list) and all(isinstance(flag, str) for flag in flags), "missing_review_flags")

    paths = row.get("images")
    require(isinstance(paths, list) and bool(paths), "missing_images")
    paths = [image_path(path) for path in paths]
    original_images = source.get("images") or []
    require(len(original_images) == 1 and paths[0] == image_path(original_images[0]), "original_image_mismatch")
    require(paths[0] not in val_images, "validation_original_image_overlap", paths[0])
    require(len(set(paths)) == len(paths), "duplicate_image_reference")
    for path in paths:
        images.check_exists(path)

    messages = row.get("messages")
    require(isinstance(messages, list) and len(messages) >= 3 and len(messages) % 2 == 1, "invalid_message_sequence")
    for index, message in enumerate(messages):
        expected = "system" if index == 0 else ("assistant" if index % 2 == 0 else "user")
        require(isinstance(message, dict) and message.get("role") == expected, "invalid_message_role", str(index))
        require(isinstance(message.get("content"), str) and bool(message["content"].strip()), "empty_message", str(index))
    require(messages[0]["content"].count("<image>") == 0, "system_image_placeholder")
    require(messages[1]["content"] == f"<image>\n{str(source['question']).strip()}", "question_mismatch")
    require(sum(message["content"].count("<image>") for message in messages) == len(paths), "image_placeholder_count_mismatch")
    final = ANSWER_RE.fullmatch(messages[-1]["content"])
    require(final is not None, "invalid_final_answer_format")
    answer = final.group(1)
    require(answer in reward.RELATION_LABELS, "noncanonical_answer")
    require(answer == reward.normalize_relation_answer(source["solution"]), "ground_truth_mismatch")

    image_count = 1
    tool_counts: Counter[str] = Counter()
    for index in range(2, len(messages) - 1, 2):
        call_match = TOOL_CALL_RE.fullmatch(messages[index]["content"])
        response_match = TOOL_RESPONSE_RE.fullmatch(messages[index + 1]["content"])
        require(call_match is not None and response_match is not None, "unmatched_tool_pair", str(index))
        call = parse_json_object(call_match.group(1), "invalid_tool_call_json")
        response = parse_json_object(response_match.group(1), "invalid_tool_response_json")
        name = call.get("name")
        arguments = call.get("arguments")
        require(isinstance(name, str) and name in {"grounding_detect", "crop_zoom"} and isinstance(arguments, dict), "unsupported_tool_call")
        target_image = arguments.get("target_image")
        require(type(target_image) is int and 0 <= target_image < image_count, "invalid_target_image")
        require("error" not in response and response.get("status") != "error", "tool_error_response")
        require("<image>" not in messages[index]["content"], "assistant_image_placeholder")
        tool_counts[name] += 1
        if name == "grounding_detect":
            query = arguments.get("query")
            require(isinstance(query, str) and bool(query.strip()), "invalid_grounding_query")
            require(isinstance(response.get("query"), str) and reward.normalize_answer(query) == reward.normalize_answer(response["query"]), "detection_query_mismatch")
            require(response.get("source") == "groundingdino", "tool_response_source_mismatch")
            require(not response_match.group(2).strip(), "unexpected_detection_suffix")
            require(response.get("coordinate_space") == "relative_0_1000", "invalid_coordinate_space")
            arrays = [response.get(key) for key in ("boxes", "labels", "confidence")]
            require(all(isinstance(value, list) for value in arrays), "invalid_detection_response")
            require(type(response.get("count")) is int and all(len(value) == response["count"] for value in arrays), "invalid_detection_response")
            require(all(valid_box(box) for box in response["boxes"]), "invalid_detection_box")
            continue

        require(response.get("source") == "crop_zoom", "tool_response_source_mismatch")
        require(type(response.get("target_image")) is int and response["target_image"] == target_image, "crop_source_image_mismatch")
        crop = response.get("crop_zoom")
        require(isinstance(crop, dict) and type(crop.get("target_image")) is int and crop["target_image"] == image_count, "invalid_crop_target_image")
        require(image_count < len(paths), "missing_crop_image_reference")
        require(valid_box(arguments.get("bbox_2d")) and valid_box(crop.get("bbox_2d")), "invalid_crop_box")
        requested = crop.get("requested_bbox_2d")
        require(valid_box(requested) and all(math.isclose(a, b, rel_tol=0, abs_tol=0.001) for a, b in zip(requested, arguments["bbox_2d"])), "crop_request_mismatch")
        require(response.get("coordinate_space") == crop.get("coordinate_space") == "relative_0_1000", "invalid_coordinate_space")
        require(response_match.group(2).count("<image>") == 1, "crop_image_placeholder_mismatch")
        require("<tool_" not in response_match.group(2), "unmatched_tool_pair")
        expected_uri = f"tool://images/{image_count}/{Path(paths[image_count]).name}"
        require(crop.get("crop_path") == expected_uri, "crop_path_mismatch")
        size = images.crop_size(paths[image_count])
        outputs_lists = [crop.get("image_outputs")]
        if "image_outputs" in response:
            outputs_lists.append(response["image_outputs"])
        for outputs in outputs_lists:
            require(isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], dict), "invalid_crop_outputs")
            output = outputs[0]
            require(type(output.get("target_image")) is int and output["target_image"] == image_count, "crop_output_target_mismatch")
            require(output.get("path") == expected_uri, "crop_path_mismatch")
            require(all(type(output.get(key)) is int and output[key] > 0 for key in ("width", "height")), "invalid_crop_dimensions")
            require((output["width"], output["height"]) == size, "crop_dimensions_mismatch", paths[image_count])
        image_count += 1
    require(image_count == len(paths), "unconsumed_image_reference")
    return answer, tool_counts, paths[0]


def validate(input_path: Path, train_parquet: Path, val_parquet: Path) -> dict[str, Any]:
    sources, val_indices, val_source_images, val_images = load_sources(train_parquet, val_parquet)
    seen_indices: set[int] = set()
    checks = ImageChecks()
    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    examples = []
    originals = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                row = parse_json_object(line, "invalid_json_line")
                answer, tool_counts, original = validate_candidate(
                    row, sources, val_indices, val_source_images, val_images, seen_indices, checks,
                )
            except InvalidCandidate as exc:
                counts["invalid_rows"] += 1
                errors[exc.code] += 1
                if len(examples) < 20:
                    examples.append({"line": line_number, "code": exc.code, "detail": str(exc)[:400]})
                continue
            counts["valid_rows"] += 1
            counts["direct_answer_rows"] += int(not tool_counts)
            counts["crop_references"] += tool_counts["crop_zoom"]
            labels[answer] += 1
            tools.update(tool_counts)
            originals.add(original)
    if not counts["rows"]:
        errors["empty_input"] += 1
    return {
        "status": "passed" if not errors else "failed",
        "input": str(input_path),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "scope": "Structure, image references, source identity and validation isolation; visual reasoning quality is not assessed.",
        "counts": {
            **{key: counts[key] for key in ("rows", "valid_rows", "invalid_rows", "direct_answer_rows", "crop_references")},
            "unique_original_images": len(originals),
            "verified_crop_files": len(checks.crop_sizes),
        },
        "label_counts": dict(sorted(labels.items())),
        "tool_counts": dict(sorted(tools.items())),
        "errors": dict(sorted(errors.items())),
        "error_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    if args.report_json.exists():
        raise FileExistsError(f"Report already exists: {args.report_json}")
    report = validate(args.input, args.train_parquet, args.val_parquet)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "errors": report["errors"], "report": str(args.report_json)}, sort_keys=True))
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
