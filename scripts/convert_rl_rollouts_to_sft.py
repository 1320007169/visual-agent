#!/usr/bin/env python3
"""Convert successful visual-agent RL rollouts into Qwen3-VL SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

try:
    from PIL import Image
except ModuleNotFoundError:
    modelarts_sdk = Path("/opt/huawei/modelarts-dev/modelarts-sdk")
    if not modelarts_sdk.is_dir():
        raise
    sys.path.insert(0, str(modelarts_sdk))
    from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_DIR = Path(
    "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts/"
    "visual-agent-zwz-rl/"
    "zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual"
)
DEFAULT_TRAIN_PARQUET = ROOT / "data/zwz_rl_vqa/rl_original_relation/train.parquet"
DEFAULT_OUTPUT = ROOT / "data/rl_distill/qwen3_groundingdino_kl_safe_success_sft.jsonl"

# Load the same scorer used by RL without importing the GPU training package.
REWARD_PATH = ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
REWARD_SPEC = importlib.util.spec_from_file_location("relation_rollout_reward", REWARD_PATH)
reward = importlib.util.module_from_spec(REWARD_SPEC)
assert REWARD_SPEC.loader is not None
REWARD_SPEC.loader.exec_module(reward)

ROLE_RE = re.compile(r"\n(user|assistant)\n")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>\s*$", re.DOTALL | re.IGNORECASE)
SUPPORTED_TOOLS = {"grounding_detect", "crop_zoom"}
RELATIONAL_QUERY_RE = re.compile(
    r"\b(?:left of|right of|above|below|behind|in front of|next to|between|"
    r"located|positioned|relative to|left side of|right side of)\b|"
    r"(?:\u4f4d\u4e8e|\u5de6\u4fa7|\u53f3\u4fa7|\u4e0a\u65b9|\u4e0b\u65b9|\u65c1\u8fb9|\u524d\u65b9|\u540e\u65b9)",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    question: str
    step: int
    output: str
    turns: list[dict[str, str]]
    calls: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    source: dict[str, Any]
    system_prompt: str
    answer: str
    review_flags: list[str]
    rollout_file: str = ""
    rollout_line: int = 0
    logged_acc: float | None = None
    logged_score: float | None = None

    @property
    def rank(self) -> tuple[int, int, int, int]:
        return len(self.review_flags), len(self.calls), len(self.output), -self.step


def extract_question(input_text: str) -> str:
    marker = "\nuser\n"
    if marker not in input_text:
        raise ValueError("rollout input has no user turn")
    question = input_text.rsplit(marker, 1)[1]
    if question.endswith("\nassistant\n"):
        question = question[: -len("\nassistant\n")]
    elif question.endswith("\nassistant"):
        question = question[: -len("\nassistant")]
    return question.strip()


def normalize_answer(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return normalized.strip(" \t\r\n.,;:!?")


def answer_assessment(prediction: str | None, ground_truth: str) -> dict[str, Any]:
    canonical = reward.normalize_relation_answer(prediction)
    if prediction is None or not prediction.strip():
        status = "missing_answer"
    elif canonical is None:
        status = "unmapped_answer"
    elif canonical == reward.normalize_relation_answer(ground_truth):
        status = "label_match"
    else:
        status = "different_label"
    return {"raw_answer": prediction, "canonical_answer": canonical,
            "ground_truth": ground_truth, "label_status": status,
            "semantic_verdict": None}


def sample_semantic_reviews(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for entry in entries:
        stratum = (entry["ground_truth"], entry["label_status"],
                   "on" if normalize_answer(entry["raw_answer"]) == "on" else "other",
                   "early" if entry["rollout_step"] <= 100 else "late")
        groups[stratum].append(entry)
    for group in groups.values():
        group.sort(key=lambda entry: hashlib.sha256(
            f"{entry['source_index']}:{entry['raw_answer']}:{entry['rollout_step']}".encode()
        ).hexdigest())
    selected = []
    seen_sources = set()
    while len(selected) < limit:
        added = False
        for key in sorted(groups):
            while groups[key]:
                entry = groups[key].pop()
                if entry["source_index"] in seen_sources:
                    continue
                selected.append(entry)
                seen_sources.add(entry["source_index"])
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def resolve_source(
    row: dict[str, Any], source_rows: dict[str, list[dict[str, Any]]]
) -> tuple[str, dict[str, Any]]:
    question = extract_question(str(row.get("input") or ""))
    matches = source_rows.get(question) or []
    metadata = row.get("source_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("invalid_source_metadata")
    source_index = metadata.get("source_index")
    if source_index is not None:
        if type(source_index) is not int:
            raise ValueError("invalid_source_index")
        matches = [source for source in matches if source["source_index"] == source_index]
    if not matches:
        raise ValueError("source_mapping_failed")
    # A model prediction cannot establish which image the model actually saw.
    if len(matches) != 1:
        raise ValueError("ambiguous_source_mapping")
    source = matches[0]
    for key, expected in (("question", question), ("ground_truth", source["solution"]),
                          ("source_image", source.get("source_image"))):
        if key in metadata and expected is not None and metadata[key] != expected:
            raise ValueError("source_metadata_mismatch")
    if metadata.get("dataset_split", "train") != "train":
        raise ValueError("non_training_source")
    return question, source


def validate_box(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("invalid_bbox")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in value):
        raise ValueError("invalid_bbox")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("invalid_bbox")


def query_word_count(query: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", query, flags=re.UNICODE))


def parse_turns(output: str) -> list[dict[str, str]]:
    parts = ROLE_RE.split(output.strip())
    turns = [{"role": "assistant", "content": parts[0].strip()}]
    for index in range(1, len(parts), 2):
        turns.append({"role": parts[index], "content": parts[index + 1].strip()})
    expected = ["assistant" if index % 2 == 0 else "user" for index in range(len(turns))]
    if [turn["role"] for turn in turns] != expected or turns[-1]["role"] != "assistant":
        raise ValueError("rollout turns do not alternate assistant/user")
    return turns


def parse_candidate(
    row: dict[str, Any],
    source_rows: dict[str, list[dict[str, Any]]],
    max_query_words: int,
) -> Candidate:
    question, source = resolve_source(row, source_rows)
    output = str(row.get("output") or "")
    turns = parse_turns(output)
    if re.search(r"</?tool_(?:call|response)\b", turns[-1]["content"], re.IGNORECASE):
        raise ValueError("unfinished_tool_turn")
    answer_match = ANSWER_RE.search(turns[-1]["content"])
    if answer_match is None or len(re.findall(r"<answer>", output, re.IGNORECASE)) != 1:
        raise ValueError("missing_final_answer")
    answer = reward.normalize_relation_answer(answer_match.group(1))
    if answer is None:
        raise ValueError("invalid_relation_label")
    if answer != reward.normalize_relation_answer(source["solution"]):
        raise ValueError("answer_mismatch")
    input_text = str(row["input"])
    system_prompt = input_text.split("\nuser\n", 1)[0].removeprefix("system\n").strip()
    if not input_text.startswith("system\n") or not system_prompt:
        raise ValueError("missing_system_prompt")

    calls = []
    responses = []
    image_count = 1
    seen_calls = set()
    review_flags = set()
    detected_count = 0
    for index in range(0, len(turns) - 1, 2):
        call_matches = TOOL_CALL_RE.findall(turns[index]["content"])
        response_matches = TOOL_RESPONSE_RE.findall(turns[index + 1]["content"])
        if len(call_matches) != 1 or len(response_matches) != 1:
            raise ValueError("incomplete_tool_pair")
        if TOOL_CALL_RE.fullmatch(turns[index]["content"]) is None:
            raise ValueError("noncanonical_tool_turn")
        call = json.loads(call_matches[0])
        response = json.loads(response_matches[0])
        if not isinstance(call, dict) or not isinstance(response, dict):
            raise ValueError("invalid_tool_pair")
        name = call.get("name")
        arguments = call.get("arguments")
        if name not in SUPPORTED_TOOLS or not isinstance(arguments, dict):
            raise ValueError("unsupported_tool_call")
        if response.get("status") == "error" or "error" in response:
            raise ValueError("tool_error")
        target_image = arguments.get("target_image", 0)
        if type(target_image) is not int or target_image < 0 or target_image >= image_count:
            raise ValueError("invalid_target_image")
        arguments["target_image"] = target_image
        if name == "grounding_detect":
            if set(arguments) - {"query", "target_image"}:
                raise ValueError("unexpected_tool_arguments")
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("invalid_grounding_query")
            if max_query_words > 0 and query_word_count(query) > max_query_words:
                raise ValueError("long_grounding_query")
            if RELATIONAL_QUERY_RE.search(query) or "?" in query or "\n" in query:
                raise ValueError("relational_grounding_query")
            if normalize_answer(response.get("query")) != normalize_answer(query):
                raise ValueError("tool_response_query_mismatch")
            if response.get("coordinate_space") != "relative_0_1000":
                raise ValueError("unknown_coordinate_space")
            boxes = response.get("boxes")
            labels = response.get("labels")
            confidence = response.get("confidence")
            if not all(isinstance(values, list) for values in (boxes, labels, confidence)):
                raise ValueError("invalid_detection_response")
            if type(response.get("count")) is not int or not (len(boxes) == len(labels) == len(confidence) == response["count"]):
                raise ValueError("invalid_detection_response")
            if any(not isinstance(label, str) or not label.strip() for label in labels):
                raise ValueError("invalid_detection_response")
            for box in boxes:
                validate_box(box)
            if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in confidence):
                raise ValueError("invalid_detection_confidence")
            detected_count += len(boxes)
            if not boxes:
                review_flags.add("empty_detection")
            if len(boxes) > 1:
                review_flags.add("multiple_detection_boxes")
        if name == "crop_zoom":
            if set(arguments) - {"bbox_2d", "target_image", "label", "slack_ratio"}:
                raise ValueError("unexpected_tool_arguments")
            validate_box(arguments.get("bbox_2d"))
            slack = arguments.get("slack_ratio", 0.1)
            if type(slack) not in (int, float) or not math.isfinite(slack) or slack < 0:
                raise ValueError("invalid_slack_ratio")
            crop_info = response.get("crop_zoom")
            if not isinstance(crop_info, dict) or type(crop_info.get("target_image")) is not int or crop_info["target_image"] != image_count:
                raise ValueError("invalid_crop_response")
            if response.get("coordinate_space") != "relative_0_1000" or crop_info.get("coordinate_space") != "relative_0_1000":
                raise ValueError("unknown_coordinate_space")
            requested_box = crop_info.get("requested_bbox_2d")
            validate_box(requested_box)
            if type(response.get("target_image")) is not int or response["target_image"] != target_image or any(
                not math.isclose(recorded, requested, rel_tol=0, abs_tol=0.001)
                for recorded, requested in zip(requested_box, arguments["bbox_2d"])
            ):
                raise ValueError("crop_response_call_mismatch")
            outputs = crop_info.get("image_outputs")
            if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
                raise ValueError("invalid_crop_image_metadata")
            if outputs[0].get("target_image") != image_count or any(
                type(outputs[0].get(key)) is not int or outputs[0][key] <= 0 for key in ("width", "height")
            ):
                raise ValueError("invalid_crop_image_metadata")
            validate_box(crop_info.get("bbox_2d"))
            if crop_info["bbox_2d"] == [0, 0, 1000, 1000]:
                raise ValueError("full_image_crop")
            image_count += 1
        call_key = json.dumps(call, sort_keys=True, ensure_ascii=False)
        if call_key in seen_calls:
            raise ValueError("duplicate_tool_call")
        seen_calls.add(call_key)
        calls.append(call)
        responses.append(response)

    if len(turns) != 2 * len(calls) + 1:
        raise ValueError("incomplete_trajectory")
    if calls and not detected_count and image_count == 1:
        raise ValueError("all_detections_empty_without_recovery")

    return Candidate(
        question=question,
        step=int(row.get("step", 0)),
        output=output,
        turns=turns,
        calls=calls,
        responses=responses,
        source=source,
        system_prompt=system_prompt,
        answer=answer,
        review_flags=sorted(review_flags),
        logged_acc=row.get("acc"),
        logged_score=row.get("score"),
    )


def load_source_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    columns = ["images", "question", "solution", "source_index", "source_image"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[str(row["question"]).strip()].append(row)
    return by_question


def relative_crop_rect(bbox: Any, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("crop response has no relative bbox_2d")
    width, height = image_size
    left, top, right, bottom = [float(value) for value in bbox]
    rect = (
        max(0, min(width - 1, round(left / 1000 * width))),
        max(0, min(height - 1, round(top / 1000 * height))),
        max(1, min(width, round(right / 1000 * width))),
        max(1, min(height, round(bottom / 1000 * height))),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError(f"invalid reconstructed crop rectangle: {rect}")
    return rect


def jpeg_roundtrip(image: Image.Image) -> tuple[bytes, Image.Image]:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    payload = buffer.getvalue()
    return payload, Image.open(io.BytesIO(payload)).convert("RGB")


def rewrite_crop_paths(value: Any, uri: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"path", "crop_path"} and isinstance(item, str) and item.startswith("tool://"):
                value[key] = uri
            else:
                rewrite_crop_paths(item, uri)
    elif isinstance(value, list):
        for item in value:
            rewrite_crop_paths(item, uri)


def convert_candidate(candidate: Candidate, crop_dir: Path) -> dict[str, Any]:
    source_images = [str(path) for path in candidate.source.get("images") or []]
    if len(source_images) != 1 or not Path(source_images[0]).is_file():
        raise ValueError("source image is missing or not singular")

    uid = f"rl_kl_safe_{int(candidate.source['source_index'])}_step{candidate.step}"
    images = [source_images[0]]
    image_objects: list[Image.Image | None] = [None]
    messages = [
        {"role": "system", "content": candidate.system_prompt},
        {"role": "user", "content": f"<image>\n{candidate.question}"},
    ]
    pending_crops: list[tuple[Path, bytes]] = []

    for pair_index, (call, response) in enumerate(zip(candidate.calls, candidate.responses), start=1):
        messages.append({"role": "assistant", "content":
                         f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"})
        response = json.loads(json.dumps(response))
        response_suffix = ""
        if call["name"] == "crop_zoom":
            arguments = call["arguments"]
            source_index = int(arguments.get("target_image", 0))
            if image_objects[source_index] is None:
                with Image.open(images[source_index]) as original:
                    image_objects[source_index] = original.convert("RGB")
            crop_info = response["crop_zoom"]
            target_image = int(crop_info["target_image"])
            crop = image_objects[source_index].crop(
                relative_crop_rect(crop_info.get("bbox_2d"), image_objects[source_index].size)
            )
            payload, decoded_crop = jpeg_roundtrip(crop)
            reported_width = int((crop_info.get("image_outputs") or [{}])[0].get("width", crop.width))
            reported_height = int((crop_info.get("image_outputs") or [{}])[0].get("height", crop.height))
            if crop.size != (reported_width, reported_height):
                raise ValueError(
                    f"reconstructed crop size {crop.size} differs from rollout {(reported_width, reported_height)}"
                )
            crop_path = crop_dir / f"{uid}_image{target_image}.jpg"
            uri = f"tool://images/{target_image}/{crop_path.name}"
            rewrite_crop_paths(response, uri)
            pending_crops.append((crop_path, payload))
            images.append(str(crop_path))
            image_objects.append(decoded_crop)
            response_suffix = "\ncrop_zoom returned a crop: <image>"

        response_text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        messages.append(
            {
                "role": "user",
                "content": f"<tool_response>\n{response_text}\n</tool_response>{response_suffix}",
            }
        )

    messages.append({"role": "assistant", "content": f"<answer>{candidate.answer}</answer>"})
    if len(images) != sum(message["content"].count("<image>") for message in messages):
        raise ValueError("image placeholder count does not match image paths")

    for crop_path, payload in pending_crops:
        crop_path.write_bytes(payload)
    return {"messages": messages, "images": images, "metadata": {
        "source_index": int(candidate.source["source_index"]),
        "source_image": candidate.source.get("source_image"),
        "rollout_step": candidate.step,
        "rollout_file": candidate.rollout_file,
        "rollout_line": candidate.rollout_line,
        "review_flags": candidate.review_flags,
        "original_final": candidate.turns[-1]["content"],
        "logged_acc": candidate.logged_acc,
        "logged_score": candidate.logged_score,
        "answer_assessment": answer_assessment(
            reward.extract_answer(candidate.turns[-1]["content"]), candidate.source["solution"]
        ),
        "final_explanation_removed": candidate.turns[-1]["content"] != f"<answer>{candidate.answer}</answer>",
        "system_prompt_sha256": hashlib.sha256(candidate.system_prompt.encode()).hexdigest(),
        "quality_status": "automatically_filtered_requires_visual_review",
    }}


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rollout_dir.is_dir():
        raise FileNotFoundError(f"rollout directory does not exist: {args.rollout_dir}")
    if not args.train_parquet.is_file():
        raise FileNotFoundError(f"train parquet does not exist: {args.train_parquet}")
    if not args.audit_only and args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    crop_dir = (args.crop_dir or args.output.with_name(f"{args.output.stem}_crops")).absolute()
    if not args.audit_only and crop_dir.exists():
        raise FileExistsError(f"crop directory already exists: {crop_dir}")
    review_path = getattr(args, "semantic_review_jsonl", None)
    sample_path = getattr(args, "review_sample_jsonl", None)
    artifact_paths = [path.absolute() for path in (args.output, review_path, sample_path,
                     getattr(args, "report_json", None)) if path is not None]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("output artifact paths must be distinct")
    for path in (review_path, sample_path):
        if path is not None and path.exists():
            raise FileExistsError(f"review output already exists: {path}")
    files = sorted(args.rollout_dir.glob(args.rollout_glob), key=lambda path: int(path.stem))
    files = [path for path in files if args.min_step <= int(path.stem) <= args.max_step]
    if not files:
        raise FileNotFoundError(f"no rollout files match {args.rollout_dir / args.rollout_glob}")

    source_rows = load_source_rows(args.train_parquet)
    stats: Counter[str] = Counter()
    per_step: dict[int, Counter[str]] = defaultdict(Counter)
    rejected_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    best: dict[int, Candidate] = {}
    reviews: dict[tuple[int, str], dict[str, Any]] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                stats["rollouts"] += 1
                step_stats = per_step[int(path.stem)]
                step_stats["rollouts"] += 1
                try:
                    row = json.loads(line)
                    _, source = resolve_source(row, source_rows)
                    prediction = reward.extract_answer(str(row.get("output") or ""))
                    assessment = answer_assessment(prediction, source["solution"])
                    label_match = assessment["label_status"] == "label_match"
                    for counter in (stats, step_stats):
                        counter["uniquely_mapped"] += 1
                        counter["logged_correct"] += int(row.get("acc", 0) == 1)
                        counter[assessment["label_status"]] += 1
                        counter["logged_correct_without_label_match"] += int(row.get("acc", 0) == 1 and not label_match)
                    if assessment["label_status"] in {"unmapped_answer", "different_label"}:
                        key = (int(source["source_index"]), normalize_answer(prediction))
                        previous = reviews.get(key)
                        output = str(row.get("output") or "")
                        rank = (row.get("acc", 0) != 1, len(output), int(path.stem))
                        occurrences = (previous["trajectory_count"] if previous else 0) + 1
                        if previous is None or rank < previous["selection_rank"]:
                            reviews[key] = {
                                **assessment, "source_index": int(source["source_index"]),
                                "images": source["images"], "question": source["question"],
                                "rollout_file": str(path.absolute()), "rollout_line": line_number,
                                "rollout_step": int(path.stem), "raw_output": output,
                                "logged_acc": row.get("acc"), "logged_score": row.get("score"),
                                "selection_rank": rank,
                                "review_instruction": "Inspect image, question and raw trajectory; label mismatch alone is not a semantic error.",
                            }
                        reviews[key]["trajectory_count"] = occurrences
                    candidate = parse_candidate(row, source_rows, args.max_query_words)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    reason = str(exc)
                    stats[f"rejected/{reason}"] += 1
                    if len(rejected_examples[reason]) < 3:
                        rejected_examples[reason].append({"file": str(path), "line": line_number})
                    continue
                stats["eligible_rollouts"] += 1
                step_stats["eligible_rollouts"] += 1
                candidate.rollout_file = str(path.absolute())
                candidate.rollout_line = line_number
                source_index = int(candidate.source["source_index"])
                previous = best.get(source_index)
                if previous is None or candidate.rank < previous.rank:
                    best[source_index] = candidate
        print(f"Scanned step {path.stem}: {step_stats['rollouts']} trajectories, "
              f"{step_stats['eligible_rollouts']} eligible", file=sys.stderr, flush=True)

    selected = sorted(best.values(), key=lambda item: int(item.source["source_index"]))
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    conversion_errors = []
    output_digest = hashlib.sha256()
    written_labels: Counter[str] = Counter()
    written_flags: Counter[str] = Counter()
    if not args.audit_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True)
        with args.output.open("x", encoding="utf-8") as handle:
            for candidate in selected:
                try:
                    item = convert_candidate(candidate, crop_dir)
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    stats["conversion_errors"] += 1
                    if len(conversion_errors) < 20:
                        conversion_errors.append(
                            f"source_index={candidate.source.get('source_index')}: {exc}"
                        )
                    continue
                serialized = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(serialized)
                output_digest.update(serialized.encode("utf-8"))
                stats["written"] += 1
                stats["crop_images"] += len(item["images"]) - 1
                stats["written_direct_answers"] += int(not candidate.calls)
                written_labels[candidate.answer] += 1
                written_flags.update(candidate.review_flags)
                if stats["written"] % 500 == 0:
                    print(f"Exported {stats['written']}/{len(selected)} candidates", file=sys.stderr, flush=True)

    review_entries = []
    for key in sorted(reviews):
        entry = dict(reviews[key])
        entry.pop("selection_rank")
        review_entries.append(entry)
    review_sample = sample_semantic_reviews(review_entries, getattr(args, "review_sample_size", 200))
    for path, entries in ((review_path, review_entries), (sample_path, review_sample)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "rollout_files": len(files),
        "rollout_steps": [int(files[0].stem), int(files[-1].stem)],
        "unique_eligible_samples": len(best),
        "audit_only": args.audit_only,
        "completion_status": "complete",
        "answer_policy": "Canonical label agreement is checked; semantic correctness is unreviewed, including for on/in/over.",
        "relation_aliases": reward.RELATION_ALIASES,
        "semantic_review_unique_source_answers": len(review_entries),
        "semantic_review_jsonl": str(review_path) if review_path else None,
        "semantic_review_label_status_counts": dict(Counter(entry["label_status"] for entry in review_entries)),
        "review_sample_jsonl": str(sample_path) if sample_path else None,
        "review_sample_count": len(review_sample),
        "review_sample_policy": "Stratified by gold label, answer status, on/other and early/late; unique source IDs. Not a random accuracy estimate.",
        "output_sha256": output_digest.hexdigest() if not args.audit_only else None,
        "written_label_counts": dict(written_labels),
        "written_review_flags": dict(written_flags),
        "selected_label_counts": dict(Counter(candidate.answer for candidate in selected)),
        "selected_direct_answers": sum(not candidate.calls for candidate in selected),
        "selected_review_flags": dict(Counter(flag for candidate in selected for flag in candidate.review_flags)),
        "per_step": {str(step): dict(counts) for step, counts in sorted(per_step.items())},
        "rejected_examples": dict(rejected_examples),
        "quality_status": "automatically_filtered_requires_visual_review",
        "output": str(args.output),
        "crop_dir": str(crop_dir),
        "stats": dict(sorted(stats.items())),
        "conversion_error_examples": conversion_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--rollout-glob", default="[0-9]*.jsonl")
    parser.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--max-query-words", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=sys.maxsize)
    parser.add_argument("--audit-only", action="store_true", help="Rescore and filter without writing SFT or crop files.")
    parser.add_argument("--report-json", type=Path, help="Write a new audit report; existing files are never overwritten.")
    parser.add_argument("--semantic-review-jsonl", type=Path, help="Preserve label disagreements for visual semantic review.")
    parser.add_argument("--review-sample-jsonl", type=Path, help="Write a stratified subset of the semantic review queue.")
    parser.add_argument("--review-sample-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_json is not None and args.report_json.exists():
        raise FileExistsError(f"report already exists: {args.report_json}")
    summary = convert(args)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
