#!/usr/bin/env python3
"""Prepare newly successful tool trajectories for visual rejection sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


BASE = Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx")
REPO_ROOT = BASE / "visual-agent"
SCRIPT_DIR = Path(__file__).resolve().parent
MIXED_PARQUET = REPO_ROOT / "data/zwz_deepeyesv2_3k_nocount_hr4k_v1/train.parquet"
OLD_RELATION_PARQUET = REPO_ROOT / "data/zwz_rl_vqa/rl_original_relation/train.parquet"
OLD_ACCEPTED = (
    REPO_ROOT
    / "rejection_sampling/zwz_relation_qwen3_kl_safe_steps001_165_success10362"
    / "outputs/deepseek_v41/accepted.jsonl"
)
ROLLOUT_ROOT = BASE / "rollouts/visual-agent-zwz-rl"
ROLLOUTS = {
    "16gpu_n8_2node": ROLLOUT_ROOT / "zwz_deepeyesv2_3k_nocount_v1_n8_2node",
    "64gpu_b224_8node": ROLLOUT_ROOT / "zwz_deepeyesv2_3k_nocount_v1_n8_b224_8node_scratch",
}
DEFAULT_OUTPUT = SCRIPT_DIR / "data/new_unique_correct_tool_trajectories.jsonl"
DEFAULT_REPORT = SCRIPT_DIR / "data/prepare_report.json"

ROLE_RE = re.compile(r"\n(user|assistant)\n")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>\s*$", re.DOTALL | re.IGNORECASE)
SUPPORTED_TOOLS = {"grounding_detect", "crop_zoom"}
RELATION_LABELS = {
    "above", "below", "left of", "right of", "in front of", "behind",
    "inside", "contain", "overlap", "next to",
}


@dataclass
class Candidate:
    key: tuple[str, str, int]
    run_name: str
    rollout_file: Path
    rollout_line: int
    step: int
    row: dict[str, Any]
    source: dict[str, Any]
    system_prompt: str
    question: str
    answer: str
    turns: list[dict[str, str]]
    calls: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    flags: list[str]

    @property
    def rank(self) -> tuple[int, int, int, int, int]:
        ideal_calls = 2 if self.source["ability"] == "spatial_relation" else 1
        return (
            len(self.flags),
            abs(len(self.calls) - ideal_calls),
            len(self.calls),
            len(str(self.row.get("output") or "")),
            -self.step,
        )


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).strip(" .,;:!?\t\r\n")


def extract_question(input_text: str) -> str:
    if "\nuser\n" not in input_text:
        raise ValueError("missing_user_turn")
    question = input_text.rsplit("\nuser\n", 1)[1]
    question = re.sub(r"\nassistant\n?$", "", question)
    return question.strip()


def parse_turns(output: str) -> list[dict[str, str]]:
    parts = ROLE_RE.split(output.strip())
    turns = [{"role": "assistant", "content": parts[0].strip()}]
    for index in range(1, len(parts), 2):
        turns.append({"role": parts[index], "content": parts[index + 1].strip()})
    expected = ["assistant" if index % 2 == 0 else "user" for index in range(len(turns))]
    if [turn["role"] for turn in turns] != expected or turns[-1]["role"] != "assistant":
        raise ValueError("non_alternating_turns")
    return turns


def validate_box(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("invalid_bbox")
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in value):
        raise ValueError("invalid_bbox")
    if any(number < 0 or number > 1000 for number in value):
        raise ValueError("invalid_bbox")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("invalid_bbox")


def parse_candidate(
    row: dict[str, Any],
    run_name: str,
    rollout_file: Path,
    rollout_line: int,
    sources: dict[tuple[str, str, int], dict[str, Any]],
) -> Candidate:
    metadata = row.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("missing_source_metadata")
    data_source = metadata.get("data_source")
    source_index = metadata.get("source_index")
    if not isinstance(data_source, str) or type(source_index) is not int:
        raise ValueError("invalid_source_key")
    source_image = metadata.get("source_image")
    if not isinstance(source_image, str) or not source_image:
        raise ValueError("invalid_source_image")
    key = (data_source, source_image, source_index)
    source = sources.get(key)
    if source is None:
        raise ValueError("source_not_in_mixed_parquet")

    input_text = str(row.get("input") or "")
    question = extract_question(input_text)
    if normalize(question) != normalize(source["question"]):
        raise ValueError("question_mismatch")
    if normalize(metadata.get("ground_truth")) != normalize(source["solution"]):
        raise ValueError("ground_truth_mismatch")
    if not input_text.startswith("system\n"):
        raise ValueError("missing_system_prompt")
    system_prompt = input_text.split("\nuser\n", 1)[0].removeprefix("system\n").strip()

    output = str(row.get("output") or "")
    turns = parse_turns(output)
    answer_match = ANSWER_RE.search(turns[-1]["content"])
    if answer_match is None or len(re.findall(r"<answer>", output, re.IGNORECASE)) != 1:
        raise ValueError("missing_final_answer")
    answer = answer_match.group(1).strip()
    if source["ability"] == "spatial_relation":
        if normalize(answer) not in RELATION_LABELS or normalize(answer) != normalize(source["solution"]):
            raise ValueError("relation_answer_mismatch")

    calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    flags: set[str] = set()
    image_count = 1
    seen_calls: set[str] = set()
    for turn_index in range(0, len(turns) - 1, 2):
        call_matches = TOOL_CALL_RE.findall(turns[turn_index]["content"])
        response_matches = TOOL_RESPONSE_RE.findall(turns[turn_index + 1]["content"])
        if len(call_matches) != 1 or len(response_matches) != 1:
            raise ValueError("incomplete_tool_pair")
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
        if type(target_image) is not int or not 0 <= target_image < image_count:
            raise ValueError("invalid_target_image")
        if name == "grounding_detect":
            query = arguments.get("query")
            boxes = response.get("boxes")
            if not isinstance(query, str) or not query.strip() or not isinstance(boxes, list):
                raise ValueError("invalid_grounding")
            if normalize(response.get("query")) != normalize(query):
                raise ValueError("grounding_query_mismatch")
            for box in boxes:
                validate_box(box)
            if not boxes:
                flags.add("empty_detection")
            if len(boxes) > 1:
                flags.add("multiple_detection_boxes")
        else:
            validate_box(arguments.get("bbox_2d"))
            crop_info = response.get("crop_zoom")
            if not isinstance(crop_info, dict):
                raise ValueError("invalid_crop_response")
            if crop_info.get("target_image") != image_count:
                raise ValueError("invalid_crop_target")
            validate_box(crop_info.get("bbox_2d"))
            if crop_info["bbox_2d"] == [0, 0, 1000, 1000]:
                flags.add("full_image_crop")
            image_count += 1
        call_key = json.dumps(call, ensure_ascii=False, sort_keys=True)
        if call_key in seen_calls:
            flags.add("duplicate_tool_call")
        seen_calls.add(call_key)
        calls.append(call)
        responses.append(response)
    if not calls or len(turns) != 2 * len(calls) + 1:
        raise ValueError("no_complete_tool_trajectory")

    return Candidate(
        key=key,
        run_name=run_name,
        rollout_file=rollout_file,
        rollout_line=rollout_line,
        step=int(row.get("step") or int(rollout_file.stem)),
        row=row,
        source=source,
        system_prompt=system_prompt,
        question=question,
        answer=answer,
        turns=turns,
        calls=calls,
        responses=responses,
        flags=sorted(flags),
    )


def load_sources() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = pq.read_table(MIXED_PARQUET).to_pylist()
    sources: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["data_source"]),
            str(row["source_image"]),
            int(row["source_index"]),
        )
        if key in sources:
            raise ValueError(f"duplicate source key in mixed parquet: {key}")
        sources[key] = row
    return sources


def first_question(item: dict[str, Any]) -> str:
    for message in item.get("messages") or []:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and "<tool_response>" not in content:
            return content.replace("<image>", "").strip()
    raise ValueError("accepted item has no question")


def load_old_accepted_keys() -> set[tuple[str, str, int]]:
    rows = pq.read_table(
        OLD_RELATION_PARQUET,
        columns=["images", "question", "source_index", "source_image"],
    ).to_pylist()
    lookup: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        image = str(Path(str(row["images"][0])).resolve())
        lookup[(normalize(row["question"]), image)].append(
            (int(row["source_index"]), str(row["source_image"]))
        )

    accepted: set[tuple[str, str, int]] = set()
    failures = []
    with OLD_ACCEPTED.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            images = item.get("images") or []
            image = str(Path(str(images[0])).resolve()) if images else ""
            matches = lookup.get((normalize(first_question(item)), image), [])
            if len(matches) != 1:
                failures.append((line_number, len(matches)))
                continue
            source_index, source_image = matches[0]
            accepted.add(("visual-agent-zwz-relation", source_image, source_index))
    if failures:
        raise ValueError(f"could not map {len(failures)} old accepted rows; first={failures[:3]}")
    return accepted


def rollout_files(path: Path) -> list[Path]:
    files = [candidate for candidate in path.glob("*.jsonl") if candidate.stem.isdigit()]
    return sorted(files, key=lambda candidate: int(candidate.stem))


def convert_candidate(candidate: Candidate) -> dict[str, Any]:
    source_images = [str(path) for path in candidate.source.get("images") or []]
    if len(source_images) != 1 or not Path(source_images[0]).is_file():
        raise ValueError("missing_source_image")
    images = [source_images[0]]
    messages = [
        {"role": "system", "content": candidate.system_prompt},
        {"role": "user", "content": f"<image>\n{candidate.question}"},
    ]
    for pair_index, (call, response) in enumerate(zip(candidate.calls, candidate.responses), start=1):
        assistant_content = candidate.turns[(pair_index - 1) * 2]["content"]
        messages.append({"role": "assistant", "content": assistant_content})
        response_text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        messages.append({
            "role": "user",
            "content": f"<tool_response>\n{response_text}\n</tool_response>",
        })

    messages.append({"role": "assistant", "content": candidate.turns[-1]["content"]})
    placeholders = sum(str(message["content"]).count("<image>") for message in messages)
    if placeholders != len(images):
        raise ValueError(f"image_placeholder_mismatch:{placeholders}:{len(images)}")
    return {
        "messages": messages,
        "images": images,
        "metadata": {
            "data_source": candidate.source["data_source"],
            "ability": candidate.source["ability"],
            "source_index": int(candidate.source["source_index"]),
            "source_image": candidate.source.get("source_image"),
            "reference_answer": candidate.source["solution"],
            "candidate_answer": candidate.answer,
            "rollout_run": candidate.run_name,
            "rollout_step": candidate.step,
            "rollout_file": str(candidate.rollout_file),
            "rollout_line": candidate.rollout_line,
            "selection_flags": candidate.flags,
            "logged_acc": candidate.row.get("acc"),
            "logged_format": candidate.row.get("format"),
            "logged_score": candidate.row.get("score"),
            "deduplicated_against": str(OLD_ACCEPTED),
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    for required in (MIXED_PARQUET, OLD_RELATION_PARQUET, OLD_ACCEPTED):
        if not required.is_file():
            raise FileNotFoundError(required)
    for path in ROLLOUTS.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output or report already exists")

    sources = load_sources()
    old_accepted = load_old_accepted_keys()
    candidates: dict[tuple[str, str, int], list[Candidate]] = defaultdict(list)
    counts = Counter()
    errors = Counter()
    scanned_files: dict[str, dict[str, Any]] = {}

    for run_name, directory in ROLLOUTS.items():
        files = rollout_files(directory)
        if not files:
            raise ValueError(f"no rollout files in {directory}")
        scanned_files[run_name] = {
            "directory": str(directory),
            "file_count": len(files),
            "first_step": int(files[0].stem),
            "last_step": int(files[-1].stem),
            "files": [path.name for path in files],
        }
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    counts[f"{run_name}:raw"] += 1
                    row = json.loads(line)
                    if not (
                        float(row.get("acc") or 0) >= 1
                        and float(row.get("format") or 0) >= 1
                        and float(row.get("tool_used") or 0) >= 1
                        and "<tool_call>" in str(row.get("output") or "")
                    ):
                        continue
                    counts[f"{run_name}:base_success"] += 1
                    metadata = row.get("source_metadata") or {}
                    raw_key = (
                        metadata.get("data_source"),
                        metadata.get("source_image"),
                        metadata.get("source_index"),
                    )
                    if raw_key in old_accepted:
                        counts[f"{run_name}:old_accepted_overlap"] += 1
                        continue
                    try:
                        candidate = parse_candidate(row, run_name, path, line_number, sources)
                    except Exception as exc:
                        errors[f"{type(exc).__name__}:{exc}"] += 1
                        continue
                    candidates[candidate.key].append(candidate)
                    counts[f"{run_name}:valid_occurrences"] += 1

    selected = [min(group, key=lambda candidate: candidate.rank) for group in candidates.values()]
    selected.sort(key=lambda candidate: (candidate.key[0], candidate.key[1]))
    args.output.parent.mkdir(parents=True, exist_ok=False)

    output_counts = Counter()
    conversion_errors = []
    with args.output.open("x", encoding="utf-8") as handle:
        for candidate in selected:
            try:
                item = convert_candidate(candidate)
            except Exception as exc:
                conversion_errors.append({
                    "key": list(candidate.key),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_counts["total"] += 1
            output_counts[f"data_source:{candidate.source['data_source']}"] += 1
            output_counts[f"ability:{candidate.source['ability']}"] += 1
            output_counts[f"run:{candidate.run_name}"] += 1
            output_counts[f"tool_calls:{len(candidate.calls)}"] += 1

    report = {
        "output": str(args.output),
        "image_policy": "original_image_only; no reconstructed crop image is included or uploaded",
        "mixed_parquet": str(MIXED_PARQUET),
        "old_accepted": str(OLD_ACCEPTED),
        "old_accepted_keys": len(old_accepted),
        "scanned_rollouts": scanned_files,
        "scan_counts": dict(sorted(counts.items())),
        "parse_errors": dict(errors.most_common()),
        "unique_new_valid_keys_before_conversion": len(selected),
        "output_counts": dict(sorted(output_counts.items())),
        "conversion_errors": conversion_errors,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
