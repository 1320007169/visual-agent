#!/usr/bin/env python3
"""Merge the leakage-cleaned old batch with the newly accepted SFT batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx")
REPO_ROOT = BASE / "visual-agent"
REJECTION_ROOT = REPO_ROOT / "rejection_sampling"
OLD_ACCEPTED = (
    REJECTION_ROOT
    / "zwz_relation_qwen3_kl_safe_steps001_165_success10362"
    / "outputs/deepseek_v41/accepted.jsonl"
)
REJUDGE_ROOT = REJECTION_ROOT / "zwz_relation_old_accepted_suspicious834_rejudge"
SUSPICIOUS_INPUT = REJUDGE_ROOT / "data/suspicious834_original_only.jsonl"
REJUDGE_DECISIONS = REJUDGE_ROOT / "outputs/deepseek_v41/decisions.jsonl"
NEW_ACCEPTED_SFT = REPO_ROOT / "data/rl_distill/qwen3_mixed_rl_deepseek_accepted_2394_sft.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/rl_distill/qwen3_rl_rejection_sampled_clean_10953_sft.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data/rl_distill/qwen3_rl_rejection_sampled_clean_10953_report.json"
SHUFFLE_SEED = 20260917


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def load_rejudge_by_old_line() -> dict[int, dict[str, Any]]:
    suspicious = read_jsonl(SUSPICIOUS_INPUT)
    decisions = read_jsonl(REJUDGE_DECISIONS)
    if len(suspicious) != 834 or len(decisions) != 834:
        raise ValueError(
            f"incomplete rejudge inputs: suspicious={len(suspicious)}, decisions={len(decisions)}"
        )
    by_old_line: dict[int, dict[str, Any]] = {}
    for decision in decisions:
        sample_index = decision.get("sample_index")
        if type(sample_index) is not int or not 1 <= sample_index <= len(suspicious):
            raise ValueError(f"invalid rejudge sample_index: {sample_index}")
        metadata = suspicious[sample_index - 1].get("metadata") or {}
        old_line = metadata.get("old_accepted_line")
        if type(old_line) is not int or old_line < 1:
            raise ValueError(f"missing old accepted line for suspect {sample_index}")
        if old_line in by_old_line:
            raise ValueError(f"duplicate old accepted line: {old_line}")
        by_old_line[old_line] = decision
    return by_old_line


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


def add_old_provenance(
    row: dict[str, Any], line_number: int, decision: dict[str, Any] | None
) -> dict[str, Any]:
    result = json.loads(json.dumps(row))
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata.update({
        "final_merge_source": "old_kl_safe_deepseek_accepted_8755",
        "old_accepted_line": line_number,
    })
    if decision is None:
        metadata["quality_status"] = (
            "old_visual_judge_accepted_no_relation_phrase_in_tool_query"
        )
        metadata["answer_leakage_rejudge"] = "not_selected_by_lexical_suspicion_filter"
    else:
        metadata["quality_status"] = "old_visual_judge_and_answer_leakage_rejudge_accepted"
        metadata["answer_leakage_rejudge"] = compact_decision(decision)
    return result


def add_new_provenance(row: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(row))
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("new accepted row has no metadata")
    metadata["final_merge_source"] = "new_mixed_rl_deepseek_accepted_2394"
    return result


def question_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and "<tool_response>" not in content:
            return content.replace("<image>", "").strip()
    raise ValueError("row has no question")


def validate_row(row: dict[str, Any], index: int) -> tuple[str, str]:
    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"row {index}: invalid messages")
    if not isinstance(images, list) or not images:
        raise ValueError(f"row {index}: missing images")
    if messages[0].get("role") == "system":
        expected = ["system"] + [
            "user" if turn % 2 else "assistant" for turn in range(1, len(messages))
        ]
    else:
        expected = ["user" if turn % 2 == 0 else "assistant" for turn in range(len(messages))]
    roles = [message.get("role") for message in messages]
    if roles != expected or roles[-1] != "assistant":
        raise ValueError(f"row {index}: invalid role sequence {roles}")
    placeholders = sum(str(message.get("content") or "").count("<image>") for message in messages)
    if placeholders != len(images):
        raise ValueError(
            f"row {index}: image placeholder mismatch {placeholders} != {len(images)}"
        )
    for image in images:
        if not isinstance(image, str) or not Path(image).is_file():
            raise FileNotFoundError(f"row {index}: missing image {image}")
    original = str(Path(images[0]).resolve())
    question = question_text(row)
    return original, question


def merge(args: argparse.Namespace) -> dict[str, Any]:
    required = (OLD_ACCEPTED, SUSPICIOUS_INPUT, REJUDGE_DECISIONS, NEW_ACCEPTED_SFT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output or report already exists")

    rejudge = load_rejudge_by_old_line()
    verdicts = Counter(decision.get("verdict") for decision in rejudge.values())
    if verdicts != Counter({"accept": 638, "reject": 195, "review": 1}):
        raise ValueError(f"unexpected rejudge verdicts: {verdicts}")

    rows: list[dict[str, Any]] = []
    counts = Counter()
    old_rows = read_jsonl(OLD_ACCEPTED)
    if len(old_rows) != 8755:
        raise ValueError(f"expected 8755 old accepted rows, got {len(old_rows)}")
    for line_number, row in enumerate(old_rows, start=1):
        decision = rejudge.get(line_number)
        if decision is not None and decision.get("verdict") != "accept":
            counts[f"old_excluded:{decision.get('verdict')}"] += 1
            continue
        rows.append(add_old_provenance(row, line_number, decision))
        counts["old_kept"] += 1
        counts["old_kept_rejudge_accepted"] += int(decision is not None)
        counts["old_kept_not_suspicious"] += int(decision is None)

    new_rows = read_jsonl(NEW_ACCEPTED_SFT)
    if len(new_rows) != 2394:
        raise ValueError(f"expected 2394 new accepted rows, got {len(new_rows)}")
    for row in new_rows:
        rows.append(add_new_provenance(row))
        counts["new_kept"] += 1
        ability = (row.get("metadata") or {}).get("ability")
        counts[f"new_ability:{ability}"] += 1

    if len(rows) != 10953:
        raise ValueError(f"expected 10953 merged rows, got {len(rows)}")

    identities: set[tuple[str, str]] = set()
    image_paths: set[str] = set()
    image_references = 0
    for index, row in enumerate(rows, start=1):
        identity = validate_row(row, index)
        if identity in identities:
            raise ValueError(f"duplicate original image/question at row {index}: {identity}")
        identities.add(identity)
        paths = [str(Path(path).resolve()) for path in row["images"]]
        image_paths.update(paths)
        image_references += len(paths)

    random.Random(SHUFFLE_SEED).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "passed",
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "shuffle_seed": SHUFFLE_SEED,
        "inputs": {
            "old_accepted": str(OLD_ACCEPTED.resolve()),
            "old_suspicious_rejudge_decisions": str(REJUDGE_DECISIONS.resolve()),
            "new_accepted_sft": str(NEW_ACCEPTED_SFT.resolve()),
        },
        "counts": dict(sorted(counts.items())),
        "merged_rows": len(rows),
        "unique_original_image_question_pairs": len(identities),
        "image_references": image_references,
        "unique_image_paths": len(image_paths),
        "validation": {
            "all_images_exist": True,
            "image_placeholders_match": True,
            "message_roles_valid": True,
            "cross_batch_duplicates": 0,
            "review_rows_included": 0,
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(merge(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
