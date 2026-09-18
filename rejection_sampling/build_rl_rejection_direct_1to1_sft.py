#!/usr/bin/env python3
"""Build a 1:1 mixture of clean RL tool trajectories and direct answers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from export_final_clean_sft_train_only import normalize_row


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
TOOL_SOURCE = (
    DATA_ROOT / "rl_distill/qwen3_rl_rejection_sampled_clean_10953_sft_train_only.jsonl"
)
DIRECT_POOL_SOURCE = (
    DATA_ROOT / "visual_agent_groundingdino_direct34644_tool22230_56874_v2.jsonl"
)
OUTPUT = (
    DATA_ROOT
    / "rl_distill/qwen3_rl_rejection_clean10953_direct10953_mixed21906_sft_train_only.jsonl"
)
REPORT = (
    DATA_ROOT
    / "rl_distill/qwen3_rl_rejection_clean10953_direct10953_mixed21906_report.json"
)
TOOL_ROWS = 10_953
DIRECT_POOL_ROWS = 34_644
DIRECT_SAMPLE_ROWS = 10_953
SAMPLE_SEED = 20260918
SHUFFLE_SEED = 20260919


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            rows.append(json.loads(line))
    return rows


def has_tool_call(row: dict[str, Any]) -> bool:
    return any(
        message.get("role") == "assistant"
        and "<tool_call>" in str(message.get("content") or "")
        for message in row.get("messages") or []
    )


def question_text(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        content = message["content"]
        if message["role"] == "user" and "<tool_response>" not in content:
            return content.replace("<image>", "").strip()
    raise ValueError("row has no user question")


def identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(Path(row["images"][0]).resolve()), question_text(row).casefold()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    for source in (TOOL_SOURCE, DIRECT_POOL_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("output or report already exists")

    raw_tool_rows = read_jsonl(TOOL_SOURCE)
    if len(raw_tool_rows) != TOOL_ROWS:
        raise ValueError(f"expected {TOOL_ROWS} tool rows, got {len(raw_tool_rows)}")
    tool_rows = [normalize_row(row, index) for index, row in enumerate(raw_tool_rows, 1)]
    if not all(has_tool_call(row) for row in tool_rows):
        raise ValueError("tool source contains a direct-answer row")

    direct_pool = []
    for line_number, row in enumerate(read_jsonl(DIRECT_POOL_SOURCE), start=1):
        if not has_tool_call(row):
            direct_pool.append((line_number, normalize_row(row, line_number)))
    if len(direct_pool) != DIRECT_POOL_ROWS:
        raise ValueError(
            f"expected {DIRECT_POOL_ROWS} direct rows, got {len(direct_pool)}"
        )

    selected = random.Random(SAMPLE_SEED).sample(direct_pool, DIRECT_SAMPLE_ROWS)
    selected_lines = sorted(line_number for line_number, _ in selected)
    direct_rows = [row for _, row in selected]

    tool_identities = {identity(row) for row in tool_rows}
    direct_identities = {identity(row) for row in direct_rows}
    if len(tool_identities) != len(tool_rows):
        raise ValueError("tool source contains duplicate image/question pairs")
    if len(direct_identities) != len(direct_rows):
        raise ValueError("selected direct rows contain duplicate image/question pairs")
    overlap = tool_identities & direct_identities
    if overlap:
        raise ValueError(f"direct/tool overlap detected: {len(overlap)} rows")

    rows = tool_rows + direct_rows
    random.Random(SHUFFLE_SEED).shuffle(rows)
    write_jsonl_atomic(OUTPUT, rows)

    role_shapes = Counter(tuple(message["role"] for message in row["messages"]) for row in rows)
    report = {
        "status": "passed",
        "output": str(OUTPUT.resolve()),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "sources": {
            "tool_trajectories": str(TOOL_SOURCE.resolve()),
            "direct_answer_pool": str(DIRECT_POOL_SOURCE.resolve()),
        },
        "counts": {
            "total": len(rows),
            "tool_trajectories": len(tool_rows),
            "direct_answers": len(direct_rows),
            "direct_answer_pool": len(direct_pool),
            "direct_tool_overlap": len(overlap),
        },
        "ratio": {"tool_trajectories": 0.5, "direct_answers": 0.5},
        "sample_seed": SAMPLE_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "selected_direct_source_lines_sha256": hashlib.sha256(
            json.dumps(selected_lines).encode("utf-8")
        ).hexdigest(),
        "role_shapes": {"/".join(shape): count for shape, count in role_shapes.items()},
        "validation": {
            "top_level_fields": ["messages", "images"],
            "image_placeholders_match": True,
            "unique_image_question_pairs": len(tool_identities | direct_identities),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
