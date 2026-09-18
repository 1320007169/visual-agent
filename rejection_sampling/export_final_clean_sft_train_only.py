#!/usr/bin/env python3
"""Export the final rejection-sampled dataset without audit metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT / "data/rl_distill/qwen3_rl_rejection_sampled_clean_10953_sft.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/rl_distill/qwen3_rl_rejection_sampled_clean_10953_sft_train_only.jsonl"
)
EXPECTED_ROWS = 10_953


def normalize_row(row: Any, line_number: int) -> dict[str, list[Any]]:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: row is not an object")

    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"line {line_number}: invalid messages")
    if not isinstance(images, list) or not images:
        raise ValueError(f"line {line_number}: invalid images")

    normalized_messages = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"line {line_number}, message {message_index}: message is not an object"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"line {line_number}, message {message_index}: role/content must be strings"
            )
        normalized_messages.append({"role": role, "content": content})

    if any(not isinstance(image, str) or not image for image in images):
        raise ValueError(f"line {line_number}: image paths must be non-empty strings")
    placeholder_count = sum(
        message["content"].count("<image>") for message in normalized_messages
    )
    if placeholder_count != len(images):
        raise ValueError(
            f"line {line_number}: image placeholders {placeholder_count} != images {len(images)}"
        )

    return {"messages": normalized_messages, "images": images}


def export(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    row_count = 0
    try:
        with source.open("r", encoding="utf-8") as source_handle, temporary.open(
            "x", encoding="utf-8"
        ) as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    raise ValueError(f"line {line_number}: blank line")
                row = normalize_row(json.loads(line), line_number)
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1

        if row_count != EXPECTED_ROWS:
            raise ValueError(f"expected {EXPECTED_ROWS} rows, got {row_count}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "source": str(source),
        "output": str(output),
        "rows": row_count,
        "top_level_fields": ["messages", "images"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
