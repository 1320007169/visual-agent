"""Validation and paired analysis for frozen Visual Agent evaluations."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_LOG_FIELDS = (
    "run_id",
    "checkpoint",
    "split",
    "sample_id",
    "source_image_id",
    "mode",
    "seed",
    "correct",
    "tool_calls",
    "finish_reason",
    "generated_tokens",
    "visual_tokens",
    "elapsed_seconds",
)
VALID_MODES = {"native", "agent"}
VALID_CONDITIONS = {"on", "off"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            records.append(value)
    return records


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_prediction_records(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return all schema and policy violations without modifying experiment logs."""
    errors: list[str] = []
    for row_number, record in enumerate(records, start=1):
        prefix = f"row {row_number}"
        for field in REQUIRED_LOG_FIELDS:
            if field not in record:
                errors.append(f"{prefix}: missing required field {field!r}")
        if any(field not in record for field in REQUIRED_LOG_FIELDS):
            continue
        if record["mode"] not in VALID_MODES:
            errors.append(f"{prefix}: mode must be one of {sorted(VALID_MODES)}")
        if not isinstance(record["correct"], bool):
            errors.append(f"{prefix}: correct must be boolean")
        if not isinstance(record["tool_calls"], list):
            errors.append(f"{prefix}: tool_calls must be a list")
            continue
        if record["mode"] == "native" and record["tool_calls"]:
            errors.append(f"{prefix}: native mode must not execute tools")
        for call_number, tool_call in enumerate(record["tool_calls"], start=1):
            call_prefix = f"{prefix}: tool_calls[{call_number}]"
            if not isinstance(tool_call, dict):
                errors.append(f"{call_prefix} must be an object")
                continue
            for field in ("name", "status", "image_id", "bbox", "coordinate_system"):
                if field not in tool_call:
                    errors.append(f"{call_prefix}: missing {field!r}")
        for numeric_field in ("generated_tokens", "visual_tokens", "elapsed_seconds"):
            value = record[numeric_field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f"{prefix}: {numeric_field} must be a non-negative number")
    return errors


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile of no values")
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summarize_utility(
    records: Iterable[dict[str, Any]], *, bootstrap_repeats: int = 2_000, seed: int = 0
) -> dict[str, Any]:
    """Calculate paired ON/OFF utility and cluster bootstrap by sample id.

    Multiple decodes for one condition are averaged within a sample before
    calculating a difference. Bootstrap resampling is therefore by task, not by
    rollout, which avoids treating repeated generations as independent questions.
    """
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        condition = record.get("condition")
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"sample {record.get('sample_id')!r}: condition must be 'on' or 'off'")
        if not isinstance(record.get("correct"), bool):
            raise ValueError(f"sample {record.get('sample_id')!r}: correct must be boolean")
        if not isinstance(record.get("tool_calls"), list):
            raise ValueError(f"sample {record.get('sample_id')!r}: tool_calls must be a list")
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("utility records require sample_id")
        grouped[sample_id][condition].append(record)

    missing = sorted(sample_id for sample_id, values in grouped.items() if set(values) != VALID_CONDITIONS)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"every preselected sample needs both ON and OFF results; missing pairs: {preview}")

    deltas: list[float] = []
    on_scores: list[float] = []
    off_scores: list[float] = []
    on_records: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        conditions = grouped[sample_id]
        on_score = sum(row["correct"] for row in conditions["on"]) / len(conditions["on"])
        off_score = sum(row["correct"] for row in conditions["off"]) / len(conditions["off"])
        on_scores.append(on_score)
        off_scores.append(off_score)
        deltas.append(on_score - off_score)
        on_records.extend(conditions["on"])

    random_source = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_repeats):
        bootstrap.append(sum(random_source.choice(deltas) for _ in deltas) / len(deltas))
    bootstrap.sort()
    all_calls = [call for record in on_records for call in record["tool_calls"]]
    successful_calls = [call for call in all_calls if str(call.get("status", "")).lower() in {"ok", "success"}]
    used_tool = sum(bool(record["tool_calls"]) for record in on_records)

    return {
        "sample_count": len(deltas),
        "on_accuracy": sum(on_scores) / len(on_scores),
        "off_accuracy": sum(off_scores) / len(off_scores),
        "delta_tool": sum(deltas) / len(deltas),
        "delta_tool_ci95": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
        "on_decode_count": len(on_records),
        "tool_rate": used_tool / len(on_records),
        "average_calls": len(all_calls) / len(on_records),
        "effective_call_rate": (len(successful_calls) / len(all_calls)) if all_calls else None,
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
    }
