#!/usr/bin/env python3
"""Audit how a visual agent relies on tool outputs.

Training rollouts (step JSONL files): per data source and tool, report call
rates, natural tool error rates against the ground truth, how often the final
answer copies the tool value, how often a wrong tool value is corrected, and
the same-question accuracy gap between calling a tool first and answering
directly. Evaluation predictions (VLMEval XLSX/JSON with raw_response): report
how often injected faults are copied. Optionally write a fault profile whose
per-source probabilities follow the measured tool error rates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
COUNT_TOOL, OCR_TOOL = "object_count", "ocr_read"
# Sources whose answer is the tool value itself: a count, or text a correct OCR must contain.
# Elsewhere a tool value may be an intermediate step (three bars for a sum of 30), not an error.
ANSWER_SOURCES = {
    COUNT_TOOL: {"visual-agent-tallyqa", "visual-agent-fsc147"},
    OCR_TOOL: {"visual-agent-ocr"},
}


def final_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def as_int(value) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def squash(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def observation(call: dict) -> dict:
    """The model-visible observation of a training tool call."""
    try:
        value = json.loads(call.get("model_observation") or "")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def tool_value(tool: str, observed: dict):
    if tool == COUNT_TOOL:
        return as_int(observed.get("count"))
    if tool == OCR_TOOL and isinstance(observed.get("text"), str):
        return observed["text"]
    return None


def tool_is_wrong(tool: str, value, ground_truth) -> bool | None:
    """Whether a tool value contradicts the ground truth; None when not comparable."""
    if value is None or ground_truth is None:
        return None
    if tool == COUNT_TOOL:
        truth = as_int(ground_truth)
        return None if truth is None else value != truth
    if tool == OCR_TOOL:
        return squash(ground_truth).lower() not in squash(value).lower()
    return None


def copies(tool: str, value, answer: str) -> bool:
    if value is None or not answer:
        return False
    if tool == COUNT_TOOL:
        return as_int(answer) == value
    return squash(answer).lower() in squash(value).lower()


def first_action(trace: dict) -> str:
    for call in trace.get("tool_calls") or []:
        if call.get("model_turn") == 1:
            return call.get("tool") or "unknown"
    return "answer"


def audit_rollouts(paths: list[Path]) -> dict:
    stats = defaultdict(lambda: defaultdict(float))
    by_question = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                meta = row.get("source_metadata") or {}
                source = meta.get("data_source", "unknown")
                if meta.get("reliance_branch"):
                    source = f"{source}#{meta['reliance_branch']}"
                truth = meta.get("ground_truth")
                correct = float(row.get("acc", 0.0)) > 0
                answer = final_answer(row.get("output", ""))
                trace = row.get("rollout_trace") or {}
                stats[source]["trajectories"] += 1
                stats[source]["acc"] += correct
                action = first_action(trace)
                question = (meta.get("source_index"), meta.get("question"))
                by_question[source][question].append((action, correct))
                seen = set()
                for call in trace.get("tool_calls") or []:
                    tool = call.get("tool")
                    if tool in seen:
                        continue
                    seen.add(tool)
                    stats[source][f"{tool}/called"] += 1
                    if call.get("status") != "success":
                        stats[source][f"{tool}/errors"] += 1
                        continue
                    observed = observation(call)
                    if tool == "grounding_detect" and observed.get("boxes") == []:
                        stats[source][f"{tool}/empty"] += 1
                    value = tool_value(tool, observed)
                    copied = copies(tool, value, answer)
                    stats[source][f"{tool}/copied"] += copied
                    comparable = source.split("#")[0] in ANSWER_SOURCES.get(tool, ())
                    wrong = tool_is_wrong(tool, value, truth) if comparable else None
                    if wrong is None:
                        continue
                    stats[source][f"{tool}/checked"] += 1
                    if wrong:
                        stats[source][f"{tool}/wrong"] += 1
                        stats[source][f"{tool}/wrong_copied"] += copied
                        stats[source][f"{tool}/wrong_corrected"] += correct
    return summarize(stats, by_question)


def summarize(stats, by_question) -> dict:
    report = {}
    for source, counts in sorted(stats.items()):
        total = counts["trajectories"]
        entry = {"trajectories": int(total), "acc": counts["acc"] / total, "tools": {}}
        tools = sorted({key.split("/")[0] for key in counts if "/" in key})
        for tool in tools:
            called = counts[f"{tool}/called"]
            checked, wrong = counts[f"{tool}/checked"], counts[f"{tool}/wrong"]
            entry["tools"][tool] = {
                "call_rate": called / total,
                "error_status_rate": counts[f"{tool}/errors"] / called if called else None,
                "empty_rate": counts[f"{tool}/empty"] / called if tool == "grounding_detect" and called else None,
                "copy_rate": counts[f"{tool}/copied"] / called if called else None,
                "checked": int(checked),
                "tool_error_rate": wrong / checked if checked else None,
                "wrong_copy_rate": counts[f"{tool}/wrong_copied"] / wrong if wrong else None,
                "wrong_correction_rate": counts[f"{tool}/wrong_corrected"] / wrong if wrong else None,
            }
        entry["first_action_effect"] = first_action_effect(by_question[source])
        report[source] = entry
    return report


def first_action_effect(questions: dict) -> dict:
    """Mean same-question accuracy of calling a tool first minus answering directly."""
    gaps, shares = defaultdict(list), defaultdict(float)
    total = 0
    for outcomes in questions.values():
        by_action = defaultdict(list)
        for action, correct in outcomes:
            by_action[action].append(correct)
            shares[action] += 1
            total += 1
        direct = by_action.get("answer")
        for action, values in by_action.items():
            if action != "answer" and direct:
                gaps[action].append(sum(values) / len(values) - sum(direct) / len(direct))
    return {
        action: {
            "share": shares[action] / total,
            "paired_questions": len(gaps[action]),
            "acc_gap_vs_answer": sum(gaps[action]) / len(gaps[action]) if gaps[action] else None,
        }
        for action in sorted(shares)
    }


def load_predictions(path: Path) -> list[dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else list(rows.values())
    import pandas as pd

    return pd.read_excel(path).to_dict("records")


def audit_faults(paths: list[Path]) -> dict:
    report = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for path in paths:
        for row in load_predictions(path):
            try:
                trace = json.loads(row.get("raw_response") or "")
            except (TypeError, json.JSONDecodeError):
                report[path.name]["_"]["unreadable"] += 1
                continue
            answer = str(row.get("prediction", ""))
            for call in trace.get("tool_calls") or []:
                fault = call.get("fault")
                if not fault:
                    continue
                tool = call.get("name")
                counts = report[path.name][tool]
                counts["faulted"] += 1
                injected = tool_value(tool, fault.get("injected") or {})
                original = tool_value(tool, fault.get("original") or {})
                if injected is not None and copies(tool, injected, answer) and not copies(tool, original, answer):
                    counts["copied_fault"] += 1
                if original is not None and copies(tool, original, answer):
                    counts["kept_original"] += 1
    return {name: {tool: dict(counts) for tool, counts in tools.items()} for name, tools in report.items()}


def matched_profile(report: dict, target_rate: float) -> dict:
    """Per-source fault probabilities proportional to measured tool error rates, averaging target_rate."""
    rates = {
        (source, tool): values["tool_error_rate"]
        for source, entry in report.items()
        for tool, values in entry["tools"].items()
        if values["tool_error_rate"] is not None
    }
    if not rates:
        raise ValueError("No tool error rate could be measured against the ground truth")
    mean = sum(rates.values()) / len(rates)
    by_source = defaultdict(dict)
    for (source, tool), rate in rates.items():
        by_source[source][tool] = min(1.0, target_rate * rate / mean) if mean else target_rate
    return {"default": target_rate, "by_source": dict(by_source)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, action="append", default=[])
    parser.add_argument("--steps", help="Inclusive step range such as 1-11")
    parser.add_argument("--predictions", type=Path, action="append", default=[], help="VLMEval XLSX/JSON with raw_response")
    parser.add_argument("--profile-out", type=Path, help="Write a matched fault profile from the rollout audit")
    parser.add_argument("--target-rate", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {}
    if args.rollout_dir:
        low, high = (int(value) for value in args.steps.split("-")) if args.steps else (None, None)
        paths = sorted(
            (path for directory in args.rollout_dir for path in directory.glob("*.jsonl") if path.stem.isdigit()
             and (low is None or low <= int(path.stem) <= high)),
            key=lambda path: int(path.stem),
        )
        result["rollouts"] = {"files": [str(path) for path in paths], "sources": audit_rollouts(paths)}
        if args.profile_out:
            args.profile_out.write_text(
                json.dumps(matched_profile(result["rollouts"]["sources"], args.target_rate), indent=2) + "\n",
                encoding="utf-8",
            )
    if args.predictions:
        result["faults"] = audit_faults(args.predictions)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
