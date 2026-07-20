#!/usr/bin/env python3
"""Evaluate one visual-tool JSONL sample through the local online agent stack."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from visual_agent_inference import HTTPVisualToolExecutor, OpenAICompatibleModelClient, VisualAgent


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def load_sample(dataset: Path, uid: str | None, index: int) -> dict[str, Any]:
    with dataset.open(encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            sample = json.loads(line)
            if (uid is not None and sample.get("uid") == uid) or (uid is None and current_index == index):
                return sample
    selector = f"uid={uid}" if uid is not None else f"index={index}"
    raise ValueError(f"No sample matching {selector} in {dataset}")


def resolve_images(repo_root: Path, dataset: Path, sample: dict[str, Any]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in sample.get("images") or [sample.get("image")]:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        candidates = [
            path,
            dataset.parent / path,
            repo_root / path,
            repo_root / "Visual_Agent" / "training_trajectories_natural" / path,
        ]
        match = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"Could not resolve sample image: {raw_path}")
        resolved.append(match)
    if not resolved:
        raise ValueError("The selected sample has no images")
    return resolved


def extract_answer(response: str) -> str:
    match = ANSWER_RE.search(response)
    return (match.group(1) if match else response).strip()


def redact_images(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return f"<image data URL omitted: {len(value)} chars>"
    if isinstance(value, list):
        return [redact_images(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_images(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/visual_tool_sft_v0.smoke.jsonl"))
    parser.add_argument("--uid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="visual-agent")
    parser.add_argument("--tool-api-base", default="http://127.0.0.1:9000")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset = args.dataset if args.dataset.is_absolute() else repo_root / args.dataset
    sample = load_sample(dataset, args.uid, args.index)
    images = resolve_images(repo_root, dataset, sample)

    agent = VisualAgent(
        OpenAICompatibleModelClient(args.api_base, model=args.model),
        tool_executor=HTTPVisualToolExecutor(args.tool_api_base),
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    result = agent.run(images, str(sample["question"]))
    predicted = extract_answer(result.response)
    reference = str(sample.get("answer", "")).strip()
    required_tools = list(sample.get("required_tools") or [])
    called_tools = [call["name"] for call in result.tool_calls]
    answer_correct = predicted.casefold() == reference.casefold()
    tool_used = bool(called_tools)
    reference_tool_match = set(required_tools).issubset(called_tools)
    task_passed = answer_correct and (not required_tools or tool_used)
    report = {
        "uid": sample.get("uid"),
        "question": sample["question"],
        "images": [str(path) for path in images],
        "reference_answer": reference,
        "predicted_answer": predicted,
        "answer_correct": answer_correct,
        "required_tools": required_tools,
        "called_tools": called_tools,
        "tool_used": tool_used,
        "reference_tool_match": reference_tool_match,
        "turns": result.turns,
        "passed": task_passed,
        "strict_passed": answer_correct and reference_tool_match,
        "trace": redact_images(result.to_dict()),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
