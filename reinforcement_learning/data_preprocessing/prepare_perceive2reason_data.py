#!/usr/bin/env python3
"""
Prepare perceive2reason training data from step165 evaluation results.

Extracts samples where:
1. Tool calls succeeded (grounding_detect returned non-empty boxes)
2. Final answer was wrong
3. Tool trajectory is valid (no errors)

For each qualifying sample, creates a training instance with:
- Fixed perception prefix (tool call history)
- Target answer (GT)
- Metadata for analysis
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def has_valid_detection(tool_calls: list[dict]) -> tuple[bool, list[dict]]:
    """
    Check if tool calls contain at least one successful grounding_detect with non-empty boxes.

    Returns:
        (has_valid, prefix): True if valid detection exists, and the tool call history to preserve
    """
    if not tool_calls:
        return False, []

    perception_prefix = []
    has_detection = False

    for call in tool_calls:
        name = call.get("name", "")
        result = call.get("result", {})
        error = call.get("error", "")

        # Check for errors
        if error or (isinstance(result, dict) and result.get("status") in {"error", "failed"}):
            continue

        # Build message pair for this tool call
        arguments = call.get("arguments", {})

        # Assistant's tool call
        tool_call_msg = {
            "role": "assistant",
            "content": f"<tool_call>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}</tool_call>"
        }

        # Tool response
        if isinstance(result, dict):
            # Check if this is a successful detection
            if name == "grounding_detect":
                boxes = result.get("boxes", [])
                if boxes and len(boxes) > 0:
                    has_detection = True

            tool_response_msg = {
                "role": "tool",
                "content": f"<tool_response>{json.dumps(result, ensure_ascii=False)}</tool_response>"
            }
        else:
            tool_response_msg = {
                "role": "tool",
                "content": f"<tool_response>{str(result)}</tool_response>"
            }

        perception_prefix.append(tool_call_msg)
        perception_prefix.append(tool_response_msg)

    return has_detection, perception_prefix


def classify_error_type(row: dict, category: str) -> str:
    """Classify error into types for analysis."""
    question = row.get("question", "").lower()

    # Depth relation errors
    if any(word in question for word in ["in front of", "behind", "front", "back"]):
        return "depth_relation"

    # Counting errors
    if any(word in question for word in ["how many", "count", "number of"]):
        return "counting"

    # Cross-type tasks
    if category in {"cross_type", "cross-type"}:
        return "cross_type"

    # Attribute errors
    if category == "attribute":
        return "attribute"

    return "other"


def process_evaluation_results(
    results_dir: Path,
    output_path: Path,
    min_tool_calls: int = 1,
) -> dict[str, Any]:
    """
    Process evaluation JSONL files to extract perceive2reason training samples.

    Args:
        results_dir: Directory containing evaluation result JSONL files
        output_path: Path to write perceive2reason training data
        min_tool_calls: Minimum number of tool calls required

    Returns:
        Statistics dictionary
    """
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {results_dir}")

    stats = Counter()
    error_types = Counter()
    samples = []

    for path in files:
        benchmark = "VStarBench" if "VStarBench" in path.name else \
                   "HRBench4K" if "HRBench4K" in path.name else \
                   "HRBench8K" if "HRBench8K" in path.name else path.stem

        logger.info(f"Processing {benchmark}: {path}")

        for line_num, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue

            row = json.loads(line)
            stats["total"] += 1

            # Filter 1: Must be wrong
            correct = bool(row.get("final_judge") or row.get("hit"))
            if correct:
                stats["correct"] += 1
                continue

            stats["wrong"] += 1

            # Parse raw_response
            try:
                raw = json.loads(row.get("raw_response") or "{}")
            except (TypeError, json.JSONDecodeError):
                stats["parse_failed"] += 1
                continue

            tool_calls = raw.get("tool_calls", [])

            # Filter 2: Must have tool calls
            if len(tool_calls) < min_tool_calls:
                stats["no_tools"] += 1
                continue

            # Filter 3: Must have valid detection
            has_detection, perception_prefix = has_valid_detection(tool_calls)
            if not has_detection:
                stats["no_valid_detection"] += 1
                continue

            # Qualified sample
            stats["qualified"] += 1

            category = row.get("category", "uncategorized")
            error_type = classify_error_type(row, category)
            error_types[error_type] += 1

            # Get image paths
            image_paths = row.get("image_path_list") or row.get("image_path") or []
            if isinstance(image_paths, str):
                try:
                    import ast
                    parsed = ast.literal_eval(image_paths)
                    image_paths = parsed if isinstance(parsed, list) else [image_paths]
                except (ValueError, SyntaxError):
                    image_paths = [image_paths]

            image_path = str(image_paths[0]) if image_paths else ""

            # Build sample
            sample = {
                "uid": f"{benchmark}_{row.get('index', line_num)}",
                "benchmark": benchmark,
                "index": row.get("index"),
                "category": category,
                "error_type": error_type,
                "image": image_path,
                "question": row.get("question", ""),
                "gt_answer": row.get("answer", ""),
                "wrong_prediction": row.get("prediction", ""),
                "perception_prefix": perception_prefix,
                "num_tool_calls": len(tool_calls),
                "original_response": raw.get("response", "")[:500],  # Truncated for space
            }

            samples.append(sample)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {len(samples)} samples to {output_path}")
    logger.info(f"\nStatistics:")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")

    logger.info(f"\nError type distribution:")
    for error_type, count in error_types.most_common():
        logger.info(f"  {error_type}: {count} ({count/stats['qualified']*100:.1f}%)")

    return {
        "stats": dict(stats),
        "error_types": dict(error_types),
        "num_samples": len(samples),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare perceive2reason training data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/outputs/"
                     "vlmeval/qwen3_dino_kl_step80/qwen3_dino_kl_step80_3bench_20260807_133359/"
                     "dino_kl_step50/VisualAgent-vllm/T20260807_G"),
        help="Directory containing evaluation JSONL files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/visual-agent/"
                     "data/perceive2reason/step165_p2r_train.jsonl"),
        help="Output path for perceive2reason training data"
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Minimum number of tool calls required"
    )

    args = parser.parse_args()

    result = process_evaluation_results(
        results_dir=args.results_dir,
        output_path=args.output,
        min_tool_calls=args.min_tool_calls,
    )

    # Write summary
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
