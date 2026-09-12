#!/usr/bin/env python3
"""
Prepare perceive2reason training data from TRAINING set rollout results.

NOT from evaluation results! That would be training on test data.

Workflow:
1. Run inference on training data with current checkpoint
2. Extract samples where perception succeeded but reasoning failed
3. Save these samples with fixed perception prefixes for p2r training

Usage:
  # Step 1: Run inference on training data (generates rollout traces)
  python run_inference_on_train.py --checkpoint step165 --data zwz_train.jsonl

  # Step 2: Extract p2r samples from rollout traces
  python prepare_perceive2reason_data_from_train.py --rollout-dir outputs/train_rollout --output p2r_train.jsonl
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


def check_answer_correctness(prediction: str, gt_answer: str) -> bool:
    """
    Check if prediction matches ground truth.

    For now, simple string matching. Can be enhanced with fuzzy matching.
    """
    pred_clean = prediction.strip().lower()
    gt_clean = gt_answer.strip().lower()
    return pred_clean == gt_clean or gt_clean in pred_clean


def process_rollout_results(
    rollout_dir: Path,
    output_path: Path,
    min_tool_calls: int = 1,
) -> dict[str, Any]:
    """
    Process training data rollout results to extract perceive2reason samples.

    Args:
        rollout_dir: Directory containing rollout JSONL files from training data inference
        output_path: Path to write perceive2reason training data
        min_tool_calls: Minimum number of tool calls required

    Returns:
        Statistics dictionary
    """
    files = list(rollout_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {rollout_dir}")

    stats = Counter()
    samples = []

    for path in files:
        logger.info(f"Processing {path}")

        for line_num, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue

            row = json.loads(line)
            stats["total"] += 1

            # Must have GT answer
            gt_answer = row.get("answer") or row.get("gt_answer")
            if not gt_answer:
                stats["no_gt"] += 1
                continue

            # Must have prediction
            prediction = row.get("prediction") or row.get("model_answer")
            if not prediction:
                stats["no_prediction"] += 1
                continue

            # Check correctness
            correct = check_answer_correctness(prediction, gt_answer)
            if correct:
                stats["correct"] += 1
                continue

            stats["wrong"] += 1

            # Parse tool calls
            tool_calls = row.get("tool_calls", [])
            if len(tool_calls) < min_tool_calls:
                stats["no_tools"] += 1
                continue

            # Check for valid detection
            has_detection, perception_prefix = has_valid_detection(tool_calls)
            if not has_detection:
                stats["no_valid_detection"] += 1
                continue

            # Qualified sample
            stats["qualified"] += 1

            # Build p2r sample
            sample = {
                "uid": row.get("uid") or f"sample_{line_num}",
                "image": row.get("image", ""),
                "question": row.get("question", ""),
                "gt_answer": gt_answer,
                "wrong_prediction": prediction,
                "perception_prefix": perception_prefix,
                "num_tool_calls": len(tool_calls),
                "task_type": row.get("task_type", ""),
                "relation": row.get("relation", ""),
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

    return {
        "stats": dict(stats),
        "num_samples": len(samples),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare perceive2reason training data from training set rollout results"
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        required=True,
        help="Directory containing rollout JSONL files from training data inference"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for perceive2reason training data"
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Minimum number of tool calls required"
    )

    args = parser.parse_args()

    result = process_rollout_results(
        rollout_dir=args.rollout_dir,
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
