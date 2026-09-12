#!/usr/bin/env python3
"""
Prepare perceive2reason training data from TRAINING set rollout results.

This version works with the actual rollout format:
- Rollout files contain: input, output, score, rollout_trace
- Tool responses are embedded in the output text, not in rollout_trace.tool_calls
- Ground truth is determined by score field (0.0 = wrong, 1.0 = correct)
- Question is in the input field

Usage:
  python prepare_perceive2reason_data_from_train_v2.py \
    --rollout-dir /path/to/rollouts \
    --output data/perceive2reason/p2r_train.jsonl
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_perception_prefix_from_output(output: str) -> list[dict]:
    """
    Extract tool call/response pairs from rollout output.

    Returns list of message dicts (role: assistant/tool, content: <tool_call>...</tool_call>)
    """
    messages = []

    # Find all tool_call and tool_response tags in order
    pattern = r'(<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>)'
    matches = re.findall(pattern, output, re.DOTALL)

    for match in matches:
        if '<tool_call>' in match:
            messages.append({
                "role": "assistant",
                "content": match
            })
        elif '<tool_response>' in match:
            messages.append({
                "role": "tool",
                "content": match
            })

    return messages


def has_valid_detection(output: str) -> bool:
    """
    Check if output contains successful grounding_detect with valid boxes.
    """
    tool_responses = re.findall(r'<tool_response>(.*?)</tool_response>', output, re.DOTALL)

    for response_json in tool_responses:
        try:
            response_data = json.loads(response_json)
            # Check for grounding_detect with boxes
            if response_data.get("boxes") and len(response_data["boxes"]) > 0:
                confidences = response_data.get("confidence", [])
                if confidences and max(confidences) > 0.3:
                    return True
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return False


def extract_question_from_input(input_text: str) -> str:
    """
    Extract the user question from the input field.
    Input format: system\n<system_prompt>\nuser\n<question>\nassistant\n
    """
    match = re.search(r'user\n(.*?)\nassistant\n', input_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_prediction_from_output(output: str) -> str:
    """
    Extract the final answer from output.
    """
    match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def process_rollout_files(rollout_dir: Path) -> dict:
    """
    Process all rollout JSONL files and extract P2R candidates.

    P2R criteria:
    - score == 0.0 (wrong answer)
    - has_valid_detection (tool calls succeeded)
    """
    stats = Counter()
    qualified_samples = []

    jsonl_files = list(rollout_dir.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} rollout files")

    for jsonl_file in sorted(jsonl_files):
        logger.info(f"Processing {jsonl_file}")

        with open(jsonl_file) as f:
            for line in f:
                sample = json.loads(line)

                # Extract fields
                score = sample.get("score", 0.0)
                output = sample.get("output", "")
                input_text = sample.get("input", "")

                stats["total"] += 1

                # Check if correct (score == 1.0)
                if score == 1.0:
                    stats["correct"] += 1
                    continue

                # Wrong answer (score == 0.0)
                stats["wrong"] += 1

                # Check for valid detection
                if not has_valid_detection(output):
                    stats["no_valid_detection"] += 1
                    continue

                # Extract perception prefix
                perception_prefix = parse_perception_prefix_from_output(output)

                if not perception_prefix:
                    stats["no_tools"] += 1
                    continue

                # Qualified for P2R training!
                stats["qualified"] += 1

                question = extract_question_from_input(input_text)
                prediction = extract_prediction_from_output(output)

                qualified_samples.append({
                    "question": question,
                    "wrong_prediction": prediction,
                    "perception_prefix": perception_prefix,
                    "num_tool_calls": len([m for m in perception_prefix if m["role"] == "assistant"]),
                })

    return {
        "samples": qualified_samples,
        "stats": dict(stats)
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare P2R data from training rollouts")
    parser.add_argument(
        "--rollout-dir",
        type=str,
        required=True,
        help="Directory containing rollout JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process rollouts
    result = process_rollout_files(rollout_dir)
    samples = result["samples"]
    stats = result["stats"]

    # Write output
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {len(samples)} samples to {output_path}")
    logger.info(f"\nStatistics:")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")

    # Write summary
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, "w") as f:
        json.dump({
            "num_samples": len(samples),
            "stats": stats,
            "output_file": str(output_path)
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSummary written to {summary_path}")

    # Print sample distribution
    if samples:
        tool_call_dist = Counter(s["num_tool_calls"] for s in samples)
        logger.info(f"\nTool call distribution:")
        for num_calls, count in sorted(tool_call_dist.items()):
            logger.info(f"  {num_calls} calls: {count} samples")


if __name__ == "__main__":
    main()
