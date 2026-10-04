#!/usr/bin/env python3
"""Summarize first-decision frequencies per training step from rollout dumps.

Reads ``<step>.jsonl`` files written by the trainer's rollout dump. The first
decision is the tool executed during the first model turn; "direct" means no tool
was executed then (a direct answer or an unparsable call). Forced branches are
excluded, so the frequencies describe the sampled policy.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def first_decision(trace: dict) -> str:
    first_turn = [call["tool"] for call in trace.get("tool_calls", []) if call.get("model_turn") == 1]
    return first_turn[0] if first_turn else "direct"


def summarize(rollout_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
        counts, scores = Counter(), defaultdict(list)
        by_source = defaultdict(Counter)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                trace = sample.get("rollout_trace")
                if trace is None:
                    raise ValueError(f"{path} has no rollout_trace; enable trainer.rollout_data_dir tracing")
                if trace.get("forced_decision") is not None:
                    continue
                decision = first_decision(trace)
                counts[decision] += 1
                scores[decision].append(float(sample["score"]))
                source = (sample.get("source_metadata") or {}).get("data_source", "unknown")
                by_source[source][decision] += 1
        total = sum(counts.values())
        rows.append({
            "step": int(path.stem),
            "samples": total,
            "rate": {decision: count / total for decision, count in sorted(counts.items())},
            "score": {decision: sum(values) / len(values) for decision, values in sorted(scores.items())},
            "rate_by_source": {
                source: {decision: count / sum(counter.values()) for decision, count in sorted(counter.items())}
                for source, counter in sorted(by_source.items())
            },
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    args = parser.parse_args()
    for row in summarize(args.rollout_dir):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
