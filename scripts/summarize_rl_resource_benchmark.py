#!/usr/bin/env python3
"""Summarize measured RL steps from A/B/C ModelArts benchmark job directories."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def read_steps(path):
    steps = {}
    if not path.exists():
        return steps
    with path.open(errors="replace") as stream:
        for line in stream:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "timing_s/step:" not in line or "training/global_step:" not in line:
                continue
            metrics = {key: float(value) for key, value in re.findall(
                r"([A-Za-z0-9_./@-]+):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)}
            steps[int(metrics["training/global_step"])] = metrics
    return steps


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def summarize(root):
    config = json.loads((root / "benchmark.json").read_text())
    expected = list(range(config["warmup_steps"] + 1, config["total_steps"] + 1))
    all_steps = read_steps(root / "logs" / f"{config['run_name']}-node0.log")
    rows = [all_steps[step] for step in expected if step in all_steps]
    exits = [json.loads(path.read_text()) for path in root.glob("exit-*.json")]
    complete = len(rows) == len(expected) and len(exits) == 2 and all(e["exit_code"] == 0 for e in exits)
    result = {"scenario": config["scenario"], "status": "complete" if complete else "incomplete_or_failed",
              "measured_steps_found": len(rows), "measured_steps_expected": len(expected),
              "missing_steps": [step for step in expected if step not in all_steps], "config": config,
              "metrics": {}, "tool_calls": 0, "tool_errors": 0, "trace_rows": 0}
    for key in sorted(set().union(*(row.keys() for row in rows))):
        values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
        if values:
            result["metrics"][key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                                       "min": min(values), "max": max(values), "samples": len(values)}
    latencies = []
    missing_traces = []
    for step in expected:
        path = root / "rollouts" / f"{step}.jsonl"
        if not path.exists():
            missing_traces.append(step)
            continue
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                trace = json.loads(line).get("rollout_trace")
                if not isinstance(trace, dict):
                    continue
                result["trace_rows"] += 1
                for call in trace.get("tool_calls", []):
                    result["tool_calls"] += 1
                    result["tool_errors"] += int(call.get("status") != "success" or bool(call.get("release_error")))
                    if isinstance(call.get("latency_ms"), (int, float)):
                        latencies.append(call["latency_ms"])
    result["missing_trace_steps"] = missing_traces
    result["tool_error_rate"] = result["tool_errors"] / result["tool_calls"] if result["tool_calls"] else None
    result["tool_p95_ms"] = percentile(latencies, .95)
    result["trace_coverage_complete"] = not missing_traces and result["trace_rows"] == len(expected) * config["train_batch_size"] * config["rollout_n"]
    peaks = {}
    for path in (root / "logs").glob("*-gpu.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    key = f"node{row['node_rank']}/gpu{int(row['index'])}"
                    peaks[key] = max(peaks.get(key, 0), float(row["memory_used_mib"]) / 1024)
                except (ValueError, KeyError):
                    continue
    result["whole_job_gpu_peak_gib"] = peaks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path, help="Directory containing A/, B/, C/")
    args = parser.parse_args()
    results = [summarize(path.parent) for path in sorted(args.comparison_dir.glob("*/benchmark.json"))]
    if not results:
        parser.error("No benchmark manifests found")
    timing = ["step", "gen", "reward", "old_log_prob", "ref", "update_actor"]
    fields = ["scenario", "status", "measured_steps_found", "trace_coverage_complete",
              *[key + "_median_s" for key in timing], "tool_calls", "tool_errors", "tool_p95_ms"]
    (args.comparison_dir / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    with (args.comparison_dir / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            for key in timing:
                row[key + "_median_s"] = result["metrics"].get("timing_s/" + key, {}).get("median")
            writer.writerow(row)
            print(f"{row['scenario']}: {row['status']}, measured={row['measured_steps_found']}, "
                  f"step_median={row['step_median_s']}s, tool_errors={row['tool_errors']}/{row['tool_calls']}")
    print(f"Wrote {args.comparison_dir / 'summary.csv'} and summary.json")


if __name__ == "__main__":
    main()
