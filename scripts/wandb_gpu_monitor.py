#!/usr/bin/env python3
"""Publish per-GPU node metrics to a grouped Weights & Biases run."""

from __future__ import annotations

import argparse
import csv
import io
import signal
import subprocess
import threading
from pathlib import Path


GPU_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,power.limit,temperature.gpu"
)


def read_gpu_metrics() -> tuple[list[int], dict[str, float]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics: dict[str, float] = {}
    indexes: list[int] = []
    fields = (
        "utilization_pct",
        "memory_utilization_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "power_limit_w",
        "temperature_c",
    )
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != len(fields) + 1:
            continue
        index = int(row[0])
        indexes.append(index)
        for field, raw_value in zip(fields, row[1:], strict=True):
            try:
                metrics[f"gpu/{index}/{field}"] = float(raw_value)
            except ValueError:
                continue
    return indexes, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--group", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--ready-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be positive")

    import wandb

    gpu_indexes, initial_metrics = read_gpu_metrics()
    settings = wandb.Settings(
        x_stats_gpu_device_ids=gpu_indexes,
        x_stats_sampling_interval=float(args.interval),
    )
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        group=args.group,
        name=args.name,
        job_type="gpu-monitor",
        mode=args.mode,
        dir=args.directory,
        settings=settings,
        config={"node_rank": args.node_rank, "node_ip": args.node_ip, "gpu_indexes": gpu_indexes},
    )
    if args.ready_file:
        ready_file = Path(args.ready_file)
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text(f"{run.id}\t{run.url or 'offline'}\n", encoding="utf-8")

    stopped = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sample = 0
    metrics = initial_metrics
    try:
        while not stopped.is_set():
            metrics["monitor/sample"] = sample
            metrics["monitor/node_rank"] = args.node_rank
            run.log(metrics, step=sample)
            sample += 1
            if stopped.wait(args.interval):
                break
            _, metrics = read_gpu_metrics()
    finally:
        run.finish(exit_code=0)


if __name__ == "__main__":
    main()
