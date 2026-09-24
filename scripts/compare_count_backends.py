#!/usr/bin/env python3
"""Compare counting backends on the exported object_count questions.

The exported questions come from :mod:`export_count_questions`. This script
loads each requested backend through the VTS registry (without starting a
server), runs it on every question, and reports exact-count rate, MAE,
under/over-counting and zero-count errors. It is meant for the CPU/GPU
backend check before spending model time on a Qwen rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_CONFIGS = {
    "countgd_plusplus": "configs/services/countgd_plusplus.yaml",
    "countgd_plusplus_pseudo": "configs/services/countgd_plusplus_pseudo_eval.yaml",
    "groundingdino_1_6_pro_api": "configs/services/groundingdino_1_6_pro_api_eval.yaml",
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_backend(pipeline_root: Path, backend: str, device: str | None = None) -> Any:
    from vts.tool_server import load_service_config
    from vts.tools import build_tool_registry

    config = load_service_config(pipeline_root / BACKEND_CONFIGS[backend])
    if device:
        config["tools"]["definitions"]["count"]["device"] = device
    registry = build_tool_registry(config["tools"])
    return registry.get("countgd_plusplus_count").backend


def metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    evaluated = exact = under = over = zero = 0
    absolute_error = 0
    for record in records:
        value = record.get(key)
        ground_truth = record.get("gt_count")
        if value is None or ground_truth is None:
            continue
        evaluated += 1
        if value == 0:
            zero += 1
        absolute_error += abs(value - ground_truth)
        if value == ground_truth:
            exact += 1
        elif value < ground_truth:
            under += 1
        else:
            over += 1
    return {
        "evaluated": evaluated,
        "exact": exact,
        "exact_rate": round(exact / evaluated, 4) if evaluated else None,
        "mae": round(absolute_error / evaluated, 4) if evaluated else None,
        "under_count": under,
        "over_count": over,
        "zero_count": zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions_jsonl", help="Output of export_count_questions.py")
    parser.add_argument("output_dir", help="Directory for calls and summary JSON")
    parser.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKEND_CONFIGS),
        help="Backend to evaluate; repeatable. Defaults to both.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N questions after --start")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N questions")
    parser.add_argument("--device", default=None, help="Override the backend device, e.g. cpu or cuda")
    parser.add_argument(
        "--pipeline-root",
        default=os.environ.get(
            "PIPELINE_ROOT",
            "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline",
        ),
    )
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root).resolve()
    load_env_file(pipeline_root / ".env")
    sys.path.insert(0, str(pipeline_root / "src"))
    sys.path.insert(0, str(pipeline_root))

    questions = [json.loads(line) for line in Path(args.questions_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = questions[args.start :]
    if args.limit:
        questions = questions[: args.limit]
    backends = args.backend or ["countgd_plusplus", "countgd_plusplus_pseudo"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"questions": len(questions), "baseline_five_tool": metrics(questions, "tool_count"), "backends": {}}
    for backend_name in backends:
        backend = build_backend(pipeline_root, backend_name, args.device)
        backend_calls = []
        for record in questions:
            entry = {
                "index": record.get("index"),
                "query": record.get("query"),
                "gt_count": record.get("gt_count"),
                "backend": backend_name,
            }
            started = time.monotonic()
            try:
                result = backend.count(str(record["image_path"]), str(record["query"]))
                entry.update(
                    {
                        "status": "success",
                        "count": int(result["count"]),
                        "count_mode": result.get("count_mode"),
                        "first_pass_count": result.get("first_pass_count"),
                        "pseudo_exemplar_count": result.get("pseudo_exemplar_count"),
                        "pseudo_exemplar_boxes": result.get("pseudo_exemplar_boxes"),
                        "confidence": result.get("confidence"),
                        "boxes": result.get("boxes"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report the failure, keep going
                entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            entry["latency_seconds"] = round(time.monotonic() - started, 3)
            backend_calls.append(entry)
            calls.append(entry)
        successful = [entry for entry in backend_calls if entry["status"] == "success"]
        summary["backends"][backend_name] = {
            **metrics(successful, "count"),
            "failed": len(backend_calls) - len(successful),
            "count_mode_pseudo_exemplar": sum(
                1 for entry in successful if entry.get("count_mode") == "pseudo_exemplar"
            ),
            "mean_latency_seconds": round(
                sum(entry["latency_seconds"] for entry in backend_calls) / len(backend_calls), 3
            )
            if backend_calls
            else None,
        }
        print(f"{backend_name}: {json.dumps(summary['backends'][backend_name], ensure_ascii=False)}")

    (output_dir / "count_backend_calls.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in calls), encoding="utf-8"
    )
    (output_dir / "count_backend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
