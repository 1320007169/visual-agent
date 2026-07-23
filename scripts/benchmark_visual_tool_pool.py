#!/usr/bin/env python3
"""Benchmark visual-tool HTTP endpoints with rollout-like concurrency."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


class LeastPendingEndpoints:
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.pending = [0] * len(endpoints)
        self.next_index = 0
        self.lock = threading.Lock()

    def acquire(self) -> tuple[int, str]:
        with self.lock:
            minimum = min(self.pending)
            candidates = [index for index, value in enumerate(self.pending) if value == minimum]
            index = candidates[self.next_index % len(candidates)]
            self.next_index += 1
            self.pending[index] += 1
            return index, self.endpoints[index]

    def release(self, index: int) -> None:
        with self.lock:
            self.pending[index] -= 1


def build_payload(image_path: Path, query: str) -> bytes:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return json.dumps(
        {
            "instance_id": "visual-tool-pool-benchmark",
            "name": "sam3_segment_multi",
            "arguments": {
                "queries": [{"role": "target", "query": query}],
                "target_image": 0,
            },
            "images": [f"data:{mime};base64,{image}"],
        }
    ).encode("utf-8")


def execute_request(pool: LeastPendingEndpoints, payload: bytes, timeout: float) -> dict:
    endpoint_index, endpoint = pool.acquire()
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            endpoint + "/execute",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        return {
            "ok": result.get("status") == "success",
            "client_ms": (time.perf_counter() - started) * 1000,
            "server_ms": float(result.get("metrics", {}).get("latency_ms", math.nan)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "client_ms": (time.perf_counter() - started) * 1000,
            "server_ms": math.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        pool.release(endpoint_index)


def run_level(endpoints: list[str], payload: bytes, concurrency: int, requests: int, timeout: float) -> dict:
    pool = LeastPendingEndpoints(endpoints)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_request, pool, payload, timeout) for _ in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    successful = [result for result in results if result["ok"]]
    client_ms = [result["client_ms"] for result in successful]
    server_ms = [result["server_ms"] for result in successful if not math.isnan(result["server_ms"])]
    ingress_ms = [max(0.0, result["client_ms"] - result["server_ms"]) for result in successful]
    errors: dict[str, int] = {}
    for result in results:
        if result["ok"]:
            continue
        error = result.get("error", "unsuccessful response")
        errors[error] = errors.get(error, 0) + 1
    return {
        "concurrency": concurrency,
        "requests": requests,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "throughput_rps": round(len(successful) / elapsed, 3),
        "client_p50_ms": round(percentile(client_ms, 0.50), 3),
        "client_p90_ms": round(percentile(client_ms, 0.90), 3),
        "client_p99_ms": round(percentile(client_ms, 0.99), 3),
        "server_p50_ms": round(percentile(server_ms, 0.50), 3),
        "server_p90_ms": round(percentile(server_ms, 0.90), 3),
        "ingress_p50_ms": round(percentile(ingress_ms, 0.50), 3),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", required=True, help="Comma-separated endpoint base URLs")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--concurrency", default="16,32,56,80")
    parser.add_argument("--requests-per-level", type=int, default=112)
    parser.add_argument("--query", default="person")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--warmup-per-endpoint", type=int, default=1)
    args = parser.parse_args()

    endpoints = [value.strip().rstrip("/") for value in args.endpoints.split(",") if value.strip()]
    if not endpoints:
        parser.error("--endpoints must contain at least one URL")
    payload = build_payload(args.image, args.query)
    warmup_pool = LeastPendingEndpoints(endpoints)
    for _ in range(args.warmup_per_endpoint * len(endpoints)):
        result = execute_request(warmup_pool, payload, args.timeout)
        if not result["ok"]:
            raise RuntimeError(f"Warmup request failed: {result.get('error')}")

    output = {
        "endpoints": endpoints,
        "image": str(args.image),
        "levels": [
            run_level(endpoints, payload, concurrency, max(args.requests_per_level, concurrency * 2), args.timeout)
            for concurrency in (int(value) for value in args.concurrency.split(","))
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
