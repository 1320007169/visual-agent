#!/usr/bin/env python3
"""Compare isolated GroundingDINO CPU/GPU services; preview unless --run is given."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from benchmark_visual_tool_pool import LeastPendingEndpoints, execute_request, run_level


def integers(value):
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            result.extend(range(first, last + 1))
        elif part.strip():
            result.append(int(part))
    if len(set(result)) != len(result) or any(x < 0 for x in result):
        raise ValueError("IDs must be unique nonnegative integers")
    return result


def scenarios(gpus, cores, threads, workers):
    plans = []
    for count in workers:
        for nthreads in threads:
            if count * nthreads <= len(cores):
                plans.append({"name": f"cpu_w{count}_t{nthreads}", "device": "cpu",
                              "workers": count, "replicas": 1, "threads": nthreads})
    if gpus:
        for workers_count, replicas in ((1, 1), (2, 3)):
            plans.append({"name": f"gpu1_w{workers_count}_r{replicas}", "device": "cuda:0",
                          "workers": workers_count, "replicas": replicas, "threads": min(4, len(cores)),
                          "gpus": gpus[:1]})
    if len(gpus) >= 2:
        plans.append({"name": "gpu2_w4_r3", "device": "cuda:0", "workers": 4,
                      "replicas": 3, "threads": min(4, len(cores)), "gpus": gpus[:2]})
    return plans


def load_payloads(path):
    payloads = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        image_path = Path(case["image"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payloads.append(json.dumps({
            "instance_id": f"resource-benchmark-{len(payloads)}",
            "name": case.get("name", "sam3_segment_multi"),
            "arguments": case.get("arguments", {
                "queries": [{"role": "target", "query": case.get("query", "person")}],
                "target_image": 0,
            }),
            "images": [f"data:{mime};base64,{encoded}"],
        }).encode())
    if not payloads:
        raise ValueError("Cases file is empty")
    return payloads


def stop_processes(processes):
    # Only signal groups created by this benchmark, never existing tool services.
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def wait_ready(processes, endpoints, token, timeout):
    pending = set(endpoints)
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while pending:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("Tool process exited; inspect scenario server logs")
        if time.monotonic() >= deadline:
            raise TimeoutError("Tool startup timed out")
        for endpoint in list(pending):
            try:
                request = urllib.request.Request(endpoint + "/health", headers={"Authorization": "Bearer " + token})
                with opener.open(request, timeout=2) as response:
                    health = json.load(response)
                if health.get("status") == "ok" and health.get("grounding_dino_loaded"):
                    pending.remove(endpoint)
            except (OSError, ValueError):
                pass
        if pending:
            time.sleep(0.5)


def run_scenario(plan, args, payloads):
    processes, logs, endpoints = [], [], []
    token = os.environ["VISUAL_TOOL_API_KEY"]
    started = time.perf_counter()
    # Set affinity and PyTorch thread pools before loading the server/model.
    launcher = (
        "import os,runpy,torch; "
        "os.sched_setaffinity(0, {cores!r}); "
        "torch.set_num_threads({threads}); torch.set_num_interop_threads(1); "
        "runpy.run_path({script!r}, run_name='__main__')"
    )
    result = {"scenario": plan, "levels": []}
    try:
        for i in range(plan["workers"]):
            port = args.port + i
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            threads = min(plan["threads"], len(args.core_ids))
            cores = args.core_ids
            if plan["device"] == "cpu":
                cores = cores[i * threads:(i + 1) * threads]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "" if plan["device"] == "cpu" else str(plan["gpus"][i % len(plan["gpus"])]),
                "GROUNDING_DINO_DEVICE": plan["device"],
                "GROUNDING_DINO_MODEL_PATH": str(args.model_path.resolve()),
                "GROUNDING_DINO_REPLICAS": str(plan["replicas"]),
                "SAM3_REPLICAS": "0", "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            })
            log = (args.output / f"{plan['name']}-server{i}.log").open("w")
            logs.append(log)
            command = [args.tool_python, "-c", launcher.format(
                cores=cores, threads=threads,
                script=str(Path(__file__).with_name("visual_tool_server.py").resolve())),
                "--backend", "groundingdino", "--host", "127.0.0.1", "--port", str(port)]
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            endpoints.append(f"http://127.0.0.1:{port}")
        wait_ready(processes, endpoints, token, args.startup_timeout)
        result["startup_s"] = round(time.perf_counter() - started, 3)
        for endpoint in endpoints:
            for i in range(args.warmup):
                response = execute_request(LeastPendingEndpoints([endpoint]), payloads[i % len(payloads)], args.timeout)
                if not response["ok"]:
                    raise RuntimeError(f"Warmup failed: {response.get('error', 'tool error')}")
        for repeat in range(args.repeats):
            for concurrency in args.levels:
                level = run_level(endpoints, payloads, concurrency, args.requests, args.timeout)
                level["repeat"] = repeat + 1
                result["levels"].append(level)
                print(f"{plan['name']} repeat={repeat+1} c={concurrency}: "
                      f"{level['elapsed_s']}s, {level['throughput_rps']} req/s, "
                      f"p95={level['client_p95_ms']}ms, failures={level['failed']}", flush=True)
                if level["failed"]:
                    # Client timeouts do not cancel server inference. Avoid contaminated later levels.
                    raise RuntimeError("Requests failed; stopping this scenario before further measurements")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{plan['name']}: {result['error']}", flush=True)
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()
    return result


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def save_results(output, report):
    (output / "results.json").write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
    fields = ["scenario", "repeat", "concurrency", "requests", "successful", "failed",
              "elapsed_s", "throughput_rps", "client_p50_ms", "client_p95_ms", "client_p99_ms"]
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in report["results"]:
            for level in result["levels"]:
                writer.writerow({**level, "scenario": result["scenario"]["name"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSONL: image + query, or image + name + arguments")
    parser.add_argument("--model-path", type=Path, required=True, help="Existing local Transformers GroundingDINO directory")
    parser.add_argument("--tool-python", default=sys.executable)
    parser.add_argument("--cpu-cores", required=True, help="Reserved logical CPU IDs, e.g. 32-63")
    parser.add_argument("--cpu-threads", default="4,8,16")
    parser.add_argument("--cpu-workers", default="1,2")
    parser.add_argument("--gpu-ids", default="", help="Reserved GPU IDs; omitted means CPU only")
    parser.add_argument("--only", default="", help="Comma-separated scenario names")
    parser.add_argument("--concurrency", default="1,8,32,112")
    parser.add_argument("--requests", type=int, default=112)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrites previous runs")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        args.core_ids = integers(args.cpu_cores)
        args.levels = integers(args.concurrency)
        threads, workers, gpus = integers(args.cpu_threads), integers(args.cpu_workers), integers(args.gpu_ids)
        if not args.core_ids or not args.levels or any(x <= 0 for x in args.levels + threads + workers):
            raise ValueError("CPU cores/levels must be nonempty; threads/workers/concurrency must be positive")
        if min(args.requests, args.repeats, args.warmup, args.timeout, args.startup_timeout) <= 0:
            raise ValueError("Request counts, repeats, warmup and timeouts must be positive")
        if args.requests < max(args.levels):
            raise ValueError("--requests must be at least the highest concurrency")
        plans = scenarios(gpus, args.core_ids, threads, workers)
        if args.only:
            names = set(args.only.split(","))
            if names - {p["name"] for p in plans}:
                raise ValueError("Unknown scenario or insufficient CPU cores for --only")
            plans = [p for p in plans if p["name"] in names]
        if not plans or args.port < 1024 or args.port + max(p["workers"] for p in plans) > 65536:
            raise ValueError("No scenarios fit the CPU budget, or invalid port range")
    except ValueError as exc:
        parser.error(str(exc))
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "plans": plans, "results": []}
    print(json.dumps(report, indent=2), flush=True)
    if not args.run:
        print("Preview only. Reserve idle resources, then add --run.")
        return
    if not set(args.core_ids).issubset(os.sched_getaffinity(0)):
        parser.error("Selected CPU cores are outside this process's allowed affinity")
    if not args.model_path.is_dir():
        parser.error("--model-path must be an existing local model directory")
    payloads = load_payloads(args.cases)
    if args.requests < len(payloads):
        parser.error("--requests must cover every input case at least once")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["VISUAL_TOOL_API_KEY"] = secrets.token_hex(24)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    save_results(args.output, report)
    try:
        for plan in plans:
            report["results"].append(run_scenario(plan, args, payloads))
            save_results(args.output, report)
    finally:
        save_results(args.output, report)
    if any("error" in r for r in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
