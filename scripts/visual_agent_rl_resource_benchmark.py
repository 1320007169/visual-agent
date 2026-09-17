#!/usr/bin/env python3
"""Prepare one two-node RL resource comparison job; preview unless --run."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def cpu_partition(allowed, threads, servers):
    count = threads * servers
    if threads < 1 or servers < 1 or len(allowed) - count < 16:
        raise ValueError("CPU scenario needs tool threads * servers plus at least 16 training CPUs per node")
    # Logical IDs, not physical cores; record the actual allocation per node.
    ordered = sorted(allowed)
    return ordered[:-count], ordered[-count:]


def configuration(args, parent, allowed):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, underscore and hyphen")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be nonnegative and steps positive")
    repo = Path(__file__).resolve().parents[1]
    base = Path(parent.get("BASE", str(repo.parent)))
    root = Path(args.output_root or repo / "outputs/rl-resource-benchmark").resolve() / args.run_id / args.scenario
    env = parent.copy()
    # Never inherit production resume pointers, endpoints or output targets.
    for key in ("CONTINUE_FROM_PATH", "RESUME_FROM_PATH", "WARM_START_DATA_PATH", "WARM_START_GLOBAL_STEP",
                "VISUAL_TOOL_API_BASE", "VISUAL_TOOL_API_BASES", "LOCAL_VISUAL_TOOL_API_BASES",
                "LOG_FILE", "TOOL_LOG_FILE", "GPU_MONITOR_LOG", "GPU_PROCESS_MONITOR_LOG",
                "TRAIN_DRIVER_SCRIPT", "BEST_HF_MODEL_DIR", "RL_CPU_COUNT", "VISUAL_TOOL_CPU_CORES",
                "VISUAL_TOOL_CPU_THREADS", "DRY_RUN", "MIXED_CONFIG_ONLY"):
        env.pop(key, None)
    train_cores = sorted(allowed)
    tool_cores = []
    servers, replicas = (1, 1) if args.scenario == "B" else (2, 3)
    if args.scenario == "C":
        servers, replicas = args.cpu_servers, 1
        train_cores, tool_cores = cpu_partition(allowed, args.cpu_threads, servers)
    run_name = f"rl_resource_{args.run_id}_{args.scenario}"
    env.update({
        "BASE": str(base), "REPO_ROOT": str(repo), "RL_ROOT": str(repo / "reinforcement_learning"),
        "NNODES": "2", "RUN_ID": run_name, "MODEL_PATH": str(Path(args.model_path).resolve()),
        "RESUME_MODE": "disable", "RL_OUTPUT_DIR": str(root / "state"),
        "OUTPUT_DIR": str(root / "state"), "RL_LOG_DIR": str(root / "logs"),
        "LOG_DIR": str(root / "logs"), "ROLLOUT_DATA_DIR": str(root / "rollouts"),
        "SYNC_DIR": str(root / "sync"), "RAY_STORAGE_ROOT": str(root / "ray"),
        "TRAIN_BATCH_SIZE": "112", "PPO_MINI_BATCH_SIZE": "28", "ROLLOUT_N": "8",
        "MAX_CONCURRENT_REQUESTS": "112", "MAX_TURNS": "6", "TRAIN_SHUFFLE": "True",
        "DUAL_STREAM_ENABLE": "False", "AGENT_ROLLOUT_N": "8", "NATIVE_ROLLOUT_N": "0",
        "TOTAL_TRAINING_STEPS": str(args.warmup + args.steps), "TOTAL_EPOCHS": "1",
        "SAVE_FREQ": "-1", "TEST_FREQ": "-1", "SAVE_HF_MODEL": "0",
        "SAVE_BEST_ONLY": "False", "SAVE_BEST_HF_MODEL": "False", "VAL_BEFORE_TRAIN": "False",
        "POST_TRAIN_SCRIPT": "", "WAIT_FOR_POST_TRAIN_ON_WORKERS": "0", "WANDB_ENABLE": "0",
        "GPU_MONITOR_ENABLE": "1", "START_VISUAL_TOOL_SERVER": "1",
        "VISUAL_TOOL_BACKEND": "groundingdino", "SAM3_REPLICAS": "0",
        "VISUAL_TOOL_SERVERS_PER_NODE": str(servers), "GROUNDING_DINO_REPLICAS": str(replicas),
        "VISUAL_TOOL_DEVICE": "cpu" if args.scenario == "C" else "cuda",
        "RL_CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7" if args.scenario == "C" else "0,1,2,3,4,5,6",
        "TOOL_CUDA_VISIBLE_DEVICES": "7", "RL_CPU_COUNT": str(len(train_cores)),
    })
    if tool_cores:
        env.update(VISUAL_TOOL_CPU_CORES=",".join(map(str, tool_cores)),
                   VISUAL_TOOL_CPU_THREADS=str(args.cpu_threads))
    # The mixed launcher preserves these explicit benchmark switches.
    manifest = {
        "scenario": args.scenario, "run_id": args.run_id, "run_name": run_name,
        "model_path": env["MODEL_PATH"], "warmup_steps": args.warmup, "measured_steps": args.steps,
        "total_steps": args.warmup + args.steps, "training_gpus": 16 if args.scenario == "C" else 14,
        "tool_device": env["VISUAL_TOOL_DEVICE"], "tool_servers_per_node": servers,
        "tool_replicas_per_server": replicas, "cpu_threads_per_server": args.cpu_threads if tool_cores else None,
        "train_batch_size": 112, "ppo_mini_batch_size": 28, "rollout_n": 8,
        "max_turns": 6, "temperature": 1, "max_concurrent_requests": 112,
        "train_files": env.get("TRAIN_FILES", str(Path(env.get("ZWZ_RL_DIR", str(repo / "data/zwz_deepeyesv2_3k_nocount_hr4k_v1"))) / "train.parquet")),
        "optimizer_resume": False, "output": str(root),
    }
    return env, manifest, train_cores, tool_cores


def register(root, manifest, local):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "benchmark.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Benchmark configuration differs between nodes; use a new run-id")
        else:
            if (root / "logs").exists() or (root / "state").exists():
                raise ValueError("Refusing to reuse existing training outputs")
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        if list(root.glob("exit-*.json")) or len(list(root.glob("node-*.json"))) >= 2:
            raise ValueError("This two-node run has already launched; use a new run-id")
        # A retry must use a new run-id, never append another run's steps.
        with (root / f"node-{socket.gethostname()}.json").open("x") as stream:
            json.dump(local, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("A", "B", "C"), required=True)
    parser.add_argument("--model-path", required=True, help="Same frozen HF checkpoint in all three jobs")
    parser.add_argument("--run-id", required=True, help="Same comparison ID across A/B/C; new ID for retries")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--cpu-servers", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        env, manifest, train_cores, tool_cores = configuration(args, os.environ, os.sched_getaffinity(0))
    except ValueError as exc:
        parser.error(str(exc))
    local = {"hostname": socket.gethostname(), "training_cpu_ids": train_cores, "tool_cpu_ids": tool_cores}
    print(json.dumps({"benchmark": manifest, "node": local}, indent=2), flush=True)
    if not args.run:
        print("Preview only. Submit on both nodes of a dedicated ModelArts job with --run.")
        return
    model = Path(manifest["model_path"])
    if not (model / "config.json").is_file():
        parser.error("HF model config.json is missing")
    index = model / "model.safetensors.index.json"
    if not index.is_file() or not all((model / name).is_file() for name in set(json.loads(index.read_text())["weight_map"].values())):
        parser.error("Complete sharded HF weights are required; do not pass a raw FSDP global_step directory")
    root = Path(manifest["output"])
    register(root, manifest, local)
    os.sched_setaffinity(0, train_cores)
    started = time.time()
    command = ["bash", str(Path(__file__).with_name("run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh"))]
    result = subprocess.run(command, env=env)
    (root / f"exit-{socket.gethostname()}.json").write_text(json.dumps({
        "exit_code": result.returncode, "wall_seconds_including_startup": time.time() - started,
    }, indent=2) + "\n")
    print(f"Benchmark finished: exit={result.returncode}, logs={root / 'logs'}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
