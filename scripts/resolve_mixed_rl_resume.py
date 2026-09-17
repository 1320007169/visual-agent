#!/usr/bin/env python3
"""Resolve fixed-topology resume or explicit cross-topology warm starts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import time


def checkpoint_world(path):
    files = list((path / "actor").glob("model_world_size_*_rank_*.pt"))
    worlds = {int(re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", p.name)[1]) for p in files}
    if len(worlds) != 1:
        raise ValueError(f"Missing or ambiguous model shards: {path}")
    world = worlds.pop()
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world):
            file = path / "actor" / f"{kind}_world_size_{world}_rank_{rank}.pt"
            if not file.is_file():
                raise FileNotFoundError(file)
    if not (path / "data.pt").is_file():
        raise FileNotFoundError(path / "data.pt")
    return world


def read_json_with_retry(path, attempts=50, delay=0.1):
    """Read metadata while another ModelArts node may be publishing it."""
    last_error = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise last_error


def write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve(env):
    nodes = int(env.get("NNODES", "2"))
    if nodes not in (2, 4, 8):
        raise ValueError("NNODES must be 2, 4, or 8 (16/32/64 physical GPUs)")
    source = Path(env["CONTINUE_FROM_PATH"]).resolve() if env.get("CONTINUE_FROM_PATH") else None
    run_id = env.get("RUN_ID", f"zwz_deepeyesv2_3k_nocount_v1_n8_{nodes}node")
    if source and "RUN_ID" not in env:
        run_id += "_from_" + hashlib.sha256(str(source).encode()).hexdigest()[:10]
    repo = Path(env["REPO_ROOT"])
    output = Path(env.get("RL_OUTPUT_DIR", str(repo / "saves/visual_agent_zwz_rl/qwen3" / run_id)))
    model = env.get("MODEL_PATH", str(Path(env["BASE"]) / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
    train_files = env.get("TRAIN_FILES", str(Path(env.get("ZWZ_RL_DIR", str(repo / "data/zwz_deepeyesv2_3k_nocount_hr4k_v1"))) / "train.parquet"))
    result = {"RUN_ID": run_id, "RL_OUTPUT_DIR": str(output), "RESUME_MODE": env.get("RESUME_MODE", "auto"),
              "WARM_START_DATA_PATH": "", "WARM_START_GLOBAL_STEP": "0"}
    metadata_path = output / "mixed_run_config.json"
    metadata = read_json_with_retry(metadata_path) if metadata_path.exists() else None
    if metadata:
        if metadata["nnodes"] != nodes:
            raise ValueError("Output directory belongs to another topology; use CONTINUE_FROM_PATH and a new run directory")
        model = metadata["initial_model_path"]
        if metadata.get("train_files") != train_files:
            raise ValueError("Training dataset changed; start a new run instead of restoring its data cursor")
        if source is None and metadata.get("continue_from"):
            source = Path(metadata["continue_from"])
    pointer = output / "latest_checkpointed_iteration.txt"
    if pointer.exists() and result["RESUME_MODE"] == "disable":
        raise ValueError("Refusing to restart over existing checkpoints; choose a new RUN_ID")
    if pointer.exists() and result["RESUME_MODE"] == "auto":
        step = pointer.read_text().strip()
        if not step.isdigit():
            raise ValueError("Invalid latest checkpoint pointer")
        if checkpoint_world(output / f"global_step_{step}") != nodes * 7:
            raise ValueError("Checkpoint GPU count differs; use CONTINUE_FROM_PATH with a new run directory")
    elif source:
        if result["RESUME_MODE"] != "auto":
            raise ValueError("CONTINUE_FROM_PATH requires RESUME_MODE=auto")
        match = re.fullmatch(r"global_step_(\d+)", source.name)
        if not match:
            raise ValueError("CONTINUE_FROM_PATH must name a global_step_N directory")
        world = checkpoint_world(source)
        source_config = source.parent / "mixed_run_config.json"
        if not source_config.exists():
            raise ValueError("Continuation requires a checkpoint from this mixed-data launcher with mixed_run_config.json")
        source_metadata = read_json_with_retry(source_config)
        if source_metadata.get("train_files") != train_files:
            raise ValueError("Cannot restore the data cursor from a different training dataset")
        model = source_metadata["initial_model_path"]
        if world == nodes * 7:
            result.update(RESUME_MODE="resume_path", RESUME_FROM_PATH=str(source))
        else:
            model = str(source / "actor/huggingface")
            if not (Path(model) / "config.json").is_file():
                raise FileNotFoundError(f"Portable HF weights missing: {model}")
            index = Path(model) / "model.safetensors.index.json"
            weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
            if not all((Path(model) / file).is_file() for file in weights):
                raise FileNotFoundError(f"Incomplete portable HF weights: {model}")
            result.update(WARM_START_DATA_PATH=str(source / "data.pt"), WARM_START_GLOBAL_STEP=match[1])
            print("Cross-topology warm start: optimizer/scheduler/RNG restart; KL reference resets to the source model.", file=sys.stderr)
    elif result["RESUME_MODE"] == "resume_path":
        explicit = Path(env["RESUME_FROM_PATH"])
        if checkpoint_world(explicit) != nodes * 7:
            raise ValueError("Use CONTINUE_FROM_PATH for a cross-topology checkpoint")
    result["MODEL_PATH"] = model
    return result, {"nnodes": nodes, "initial_model_path": model, "train_files": train_files,
                    "continue_from": str(source) if source else None}


if __name__ == "__main__":
    values, metadata = resolve(os.environ)
    if os.environ.get("MIXED_CONFIG_ONLY") == "1":
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")
        sys.exit(0)
    output = Path(values["RL_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    # Every node runs this helper; serialize creation of the shared model origin.
    with (output / ".mixed_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = output / "mixed_run_config.json"
        if target.exists():
            if read_json_with_retry(target) != metadata:
                raise ValueError("Run metadata differs; choose a new RUN_ID")
        else:
            write_json_atomic(target, metadata)
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")
