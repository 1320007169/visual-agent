#!/usr/bin/env python3
"""Coordinate a fresh launch token through ModelArts shared storage."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def cluster_identity(env):
    raw = env.get("VC_WORKER_HOSTS") or env.get("MA_WORKER_HOSTS") or env.get("WORKER_HOSTS")
    hosts = []
    if raw:
        try:
            hosts = json.loads(raw)
        except json.JSONDecodeError:
            hosts = [item.strip().strip("'\"") for item in raw.strip("[]").split(",") if item.strip()]
        if isinstance(hosts, dict):
            hosts = next((hosts[key] for key in ("hosts", "worker_hosts", "workers") if key in hosts), [])
        if not isinstance(hosts, list):
            raise ValueError("platform worker host list must contain an array of hosts")
        hosts = [str(host) for host in hosts]
    nodes = int(env.get("NNODES", "1"))
    if hosts and len(hosts) != nodes:
        raise ValueError(f"platform host list has {len(hosts)} nodes; NNODES={nodes}")
    rank = next((env[key] for key in ("NODE_RANK", "MA_NODE_RANK", "VC_TASK_INDEX", "SLURM_NODEID", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK") if env.get(key)), None)
    if rank is None and hosts:
        names = {env.get("MA_CURRENT_HOST", ""), env.get("HOSTNAME", ""), socket.gethostname(), socket.getfqdn()}
        names |= {name.split(".")[0] for name in names}
        rank = next((str(index) for index, host in enumerate(hosts) if host in names or host.split(".")[0] in names), None)
    master = env.get("MASTER_ADDR") or env.get("MA_MASTER_ADDR")
    if rank is None or (not hosts and not master):
        raise ValueError("cannot detect cluster node information; provide the platform worker host list or MASTER_ADDR and NODE_RANK")
    rank = int(rank)
    if not 0 <= rank < nodes:
        raise ValueError(f"node rank {rank} is outside NNODES={nodes}")
    return nodes, rank, hosts or [master]


def resolve(env):
    nodes, rank, hosts = cluster_identity(env)
    identity = json.dumps([nodes, hosts, env.get("RUN_ID", "visual_agent_rl")])
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    root = Path(env["BASE"]) / "tmp/visual-agent-run-tokens" / key
    root.mkdir(parents=True, exist_ok=True)
    timeout = float(env.get("RUN_TOKEN_TIMEOUT_SECONDS", "600"))
    if timeout <= 0:
        raise ValueError("RUN_TOKEN_TIMEOUT_SECONDS must be positive")
    deadline = time.monotonic() + timeout
    nonce = uuid.uuid4().hex
    generation_path = root / "generation.json"
    ready_path = root / "ready.json"
    generation = None
    if rank == 0:
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + nonce[:8]
        generation = {"token": token}
        write_json(generation_path, generation)
    acknowledged_token = None
    while time.monotonic() < deadline:
        current = generation if rank == 0 else read_json(generation_path)
        if current:
            token = current["token"]
            if acknowledged_token != token:
                write_json(root / f"node-{rank}.json", {"token": token, "nonce": nonce})
                acknowledged_token = token
            if rank == 0:
                arrivals = [read_json(root / f"node-{index}.json") for index in range(nodes)]
                if all(item and item.get("token") == token for item in arrivals):
                    write_json(ready_path, {"token": token, "nonces": [item["nonce"] for item in arrivals]})
                    return token
            else:
                ready = read_json(ready_path)
                if ready and ready.get("token") == token and ready.get("nonces", [])[rank:rank + 1] == [nonce]:
                    return token
        time.sleep(0.1)
    raise TimeoutError(f"node {rank} timed out waiting for all {nodes} nodes to initialize run directories")


if __name__ == "__main__":
    try:
        print(resolve(os.environ))
    except (ValueError, TimeoutError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
