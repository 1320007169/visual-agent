"""Restore one-dimensional FSDP1 DTensor checkpoints on fewer ranks."""

import copy
import math
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, Shard


def checkpoint_world_size(path):
    files = list(Path(path).glob("model_world_size_*_rank_0.pt"))
    if len(files) != 1:
        raise ValueError(f"Expected one source world size in {path}")
    world = int(files[0].name.split("_")[3])
    for component in ("model", "optim", "extra_state"):
        for rank in range(world):
            file = Path(path) / f"{component}_world_size_{world}_rank_{rank}.pt"
            if not file.is_file() or file.stat().st_size == 0:
                raise FileNotFoundError(file)
    return world


def _load_shards(path, component, world):
    return [torch.load(Path(path) / f"{component}_world_size_{world}_rank_{rank}.pt",
                       map_location="cpu", mmap=True, weights_only=False)
            for rank in range(world)]


def _slice_shards(shards, start, length, total):
    """Copy an interval without assembling the full tensor; zero only padding."""
    result = shards[0].new_zeros((length, *shards[0].shape[1:]))
    offset = 0
    copied = 0
    for shard in shards:
        end = offset + shard.shape[0]
        left, right = max(start, offset), min(start + length, end, total)
        if left < right:
            result[left - start:right - start].copy_(shard[left - offset:right - offset])
            copied += right - left
        offset = end
    if copied != max(0, min(length, total - start)):
        raise ValueError("Source shards do not cover the requested tensor interval")
    return result


def reshard_model_state(path, source_world, template):
    shards = _load_shards(path, "model", source_world)
    if any(state.keys() != template.keys() for state in shards):
        raise ValueError("Checkpoint model keys differ from the current model")
    result = {}
    for name, target in template.items():
        values = [state[name] for state in shards]
        if not isinstance(target, DTensor):
            if any(isinstance(value, DTensor) or not torch.equal(value, values[0]) for value in values):
                raise ValueError(f"Unsupported replicated model state: {name}")
            result[name] = values[0].clone()
            continue
        if target.device_mesh.ndim != 1 or target.placements != (Shard(0),):
            raise ValueError("Resharding requires a one-dimensional Shard(0) model layout")
        for value in values:
            if (not isinstance(value, DTensor) or value.placements != (Shard(0),)
                    or value.device_mesh.ndim != 1 or value.device_mesh.size() != source_world
                    or value.shape != target.shape or value.dtype != target.dtype):
                raise ValueError(f"Checkpoint layout/shape/dtype mismatch: {name}")
        world = target.device_mesh.size()
        rank = target.device_mesh.get_local_rank()
        chunk = math.ceil(target.shape[0] / world)
        local = _slice_shards([value.to_local() for value in values], rank * chunk,
                              target.to_local().shape[0], target.shape[0])
        result[name] = DTensor.from_local(local, target.device_mesh, target.placements,
                                         run_check=False, shape=target.shape, stride=target.stride())
    return result


def reshard_optimizer_state(path, source_world, optimizer, rank, world):
    """Repartition Adam moments for unchanged use_orig_params=False wrapping."""
    shards = _load_shards(path, "optim", source_world)
    saved_groups = shards[0]["param_groups"]
    current_groups = optimizer.state_dict()["param_groups"]
    if (len(saved_groups) != len(current_groups)
            or any(a["params"] != b["params"] for a, b in zip(saved_groups, current_groups))
            or any(state["param_groups"] != saved_groups for state in shards)):
        raise ValueError("Optimizer parameter groups differ; keep the original FSDP wrapping")
    result = {"param_groups": copy.deepcopy(saved_groups), "state": {}}
    for group, actual in zip(saved_groups, optimizer.param_groups):
        for parameter_id, parameter in zip(group["params"], actual["params"]):
            if not hasattr(parameter, "_unpadded_unsharded_size"):
                raise ValueError("Optimizer resharding requires FSDP1 flat parameters")
            total = parameter._unpadded_unsharded_size.numel()
            source_chunk, chunk = math.ceil(total / source_world), math.ceil(total / world)
            if parameter.numel() != chunk:
                raise ValueError("Current flat parameter is not fully sharded")
            states = [state["state"][parameter_id] for state in shards]
            if any(set(state) != {"step", "exp_avg", "exp_avg_sq"} for state in states):
                raise ValueError("Only Adam/AdamW step, exp_avg and exp_avg_sq are supported")
            if any(not torch.equal(state["step"], states[0]["step"]) for state in states):
                raise ValueError("Optimizer step counters differ between source ranks")
            result["state"][parameter_id] = {"step": states[0]["step"].clone()}
            for key in ("exp_avg", "exp_avg_sq"):
                values = [state[key] for state in states]
                if any(value.shape != (source_chunk,) or value.dtype != values[0].dtype for value in values):
                    raise ValueError(f"Optimizer shard shape mismatch for parameter {parameter_id}")
                result["state"][parameter_id][key] = _slice_shards(values, rank * chunk, chunk, total)
    return result
