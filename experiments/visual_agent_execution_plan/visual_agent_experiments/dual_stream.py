"""Reference math for the planned, simultaneous two-stream GRPO update."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence


def grouped_outcome_advantages(
    rewards: Sequence[float], group_ids: Sequence[str], stream_ids: Sequence[str], *, normalize_std: bool = True
) -> list[float]:
    """Compute outcome advantages inside each rollout group only.

    A group id must identify `(sample_id, stream_id, batch_id)`. Rejecting a
    group shared by streams prevents Native rewards from becoming an Agent
    baseline, which would invalidate the A/D comparison.
    """
    if not rewards or len(rewards) != len(group_ids) or len(rewards) != len(stream_ids):
        raise ValueError("rewards, group_ids, and stream_ids must be non-empty and have equal lengths")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[str(group_id)].append(index)

    advantages = [0.0] * len(rewards)
    for group_id, indices in grouped.items():
        streams = {stream_ids[index] for index in indices}
        if len(streams) != 1:
            raise ValueError(f"group {group_id!r} mixes streams: {sorted(streams)}")
        values = [float(rewards[index]) for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        for index, value in zip(indices, values, strict=True):
            advantages[index] = (value - mean) / scale if normalize_std and scale > 0 else value - mean
    return advantages


def combine_stream_losses(agent_loss: float, native_loss: float, *, alpha: float = 1.0) -> float:
    """Return `(L_A + alpha * L_N) / (1 + alpha)` for one shared update."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return (float(agent_loss) + alpha * float(native_loss)) / (1.0 + alpha)
