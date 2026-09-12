from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl import DataProto
from verl.workers import fsdp_workers


class _ShardingManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def preprocess_data(self, data):
        return data

    def postprocess_data(self, data):
        return data


def test_compute_old_log_prob_skips_entropy_when_coefficient_is_zero():
    calculate_entropy_values = []

    def compute_log_prob(data, calculate_entropy):
        calculate_entropy_values.append(calculate_entropy)
        return torch.zeros((1, 2)), None

    worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker._world_size = 1
    worker.actor = SimpleNamespace(compute_log_prob=compute_log_prob)
    worker.config = SimpleNamespace(
        actor=SimpleNamespace(entropy_coeff=0),
        rollout=SimpleNamespace(
            log_prob_micro_batch_size_per_gpu=1,
            log_prob_max_token_len_per_gpu=1024,
            log_prob_use_dynamic_bsz=False,
            temperature=1.0,
        ),
    )
    worker.ulysses_sharding_manager = _ShardingManager()
    data = DataProto.from_dict(
        tensors={"input_ids": torch.ones((1, 2), dtype=torch.long)},
    )

    with (
        patch.object(fsdp_workers, "get_device_id", return_value="cpu"),
        patch.object(fsdp_workers, "get_torch_device", return_value=SimpleNamespace(empty_cache=lambda: None)),
    ):
        output = worker.compute_log_prob(data)

    assert calculate_entropy_values == [False]
    assert set(output.batch.keys()) == {"old_log_probs"}
