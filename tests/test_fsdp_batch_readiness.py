import ast
from datetime import timedelta
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf


WORKER = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/workers/fsdp_workers.py"


class ForwardStarted(Exception):
    pass


def delayed_batch_worker(rank, directory, early_arrived, late_arrived, results):
    torch.set_num_threads(1)
    dist.init_process_group("cpu:gloo", init_method=f"file://{directory}/rendezvous",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    tree = ast.parse(WORKER.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ActorRolloutRefWorker")

    class ShardingManager:
        def __enter__(self):
            results.put((method_name, rank, late_arrived.is_set()))
            raise ForwardStarted

        def __exit__(self, *args):
            pass

    config = OmegaConf.create({
        "actor": {"entropy_coeff": 0},
        "rollout": {"log_prob_micro_batch_size_per_gpu": 1,
                    "log_prob_max_token_len_per_gpu": 32,
                    "log_prob_use_dynamic_bsz": False, "temperature": 1},
        "ref": {"log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 32, "log_prob_use_dynamic_bsz": False},
    })
    instance = SimpleNamespace(
        config=config, _is_actor=True, _is_ref=True, _is_lora=False,
        _is_offload_param=False, _is_offload_optimizer=False,
        ulysses_sharding_manager=ShardingManager(),
    )
    for method_name in ("compute_log_prob", "compute_ref_log_prob", "update_actor"):
        if rank == 0:
            early_arrived.clear()
            late_arrived.clear()
        dist.barrier()
        method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                      and node.name == method_name)
        method.decorator_list = []
        for arg in method.args.args:
            arg.annotation = None
        namespace = {"torch": torch, "dist": dist, "get_device_id": lambda: "cuda:0"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
        if rank == 0:
            early_arrived.set()
        else:
            assert early_arrived.wait(10)
            time.sleep(0.3)
            late_arrived.set()
        data = SimpleNamespace(meta_info={})
        data.to = lambda device: data
        try:
            namespace[method_name](instance, data)
        except ForwardStarted:
            pass
        dist.barrier()
    dist.destroy_process_group()


class FSDPBatchReadinessTest(unittest.TestCase):
    def test_delayed_rank_receives_batch_before_any_rank_starts_forward(self):
        context = mp.get_context("spawn")
        early_arrived, late_arrived, results = context.Event(), context.Event(), context.Queue()
        directory = tempfile.mkdtemp(prefix="fsdp-batch-readiness-")
        mp.spawn(delayed_batch_worker, args=(directory, early_arrived, late_arrived, results),
                 nprocs=2, join=True)
        observations = [results.get(timeout=5) for _ in range(6)]
        self.assertEqual({(name, rank) for name, rank, _ in observations}, {
            (name, rank) for name in ("compute_log_prob", "compute_ref_log_prob", "update_actor")
            for rank in range(2)
        })
        self.assertTrue(all(ready for _, _, ready in observations), observations)


if __name__ == "__main__":
    unittest.main()
