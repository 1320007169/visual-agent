"""Real CPU FSDP/Adam continuation across world sizes; no GPUs required."""

import functools
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.fsdp_reshard import _slice_shards, reshard_optimizer_state


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 7)

    def forward(self, x):
        return self.linear(x).tanh()


def _worker(rank, world, directory, resume):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root}/init_{world}", rank=rank, world_size=world)
    try:
        torch.manual_seed(7)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
        model = FSDP(torch.nn.Sequential(Block(), Block(), torch.nn.Linear(7, 2)),
                     device_id=torch.device("cpu"), use_orig_params=False, device_mesh=mesh,
                     auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        manager = FSDPCheckpointManager(model, optimizer, scheduler, processing_class=object(),
                                       checkpoint_contents=OmegaConf.create({"allow_world_size_change": world != 3}))
        x = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 28

        def update():
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
            scheduler.step()

        def save(path):
            path.mkdir(exist_ok=True)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig()):
                torch.save(model.state_dict(), path / f"model_world_size_{world}_rank_{rank}.pt")
            torch.save(optimizer.state_dict(), path / f"optim_world_size_{world}_rank_{rank}.pt")
            torch.save({"lr_scheduler": scheduler.state_dict(), "rng": manager.get_rng_state()},
                       path / f"extra_state_world_size_{world}_rank_{rank}.pt")
            dist.barrier()

        if not resume:
            update()
            update()
            save(root / "source")
            update()
            save(root / "expected")
        else:
            manager.load_checkpoint(str(root / "source"))
            assert scheduler.last_epoch == 2
            assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003 * 0.9 ** 2)
            assert all(state["step"].item() == 2 for state in optimizer.state.values())
            update()
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig()):
                from verl.utils.checkpoint.fsdp_reshard import reshard_model_state

                actual = model.state_dict()
                expected = reshard_model_state(root / "expected", 3, actual)
                for key in actual:
                    torch.testing.assert_close(actual[key].to_local(), expected[key].to_local(), rtol=2e-5, atol=1e-7)
            expected_optim = reshard_optimizer_state(root / "expected", 3, optimizer, rank, world)
            for key, state in optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, expected_optim["state"][key][name], rtol=2e-5, atol=1e-7)
            assert scheduler.last_epoch == 3
            save(root / f"resumed_{world}")
            # Subsequent checkpoints use the new topology and the ordinary loader.
            manager.allow_world_size_change = False
            manager.load_checkpoint(str(root / f"resumed_{world}"))
            assert scheduler.last_epoch == 3
            assert all(state["step"].item() == 3 for state in optimizer.state.values())
            (root / f"passed_{world}_{rank}").write_text("model, Adam moments, step, scheduler and next update matched\n")
    finally:
        dist.destroy_process_group()


def test_real_fsdp_three_to_two_rank_resume_and_update(tmp_path):
    mp.spawn(_worker, args=(3, str(tmp_path), False), nprocs=3, join=True)
    mp.spawn(_worker, args=(2, str(tmp_path), True), nprocs=2, join=True)
    # Default loading on the original world size must still match the same update.
    mp.spawn(_worker, args=(3, str(tmp_path), True), nprocs=3, join=True)
    assert all((tmp_path / f"passed_{world}_{rank}").is_file() for world in (2, 3) for rank in range(world))


def test_slice_covers_uneven_shards_and_excludes_old_padding():
    shards = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6, 99, 99])]
    assert _slice_shards(shards, 4, 4, 7).tolist() == [4, 5, 6, 0]
    with pytest.raises(ValueError, match="cover"):
        _slice_shards(shards[:1], 4, 4, 7)
