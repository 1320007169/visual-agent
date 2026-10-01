import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "prepare_rl_data_continuation", Path(__file__).resolve().parents[1] / "scripts/prepare_rl_data_continuation.py",
)
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


def inputs(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    rows = [{"data_source": "visual-agent-zwz-relation", "images": [f"/old/{i}.png"],
             "split": "train", "question": f"Q{i}", "solution": "left", "uid": f"old_{i}"}
            for i in range(23)]
    added = [{"data_source": "visual-agent-ocr" if i < 4 else "visual-agent-chartqa",
              "images": [f"/added/{i}.png"], "split": "train", "question": f"new Q{i}",
              "solution": "5", "uid": f"new_{i}"} for i in range(8)]
    pq.write_table(pa.Table.from_pylist(rows), old_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(list(reversed(rows + added))), new_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist([{**added[0], "split": "val", "images": ["/val.png"]}]), new_dir / "val.parquet")
    checkpoint = tmp_path / "original" / "global_step_2"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "sentinel").write_text("source weights")
    loader = prepare_module.index_loader(23, 4, 2, 1, True)
    iterator = iter(loader)
    prefix = [*next(iterator).tolist(), *next(iterator).tolist()]
    torch.save(loader.state_dict(), checkpoint / "data.pt")
    rollout = checkpoint / "rollout.jsonl"
    rollout.write_text("".join(json.dumps({"step": 2, "source_metadata": rows[i]}) + "\n" for i in prefix[-4:]))
    return old_dir, new_dir, checkpoint, prefix


def test_prepare_preserves_weights_skips_seen_and_includes_every_added_row(tmp_path):
    old_dir, new_dir, source, prefix = inputs(tmp_path)
    original_state = (source / "data.pt").read_bytes()
    out = tmp_path / "output"
    manifest = prepare_module.prepare(source, old_dir, new_dir, out, source / "rollout.jsonl", batch_size=4)
    rows = pq.read_table(out / "train.parquet").to_pylist()
    assert [row["uid"] for row in rows[:8]] == [f"old_{i}" for i in prefix]
    future = rows[8:28]
    assert not {row["uid"] for row in future} & {row["uid"] for row in rows[:8]}
    assert {row["uid"] for row in future if row["uid"].startswith("new_")} == {f"new_{i}" for i in range(8)}
    assert all(row["uid"].startswith("old_") for row in rows[28:])
    assert manifest["total_training_steps"] == 7
    assert manifest["remaining_training_steps"] == 5
    assert len(set(json.dumps(row, sort_keys=True) for row in rows)) == 31
    checkpoint = out / "resume/global_step_2"
    assert (checkpoint / "actor").resolve() == source / "actor"
    assert (source / "data.pt").read_bytes() == original_state
    assert (checkpoint / "actor/sentinel").read_text() == "source weights"
    loader = prepare_module.index_loader(31, 4, 2, 999, False)
    loader.load_state_dict(torch.load(checkpoint / "data.pt", weights_only=False))
    iterator = iter(loader)
    assert next(iterator).tolist() == list(range(8, 12))
    new_state = loader.state_dict()
    another = prepare_module.index_loader(31, 4, 2, 999, False)
    another.load_state_dict(new_state)
    assert [index for batch in another for index in batch.tolist()] == list(range(12, 28))
    second = prepare_module.prepare(source, old_dir, new_dir, tmp_path / "second", source / "rollout.jsonl", batch_size=4)
    assert second["schedule"] == manifest["schedule"]


def test_reject_wrong_sampler_seed_and_changed_old_rows(tmp_path):
    old_dir, new_dir, source, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="cursor"):
        prepare_module.prepare(source, old_dir, new_dir, tmp_path / "wrong_seed", source / "rollout.jsonl", batch_size=4, sampler_seed=99)
    table = pq.read_table(new_dir / "train.parquet")
    rows = table.to_pylist()
    rows[-1]["solution"] = "changed"
    pq.write_table(pa.Table.from_pylist(rows), new_dir / "train.parquet")
    with pytest.raises(ValueError, match="preserve"):
        prepare_module.prepare(source, old_dir, new_dir, tmp_path / "wrong_rows", source / "rollout.jsonl", batch_size=4)
