#!/usr/bin/env python3
"""Mix new QA into the unconsumed part of a one-epoch RL run on CPU."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader


def index_loader(size, batch_size, workers, seed, shuffle):
    dataset = range(size)
    sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
               if shuffle else SequentialSampler(dataset))
    return StatefulDataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              drop_last=True, sampler=sampler,
                              generator=torch.Generator().manual_seed(seed))


def continuation_order(old_size, added_size, state, step, batch_size, workers, sampler_seed, seed):
    original_batches = [batch.tolist() for batch in index_loader(old_size, batch_size, workers, sampler_seed, True)]
    if not 0 < step < len(original_batches):
        raise ValueError("Source step must be inside the first training epoch")
    restored = index_loader(old_size, batch_size, workers, sampler_seed, True)
    restored.load_state_dict(state)
    if [batch.tolist() for batch in restored] != original_batches[step:]:
        raise ValueError("Source cursor does not match the old dataset, batch size, seed and step")
    prefix = [index for batch in original_batches[:step] for index in batch]
    scheduled = {index for batch in original_batches for index in batch}
    old_tail = sorted(set(range(old_size)) - scheduled)
    remainder = (old_size + added_size) % batch_size
    if remainder > len(old_tail):
        raise ValueError("The new drop_last tail would omit scheduled old QA; choose an aligned addition size")
    rng = random.Random(seed)
    dropped = rng.sample(old_tail, remainder)
    excluded = set(prefix) | set(dropped)
    suffix = [index for index in range(old_size + added_size) if index not in excluded]
    rng.shuffle(suffix)
    return prefix + suffix + dropped, len(prefix), dropped


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source_checkpoint, old_data_dir, new_data_dir, output_dir,
            source_rollout, batch_size=126, workers=2, sampler_seed=1, seed=20261001):
    source_checkpoint, old_data_dir, new_data_dir, output_dir = (
        Path(path).resolve() for path in (source_checkpoint, old_data_dir, new_data_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    step = int(source_checkpoint.name.removeprefix("global_step_"))
    if not (source_checkpoint / "actor").is_dir():
        raise FileNotFoundError(source_checkpoint / "actor")
    old_table = pq.read_table(old_data_dir / "train.parquet")
    merged_table = pq.read_table(new_data_dir / "train.parquet")
    validation_table = pq.read_table(new_data_dir / "val.parquet")
    old_rows, merged_rows = old_table.to_pylist(), merged_table.to_pylist()
    added_sources = {"visual-agent-ocr", "visual-agent-chartqa"}
    added = [row for row in merged_rows if row["data_source"] in added_sources]
    if not added or any(row["data_source"] in added_sources for row in old_rows):
        raise ValueError("Expected OCR/Chart to be new sources relative to the original dataset")

    def original_row(row):
        return json.dumps({key: row[key] for key in old_table.column_names}, sort_keys=True)

    retained = [row for row in merged_rows if row["data_source"] not in added_sources]
    if Counter(map(original_row, retained)) != Counter(map(original_row, old_rows)):
        raise ValueError("New dataset must preserve all original rows and their multiplicities")
    if any(row["split"] != "train" for row in merged_rows):
        raise ValueError("Training input contains non-training rows")
    if any(row["split"] != "val" for row in validation_table.to_pylist()):
        raise ValueError("Validation input contains non-validation rows")
    train_images = {path for row in merged_rows for path in row["images"]}
    val_images = {path for row in validation_table.to_pylist() for path in row["images"]}
    if train_images & val_images:
        raise ValueError("Training and validation share image paths")

    source_state = torch.load(source_checkpoint / "data.pt", map_location="cpu", weights_only=False)
    order, prefix_size, dropped = continuation_order(
        len(old_rows), len(added), source_state, step, batch_size, workers, sampler_seed, seed,
    )
    # Plain torch RandomSampler checkpoints do not record their seed. Bind the
    # replay to the saved rollout, rather than trusting a self-consistent replay.
    identity_keys = ("source_index", "source_image", "data_source", "question")
    identity = lambda row: tuple(row.get(key) for key in identity_keys)
    expected = Counter(identity(old_rows[index]) for index in order[prefix_size - batch_size:prefix_size])
    observed = Counter()
    with Path(source_rollout).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["step"] != step:
                raise ValueError("Source rollout step differs from checkpoint step")
            observed[identity(record["source_metadata"])] += 1
    repetitions, remainder = divmod(sum(observed.values()), batch_size)
    if not repetitions or remainder or observed != Counter({key: count * repetitions for key, count in expected.items()}):
        raise ValueError("Replayed cursor batch does not match the actual saved rollout; check sampler seed and old dataset")
    combined = old_rows + added
    rows = [combined[index] for index in order]
    total_steps = len(rows) // batch_size
    future = rows[prefix_size:total_steps * batch_size]
    future_added = [row for row in future if row["data_source"] in added_sources]
    if len(future_added) != len(added):
        raise ValueError("Not all new QA is scheduled for training")

    # Prefix rows preserve global step accounting; the new cursor skips them.
    sequential = index_loader(len(rows), batch_size, workers, seed, False)
    iterator = iter(sequential)
    for _ in range(step):
        next(iterator)
    new_state = sequential.state_dict()
    resumed = index_loader(len(rows), batch_size, workers, seed, False)
    resumed.load_state_dict(new_state)
    remaining_indices = [index for batch in resumed for index in batch.tolist()]
    if remaining_indices != list(range(prefix_size, total_steps * batch_size)):
        raise ValueError("Prepared sequential cursor does not cover exactly the continuation")

    output_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=merged_table.schema), output_dir / "train.parquet", compression="zstd")
    pq.write_table(validation_table, output_dir / "val.parquet", compression="zstd")
    resume_dir = output_dir / "resume" / f"global_step_{step}"
    resume_dir.mkdir(parents=True)
    (resume_dir / "actor").symlink_to(source_checkpoint / "actor", target_is_directory=True)
    torch.save(new_state, resume_dir / "data.pt")
    manifest = {
        "source_checkpoint": str(source_checkpoint), "source_data_sha256": sha256(source_checkpoint / "data.pt"),
        "source_rollout": str(Path(source_rollout).resolve()), "source_rollout_sha256": sha256(source_rollout),
        "old_train_sha256": sha256(old_data_dir / "train.parquet"),
        "new_train_sha256": sha256(new_data_dir / "train.parquet"),
        "seed": seed, "original_sampler_seed": sampler_seed, "batch_size": batch_size,
        "dataloader_num_workers": workers, "train_shuffle": False, "source_step": step,
        "train_rows": len(rows), "already_consumed_rows": prefix_size,
        "remaining_training_rows": len(future), "remaining_training_steps": total_steps - step,
        "total_training_steps": total_steps, "added_training_rows": len(added),
        "future_sources": dict(Counter(row["data_source"] for row in future)),
        "validation_sources": dict(Counter(row["data_source"] for row in validation_table.to_pylist())),
        "dropped_old_tail_indices": dropped,
        "schedule": [{"old_index": index} if index < len(old_rows)
                     else {"added_uid": added[index - len(old_rows)]["uid"]} for index in order],
        "train_sha256": sha256(output_dir / "train.parquet"),
        "val_sha256": sha256(output_dir / "val.parquet"),
        "resume_data_sha256": sha256(resume_dir / "data.pt"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-rollout", required=True, type=Path)
    parser.add_argument("--old-data-dir", required=True, type=Path)
    parser.add_argument("--new-data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sampler-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261001)
    manifest = prepare(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in manifest.items() if key != "schedule"}, indent=2))


if __name__ == "__main__":
    main()
