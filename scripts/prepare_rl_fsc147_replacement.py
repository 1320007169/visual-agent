#!/usr/bin/env python3
"""Keep half the TallyQA rows and add sampled official FSC147 training rows."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def prepare(base_dir: Path, fsc_root: Path, output_dir: Path, seed: int = 20261004,
            fsc_count: int = 3000) -> dict:
    base_dir, fsc_root, output_dir = (p.resolve() for p in (base_dir, fsc_root, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    train = pq.read_table(base_dir / "train.parquet")
    rows = train.to_pylist()
    tally_indices = [index for index, row in enumerate(rows) if row["data_source"] == "visual-agent-tallyqa"]
    if not tally_indices:
        raise ValueError("No TallyQA training rows to sample")
    rng = random.Random(seed)
    retained_indices = set(rng.sample(tally_indices, len(tally_indices) // 2))
    removed = [rows[index] for index in tally_indices if index not in retained_indices]
    splits = json.loads((fsc_root / "Train_Test_Val_FSC_147.json").read_text())
    annotations = json.loads((fsc_root / "annotation_FSC147_384.json").read_text())
    classes = dict(line.split("\t", 1) for line in (fsc_root / "ImageClasses_FSC147.txt").read_text().splitlines() if line.strip())
    filenames = splits["train"]
    if len(filenames) != len(set(filenames)) or set(filenames) & (set(splits["val"]) | set(splits["test"])):
        raise ValueError("FSC147 training images must be unique and disjoint from held-out splits")
    if not 0 < fsc_count <= len(filenames):
        raise ValueError("FSC147 sample count must fit the official training split")
    selected = rng.sample(list(enumerate(filenames)), fsc_count)
    added = []
    for index, filename in selected:
        image_path = fsc_root / "images_384_VarV2" / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        answer = str(len(annotations[filename]["points"]))
        value = {
            "images": [str(image_path)], "source_image": str(image_path),
            "question": f"Count every instance of {classes[filename]} in the image. Respond with only the nonnegative integer count.",
            "solution": answer, "bbox": [], "source_index": index,
            "data_source": "visual-agent-fsc147", "ability": "counting", "split": "train",
            "uid": f"fsc147_train_{Path(filename).stem}", "original_source": "fsc147",
            "image_digest": "", "answer_aliases": [answer], "transform": "",
        }
        added.append({field: value.get(field) for field in train.column_names})
    mixed = [row for index, row in enumerate(rows)
             if row["data_source"] != "visual-agent-tallyqa" or index in retained_indices] + added
    rng.shuffle(mixed)
    output_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(mixed, schema=train.schema), output_dir / "train.parquet", compression="zstd")
    for filename, values in (("train.jsonl", mixed), ("removed_tallyqa.jsonl", removed), ("added_fsc147.jsonl", added)):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for filename in ("val.parquet", "val.jsonl"):
        shutil.copyfile(base_dir / filename, output_dir / filename)
    manifest = {
        "base_data_dir": str(base_dir), "fsc147_root": str(fsc_root),
        "experiment_mode": "new_experiment", "seed": seed,
        "removed_tallyqa_rows": len(removed), "added_fsc147_rows": len(added),
        "retained_tallyqa_rows": len(retained_indices), "available_fsc147_train_rows": len(filenames),
        "train_rows": len(mixed), "validation_rows": pq.ParquetFile(output_dir / "val.parquet").metadata.num_rows,
        "train_sources_before": dict(Counter(row["data_source"] for row in rows)),
        "train_sources_after": dict(Counter(row["data_source"] for row in mixed)),
        "sampling_policy": "Uniform sampling without replacement; retain floor(TallyQA rows / 2) and sample the requested FSC147 count.",
        "split_policy": "Use each selected official FSC147 training image once; no validation/test images. Preserve base validation files byte-for-byte.",
        "input_policy": "Unmarked full images and text category; no exemplar boxes, point annotations, or density maps in model inputs.",
        "answer_policy": "Number of official annotated points for the requested category.",
        "resume_policy": "Fresh data schedule; do not reuse a previous data.pt cursor.",
        "file_sha256": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("train.parquet", "val.parquet")
        },
        "source_sha256": {
            name: hashlib.sha256((fsc_root / name).read_bytes()).hexdigest()
            for name in ("Train_Test_Val_FSC_147.json", "annotation_FSC147_384.json", "ImageClasses_FSC147.txt")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--fsc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--fsc-count", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base_data_dir, args.fsc_root, args.output_dir, args.seed, args.fsc_count), indent=2))


if __name__ == "__main__":
    main()
