#!/usr/bin/env python3
"""Mix the latest RL data with raw depth questions and local TallyQA counting QA."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq


BOX = re.compile(r"\s*\[\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")
FIELDS = (
    "images", "question", "solution", "bbox", "source_index", "source_image",
    "data_source", "ability", "split", "image_digest", "cycle_category", "uid",
    "original_source", "count_complexity",
)


def normalized(row: dict) -> dict:
    return {field: row.get(field) for field in FIELDS}


def depth_rows(path: Path, split: str) -> list[dict]:
    result = []
    for index, source in enumerate(pq.read_table(path).to_pylist()):
        images = source["images"]
        if not images or not all(Path(image).is_file() for image in images):
            raise FileNotFoundError(f"Missing depth image for {source['uid']}")
        question = str(source["question"]).strip()
        if source["source"] == "ca_vqa_multichoice":
            if len(BOX.findall(question)) < 2:
                raise ValueError(f"Missing source boxes for {source['uid']}")
            question = BOX.sub("", question)
            images = [images[-1]]
        if BOX.search(question):
            raise ValueError(f"Source box leaked into question {source['uid']}")
        result.append(normalized({
            "images": images,
            "question": question,
            "solution": str(source["solution"]).strip(),
            "bbox": [],
            "source_index": index,
            "source_image": images[0],
            "data_source": "visual-agent-depth-raw",
            "ability": "depth_distance",
            "split": split,
            "image_digest": "",
            "cycle_category": "",
            "uid": source["uid"],
            "original_source": source["source"],
            "count_complexity": "",
        }))
    return result


def tally_rows(manifest_path: Path, official_path: Path) -> dict[str, list[dict]]:
    official = {int(row["question_id"]): row for row in json.loads(official_path.read_text())}
    result = {"train": [], "val": []}
    seen = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            uid = source["uid"]
            if uid in seen:
                raise ValueError(f"Duplicate TallyQA UID: {uid}")
            seen.add(uid)
            if source["status"] != "ok":
                raise ValueError(f"Unverified TallyQA image: {uid}")
            split = source["rl_split"]
            if split not in result:
                raise ValueError(f"Unsupported TallyQA split: {split}")
            original = official[int(source["question_id"])]
            if original["answer"] != source["answer"] or original["image"] != source["image"]:
                raise ValueError(f"TallyQA manifest mismatch: {uid}")
            image = str(source["image_path"])
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            result[split].append(normalized({
                "images": [image],
                "question": str(original["question"]).strip(),
                "solution": str(original["answer"]),
                "bbox": [],
                "source_index": int(source["question_id"]),
                "source_image": image,
                "data_source": "visual-agent-tallyqa",
                "ability": "counting",
                "split": split,
                "image_digest": "",
                "cycle_category": "",
                "uid": uid,
                "original_source": source["data_source"],
                "count_complexity": "simple" if source["issimple"] else "complex",
            }))
    return result


def sample_tally_rows(rows_by_split: dict[str, list[dict]], limit: int, seed: int) -> dict[str, list[dict]]:
    groups = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            groups.setdefault((split, row["count_complexity"]), []).append(row)
    total = sum(map(len, groups.values()))
    if not 0 < limit <= total:
        raise ValueError(f"TallyQA limit must be between 1 and {total}: {limit}")
    expected = {key: len(rows) * limit / total for key, rows in groups.items()}
    quotas = {key: int(value) for key, value in expected.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(groups, key=lambda key: (expected[key] - quotas[key], len(groups[key])), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = {
        row["uid"]
        for key, rows in sorted(groups.items())
        for row in rng.sample(rows, quotas[key])
    }
    return {
        split: [row for row in rows if row["uid"] in selected]
        for split, rows in rows_by_split.items()
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(base_train: Path, base_val: Path, depth_dir: Path, tally_manifest: Path,
            tally_official: Path, output_dir: Path, seed: int = 20260924,
            tally_limit: int = 5000) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base = {
        "train": [normalized(row) for row in pq.read_table(base_train).to_pylist()],
        "val": [normalized(row) for row in pq.read_table(base_val).to_pylist()],
    }
    depth = {split: depth_rows(depth_dir / f"{split}.parquet", split) for split in base}
    tally = sample_tally_rows(tally_rows(tally_manifest, tally_official), tally_limit, seed)
    tally_train_images = {row["source_image"] for row in tally["train"]}
    tally_val_images = {row["source_image"] for row in tally["val"]}
    if tally_train_images & tally_val_images:
        raise ValueError("TallyQA train and val images overlap")

    output_dir.mkdir(parents=True)
    summary = {"seed": seed, "tally_limit": tally_limit, "sources": {}, "files": {}}
    for split in base:
        new_rows = depth[split] + tally[split]
        mixed = base[split] + new_rows
        if split == "train":
            random.Random(seed).shuffle(mixed)
        write_jsonl(output_dir / f"new_{split}.jsonl", new_rows)
        write_jsonl(output_dir / f"{split}.jsonl", mixed)
        pq.write_table(pa.Table.from_pylist(mixed), output_dir / f"{split}.parquet", compression="zstd")
        summary["sources"][split] = dict(Counter(row["data_source"] for row in mixed))
        summary["files"][split] = {
            "jsonl": str(output_dir / f"{split}.jsonl"),
            "parquet": str(output_dir / f"{split}.parquet"),
            "new_jsonl": str(output_dir / f"new_{split}.jsonl"),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-val", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--tally-manifest", type=Path, required=True)
    parser.add_argument("--tally-official", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tally-limit", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.base_train, args.base_val, args.depth_dir,
        args.tally_manifest, args.tally_official, args.output_dir,
        tally_limit=args.tally_limit,
    ), indent=2))


if __name__ == "__main__":
    main()
