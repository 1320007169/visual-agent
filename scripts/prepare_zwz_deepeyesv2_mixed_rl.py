#!/usr/bin/env python3
"""Build a reproducible ZWZ + DeepEyesV2 perception training version."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def category(question):
    # Answer-format boilerplate contains "word" and "letter" even for non-OCR QA.
    text = "\n".join(line for line in question.lower().splitlines()
                     if not line.strip().startswith(("answer ", "please answer "))
                     and not ("option" in line and "letter" in line))
    if re.search(r"red (?:bounding )?box|<code>|python", text):
        return None
    if re.search(r"how many|number of|most numerous|least numerous|\bcount\b", text):
        return "counting"
    if re.search(r"color|colour|material|shape|made of|made from", text):
        return "attribute"
    if re.search(r"\b(word|words|letter|letters|written|text|read|reads)\b", text):
        return "text"
    return None


def image_digest(source):
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        return digest, image.format


def prepare(source_dir, zwz_dir, output_dir, quotas, seed=20260915, reuse_images=False):
    output_dir = output_dir.resolve()
    if output_dir.exists() and not (
        reuse_images and {p.name for p in output_dir.iterdir()} <= {"images"}
    ):
        raise FileExistsError(f"Refusing to overwrite dataset version: {output_dir}")
    train = pq.read_table(zwz_dir / "train.parquet").to_pylist()
    val = pq.read_table(zwz_dir / "val.parquet").to_pylist()
    pools = defaultdict(list)
    metadata = {}
    files = sorted(source_dir.glob("perception_all_*.parquet"))
    for path in files:
        rows = pq.read_table(path, columns=["extra_info", "reward_model"]).to_pylist()
        for index, row in enumerate(rows):
            question = str((row["extra_info"] or {}).get("question", "")).replace("<image>", "").strip()
            answer = (row["reward_model"] or {}).get("ground_truth")
            kind = category(question)
            if kind not in quotas or not isinstance(answer, str) or not answer.strip():
                continue
            key = (path.name, index)
            metadata[key] = (question, answer.strip(), kind)
            pools[kind].append(key)
    rng = random.Random(seed)
    selected = {}
    for kind, count in quotas.items():
        rng.shuffle(pools[kind])
        # Keep spare candidates to replace duplicate or evaluation images.
        for rank, key in enumerate(pools[kind][:count + max(100, count // 4)]):
            selected[key] = rank
    stats = Counter()
    eligible = defaultdict(list)
    output_dir.mkdir(parents=True, exist_ok=reuse_images)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=reuse_images)
    for path in files:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["images"]):
            for local_index, row in enumerate(batch.to_pylist()):
                key = (path.name, offset + local_index)
                if key not in selected:
                    continue
                if len(row["images"]) != 1 or not row["images"][0].get("bytes"):
                    stats["invalid_image_count"] += 1
                    continue
                raw = row["images"][0]["bytes"]
                try:
                    digest, fmt = image_digest(io.BytesIO(raw))
                except (OSError, ValueError):
                    stats["invalid_image"] += 1
                    continue
                question, answer, kind = metadata[key]
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt)
                if suffix is None:
                    stats["unsupported_image_format"] += 1
                    continue
                image_path = image_dir / (digest + suffix)
                if not image_path.exists() or image_path.stat().st_size != len(raw):
                    image_path.write_bytes(raw)
                eligible[kind].append((selected[key], {
                    "images": [str(image_path)], "question": question,
                    "solution": answer, "bbox": [], "source_index": key[1],
                    "source_image": f"{key[0]}:{key[1]}",
                    "data_source": "visual-agent-deepeyesv2", "ability": kind,
                    "image_digest": digest,
                }))
            offset += batch.num_rows
        print(f"Scanned {path.name}: {dict(stats)}", flush=True)
    added = []
    seen = set()
    for kind, count in quotas.items():
        kept = 0
        for _, row in sorted(eligible[kind], key=lambda item: item[0]):
            key = (row["image_digest"], " ".join(row["question"].lower().split()))
            if key in seen:
                stats["duplicate_question_image"] += 1
                continue
            seen.add(key)
            added.append(row)
            kept += 1
            if kept == count:
                break
        if kept != count:
            raise ValueError(f"Insufficient {kind} samples: {kept}/{count}")
    counts = Counter()
    for row in added:
        train.append(row)
        counts[f"train/{row['ability']}"] += 1
    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            row["split"] = split
            row.setdefault("image_digest", "")
        rng.shuffle(rows)
        pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{split}.parquet", compression="zstd")
    summary = {
        "seed": seed, "source_dir": str(source_dir.resolve()), "zwz_dir": str(zwz_dir.resolve()),
        "quotas": quotas, "added_counts": dict(counts), "filter_counts": dict(stats),
        "train_rows": len(train), "val_rows": len(val),
        "steps_per_epoch_batch112": len(train) // 112,
        "classification": "Question-keyword buckets, not human-verified task labels",
        "evaluation": "HRBench4K after training; legacy val.parquet retained only for trainer compatibility",
        "image_exclusion": "No validation overlap filtering, as requested; added samples deduplicated by decoded image and question",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--zwz-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--reuse-images", action="store_true",
                        help="Reuse images from an interrupted run; refuse completed datasets")
    args = parser.parse_args()
    prepare(args.source_dir, args.zwz_dir, args.output_dir,
            {"attribute": 2000, "text": 1000}, args.seed, args.reuse_images)
