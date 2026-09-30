#!/usr/bin/env python3
"""Select local OCR and chart training QA for online Visual-Agent RL."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import random
from pathlib import Path
import sys

from PIL import Image, ImageOps
import pyarrow as pa
import pyarrow.parquet as pq


def image_digest(image: Image.Image, chart: bool = False) -> str:
    image = image.convert("RGB")
    variants = [image]
    if chart:
        variants = [image.transpose(operation) for operation in (
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )] + variants
        variants += [ImageOps.mirror(value) for value in variants]
    return min(hashlib.sha256(str(value.size).encode() + value.tobytes()).hexdigest()
               for value in variants)


def benchmark_digests(eval_dir: Path) -> dict[str, set[str]]:
    csv.field_size_limit(sys.maxsize)
    result = {}
    for name in ("OCRBench", "ChartQA_TEST"):
        with (eval_dir / f"{name}.tsv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        images = {row["index"]: row["image"] for row in rows}
        digests = set()
        for value in set(images.values()):
            while value in images:
                value = images[value]
            with Image.open(io.BytesIO(base64.b64decode(value))) as image:
                digests.add(image_digest(image, chart=name == "ChartQA_TEST"))
        result[name] = digests
    return result


def prepare(data_root: Path, eval_dir: Path, output: Path, limit: int = 5000) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if limit < 100:
        raise ValueError("Each task must contain at least 100 samples")
    heldout = benchmark_digests(eval_dir)
    rng = random.Random(20260930)
    counts = Counter()
    selected = []
    seen = set()
    output.mkdir(parents=True)
    image_dir = output / "images"
    image_dir.mkdir()

    # Original OCR annotations supply QA only; source boxes and CoT are omitted.
    quotas = {"textvqa": limit * 4 // 10, "docvqa": limit * 4 // 10,
              "sroie": limit // 10}
    quotas["infographicsvqa"] = limit - sum(quotas.values())
    for source, quota in quotas.items():
        path = data_root / "Visual-CoT/metadata" / f"{source}_cot_train.jsonl"
        with path.open(encoding="utf-8") as handle:
            candidates = [json.loads(line) for line in handle]
        for index, row in enumerate(candidates):
            row["source_index"] = index
        counts[f"available_{source}"] = sum(row["split"] == "train" for row in candidates)
        rng.shuffle(candidates)
        accepted = 0
        for row in candidates:
            if accepted == quota:
                break
            if row["split"] != "train":
                continue
            image_path = (data_root / "Visual-CoT/cot_image_data/cot_image_data"
                          / source / row["image"]).resolve()
            with Image.open(image_path) as image:
                digest = image_digest(image)
            key = (digest, row["question"])
            if digest in heldout["OCRBench"]:
                counts[f"excluded_benchmark_{source}"] += 1
                continue
            if key in seen:
                counts[f"duplicate_{source}"] += 1
                continue
            seen.add(key)
            aliases = list(dict.fromkeys([row["answer"], *row.get("possible_answers", [])]))
            selected.append({
                "images": [str(image_path)], "question": row["question"],
                "solution": row["answer"], "bbox": [], "source_index": row["source_index"],
                "source_image": str(image_path), "data_source": "visual-agent-ocr",
                "ability": "text_reading", "image_digest": digest,
                "uid": f"{source}_{row['image']}_{hashlib.sha256(row['question'].encode()).hexdigest()[:16]}",
                "original_source": source, "answer_aliases": aliases,
                "transform": "",
            })
            accepted += 1
        if accepted != quota:
            raise ValueError(f"Insufficient {source} samples: {accepted}/{quota}")

    chart_path = data_root / "gx/datasets/codevision_rl/train_group_1.parquet"
    candidates = []
    for batch in pq.ParquetFile(chart_path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if row["extra_info"]["split"] == "train" and row["reward_model"]["style"] == "rule":
                candidates.append(row)
    counts["available_chart"] = len(candidates)
    rng.shuffle(candidates)
    accepted = 0
    for row in candidates:
        if accepted == limit:
            break
        info = row["extra_info"]
        image_bytes = row["images"][0]["bytes"]
        with Image.open(io.BytesIO(image_bytes)) as image:
            digest = image_digest(image, chart=True)
            ocr_digest = image_digest(image)
            suffix = image.format.lower()
        key = (digest, info["question"])
        if digest in heldout["ChartQA_TEST"] or ocr_digest in heldout["OCRBench"]:
            counts["excluded_benchmark_chart"] += 1
            continue
        if key in seen:
            counts["duplicate_chart"] += 1
            continue
        seen.add(key)
        uid = f"codevision_rl_{info['index']}"
        image_path = (image_dir / f"{uid}.{suffix}").resolve()
        image_path.write_bytes(image_bytes)
        aliases = row["reward_model"]["ground_truth"]
        selected.append({
            "images": [str(image_path)], "question": info["question"],
            "solution": aliases[0], "bbox": [], "source_index": info["index"],
            "source_image": str(image_path), "data_source": "visual-agent-chartqa",
            "ability": "chart_reasoning", "image_digest": digest, "uid": uid,
            "original_source": "codevision_rl", "answer_aliases": aliases,
            "transform": info["transform"] or "",
        })
        accepted += 1
    if accepted != limit:
        raise ValueError(f"Insufficient chart samples: {accepted}/{limit}")

    splits = {"train": [], "val": []}
    for row in selected:
        split = "val" if int(row["image_digest"][:8], 16) % 100 < 2 else "train"
        row["split"] = split
        splits[split].append(row)
    summary = {"seed": 20260930, "limit_per_task": limit, "counts": dict(counts),
               "benchmark_image_counts": {key: len(value) for key, value in heldout.items()},
               "exclusion": "Exact decoded RGB pixels; chart rotations and flips included. Recompressed or cropped copies are not detected.",
               "splits": {}}
    for split, rows in splits.items():
        rng.shuffle(rows)
        pq.write_table(pa.Table.from_pylist(rows), output / f"{split}.parquet", compression="zstd")
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][split] = {
            "total": len(rows), "sources": dict(Counter(row["original_source"] for row in rows)),
            "tasks": dict(Counter(row["data_source"] for row in rows)),
        }
        for task in ("ocr", "chartqa"):
            task_rows = [row for row in rows if row["data_source"] == f"visual-agent-{task}"]
            task_dir = output / ("chart" if task == "chartqa" else task)
            task_dir.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pylist(task_rows), task_dir / f"{split}.parquet", compression="zstd")
            with (task_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in task_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.eval_dir, args.output_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
