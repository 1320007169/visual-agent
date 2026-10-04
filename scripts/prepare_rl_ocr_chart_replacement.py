#!/usr/bin/env python3
"""Replace part of the relation QA with official training OCR and chart QA."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_ocr_chart_rl import benchmark_digests, image_digest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_file_digests(path: str) -> tuple[str, str]:
    with Image.open(path) as image:
        return image_digest(image), image_digest(image, chart=True)


def hme_candidates(root: Path, count: int, rng: random.Random, blocked: set[str], stats: Counter) -> list[dict]:
    files = sorted((root / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(root / "data/train-*.parquet")
    eligible = []
    offset = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for index, label in enumerate(parquet.read(columns=["label"])["label"].to_pylist()):
            if not isinstance(label, str) or not label.strip() or "<" in label or ">" in label:
                stats["hme_excluded_answer_protocol"] += 1
                continue
            eligible.append(offset + index)
        offset += parquet.metadata.num_rows
    stats["hme_train_rows"] = offset
    rng.shuffle(eligible)
    # Read a reserve for image exclusions while keeping memory bounded to batches.
    priorities = {index: rank for rank, index in enumerate(eligible[:count * 2])}
    accepted = []
    seen = set()
    offset = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        local_index = 0
        for batch in parquet.iter_batches(batch_size=128, columns=["image", "label", "image_path"]):
            for row in batch.to_pylist():
                index = offset + local_index
                local_index += 1
                if index not in priorities:
                    continue
                raw = row["image"]["bytes"]
                with Image.open(io.BytesIO(raw)) as image:
                    digest = image_digest(image)
                if digest in blocked or digest in seen:
                    stats["hme_excluded_image_overlap"] += 1
                    continue
                seen.add(digest)
                accepted.append({
                    "rank": priorities[index], "source_index": index,
                    "image_bytes": raw, "image_digest": digest,
                    "solution": row["label"].strip(), "original_image_path": row["image_path"],
                })
        offset += parquet.metadata.num_rows
    accepted.sort(key=lambda row: row["rank"])
    if len(accepted) < count:
        raise ValueError(f"Insufficient eligible HME100K samples: {len(accepted)}/{count}")
    return accepted[:count]


def chart_candidates(root: Path, count: int, rng: random.Random, blocked_raw: set[str], blocked_chart: set[str], stats: Counter) -> list[dict]:
    train = root / "ChartQA Dataset/train"
    candidates = []
    for kind in ("human", "augmented"):
        path = train / f"train_{kind}.json"
        for index, row in enumerate(json.loads(path.read_text())):
            candidates.append({**row, "annotation_kind": kind, "source_index": index})
    stats["chart_train_rows"] = len(candidates)
    rng.shuffle(candidates)
    accepted = []
    seen = set()
    for row in candidates:
        path = (train / "png" / row["imgname"]).resolve()
        with Image.open(path) as image:
            raw_digest = image_digest(image)
            digest = image_digest(image, chart=True)
        if raw_digest in blocked_raw or digest in blocked_chart or digest in seen:
            stats["chart_excluded_image_overlap"] += 1
            continue
        question, answer = str(row["query"]).strip(), str(row["label"]).strip()
        if not question or not answer or "<" in answer or ">" in answer:
            stats["chart_excluded_answer_protocol"] += 1
            continue
        seen.add(digest)
        accepted.append({**row, "path": str(path), "image_digest": digest, "solution": answer, "question": question})
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(f"Insufficient eligible ChartQA samples: {len(accepted)}/{count}")
    return accepted


def prepare(base_dir: Path, hme_root: Path, chart_root: Path, eval_dir: Path, output_dir: Path,
            ratio: float = 0.2, seed: int = 20261004) -> dict:
    base_dir, hme_root, chart_root, eval_dir, output_dir = (
        path.resolve() for path in (base_dir, hme_root, chart_root, eval_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 0 < ratio <= 1:
        raise ValueError("replacement ratio must be in (0, 1]")
    train = pq.read_table(base_dir / "train.parquet")
    rows = train.to_pylist()
    val = pq.read_table(base_dir / "val.parquet").to_pylist()
    relations = [index for index, row in enumerate(rows) if row["data_source"] == "visual-agent-zwz-relation"]
    replace_count = int(len(relations) * ratio) // 2 * 2
    if replace_count < 2:
        raise ValueError("Replacement must include at least one OCR and one chart sample")
    count = replace_count // 2
    rng = random.Random(seed)
    stats = Counter()
    print("Loading OCRBench and ChartQA benchmark image exclusions.", flush=True)
    heldout = benchmark_digests(eval_dir)
    blocked_raw = set(heldout["OCRBench"])
    blocked_chart = set(heldout["ChartQA_TEST"])
    # Official validation/test charts are not part of this training pool.
    for split in ("val", "test"):
        for path in sorted((chart_root / "ChartQA Dataset" / split / "png").glob("*.png")):
            with Image.open(path) as image:
                blocked_chart.add(image_digest(image, chart=True))
    for row in rows:
        if row.get("image_digest"):
            blocked_raw.add(row["image_digest"])
            blocked_chart.add(row["image_digest"])
    validation_images = sorted({path for row in val for path in row["images"]})
    print(f"Hashing {len(validation_images)} distinct validation images.", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for raw_digest, chart_digest in executor.map(image_file_digests, validation_images):
            blocked_raw.add(raw_digest)
            blocked_chart.add(chart_digest)
    print(f"Selecting {count} HME100K and {count} ChartQA training samples.", flush=True)
    hme = hme_candidates(hme_root, count, rng, blocked_raw, stats)
    chart = chart_candidates(chart_root, count, rng, blocked_raw, blocked_chart, stats)
    output_dir.mkdir(parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    added = []
    for row in hme:
        uid = f"hme100k_train_{row['source_index']:06d}"
        image_path = image_dir / f"{uid}.png"
        with Image.open(io.BytesIO(row["image_bytes"])) as image:
            image.convert("RGB").save(image_path)
        value = {
            "images": [str(image_path)], "source_image": str(image_path),
            "question": "Transcribe the handwritten mathematical expression in the image into LaTeX. Return only the expression, without explanations or surrounding math delimiters.",
            "solution": row["solution"], "bbox": [], "source_index": row["source_index"],
            "data_source": "visual-agent-ocr", "ability": "text_reading", "split": "train",
            "uid": uid, "original_source": "hme100k", "image_digest": row["image_digest"],
            "answer_aliases": [row["solution"]], "transform": "",
        }
        added.append({field: value.get(field) for field in train.column_names})
    for row in chart:
        value = {
            "images": [row["path"]], "source_image": row["path"],
            "question": row["question"], "solution": row["solution"], "bbox": [],
            "source_index": row["source_index"], "data_source": "visual-agent-chartqa",
            "ability": "chart_reasoning", "split": "train", "image_digest": row["image_digest"],
            "uid": f"chartqa_train_{row['annotation_kind']}_{row['source_index']:06d}",
            "original_source": "chartqa", "answer_aliases": [row["solution"]], "transform": "",
        }
        added.append({field: value.get(field) for field in train.column_names})
    removed_indices = sorted(rng.sample(relations, replace_count))
    removed = [rows[index] for index in removed_indices]
    rng.shuffle(added)
    mixed = list(rows)
    for index, row in zip(removed_indices, added, strict=True):
        mixed[index] = row
    rng.shuffle(mixed)
    pq.write_table(pa.Table.from_pylist(mixed, schema=train.schema), output_dir / "train.parquet", compression="zstd")
    write_jsonl(output_dir / "train.jsonl", mixed)
    for filename in ("val.parquet", "val.jsonl"):
        shutil.copyfile(base_dir / filename, output_dir / filename)
    write_jsonl(output_dir / "removed_relations.jsonl", removed)
    write_jsonl(output_dir / "added_ocr_chart.jsonl", added)
    manifest = {
        "base_data_dir": str(base_dir), "experiment_mode": "new_experiment",
        "seed": seed, "requested_relation_replacement_ratio": ratio,
        "original_relation_rows": len(relations), "replaced_relation_rows": replace_count,
        "actual_relation_replacement_ratio": replace_count / len(relations),
        "added_hme100k_rows": count, "added_chartqa_rows": count,
        "train_rows": len(mixed), "validation_rows": len(val),
        "train_sources_before": dict(Counter(row["data_source"] for row in rows)),
        "train_sources_after": dict(Counter(row["data_source"] for row in mixed)),
        "validation_sources": dict(Counter(row["data_source"] for row in val)),
        "added_chartqa_annotation_kinds": dict(Counter(row["annotation_kind"] for row in chart)),
        "removed_relation_indices": removed_indices,
        "source_roots": {"hme100k": str(hme_root), "chartqa": str(chart_root), "benchmarks": str(eval_dir)},
        "exclusion_counts": dict(stats),
        "image_exclusion": "Decoded RGB pixels; chart rotations and flips included. Pixel-altering recompression, resized or cropped copies are not detected.",
        "split_policy": "Use only official source training splits. Preserve original validation files byte-for-byte.",
        "resume_policy": "Fresh data schedule; do not reuse the step80 continuation data.pt cursor.",
        "formula_scoring": "Rule: case-sensitive LaTeX token equality with insignificant whitespace ignored. Current shared reward uses a transcription-aware semantic judge when this rule fails.",
        "base_train_sha256": sha256(base_dir / "train.parquet"),
        "base_val_sha256": sha256(base_dir / "val.parquet"),
        "train_sha256": sha256(output_dir / "train.parquet"),
        "val_sha256": sha256(output_dir / "val.parquet"),
    }
    if (hme_root / "download_manifest.json").exists():
        manifest["hme_download_manifest_sha256"] = sha256(hme_root / "download_manifest.json")
    manifest["chart_annotation_sha256"] = {
        kind: sha256(chart_root / f"ChartQA Dataset/train/train_{kind}.json")
        for kind in ("human", "augmented")
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--hme-root", type=Path, required=True)
    parser.add_argument("--chart-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relation-replacement-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    result = prepare(args.base_data_dir, args.hme_root, args.chart_root, args.eval_dir,
                     args.output_dir, args.relation_replacement_ratio, args.seed)
    print(json.dumps({key: value for key, value in result.items() if key != "removed_relation_indices"}, indent=2))


if __name__ == "__main__":
    main()
