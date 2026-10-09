#!/usr/bin/env python3
"""Append a fixed random sample of DrivingVQA and GRAID-BDD training candidates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def prepare(base_dir: Path, candidates_dir: Path, output_dir: Path,
            seed: int = 20261009, count: int = 3000) -> dict:
    base_dir, candidates_dir, output_dir = (p.resolve() for p in (base_dir, candidates_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    candidate_file = candidates_dir / "candidates_5000.jsonl"
    candidates = [json.loads(line) for line in candidate_file.read_text().splitlines()]
    if not 0 < count <= len(candidates):
        raise ValueError("Sample count must fit the candidate pool")
    selected = random.Random(seed).sample(list(enumerate(candidates)), count)
    train = pq.read_table(base_dir / "train.parquet")
    added = []
    for index, candidate in selected:
        if candidate["split"] != "train":
            raise ValueError(f"Candidate {index} is not from the official training split")
        image = candidates_dir / candidate["image_path"]
        if not image.is_file():
            raise FileNotFoundError(image)
        raw = candidate["raw"]
        if candidate["source"] == "DrivingVQA":
            questions = raw["questions"]
            questions = [questions] if isinstance(questions, str) else questions
            if raw["has_multiple_questions"] or len(questions) != 1 or len(raw["true_answers"]) != 1:
                raise ValueError(f"DrivingVQA candidate {index} must have one question and answer")
            answer = raw["true_answers"][0]
            if answer not in raw["possible_answers"]:
                raise ValueError(f"DrivingVQA candidate {index} has an unknown answer label")
            question = questions[0] + "\n" + "\n".join(
                f"{label}. {text}" for label, text in raw["possible_answers"].items()
            )
            source, ability = "visual-agent-drivingvqa", "multiple_choice"
            uid = f"drivingvqa_train_{candidate['source_id']}"
        elif candidate["source"] == "GRAID-BDD":
            question, answer = raw["question"], str(raw["answer"])
            source = "visual-agent-graid-bdd"
            ability = ("multiple_choice" if candidate["question_type"] in {"MultiChoiceHowMany", "IsObjectCentered"}
                       else "counting" if candidate["question_type"] == "HowMany" else "perception")
            uid = f"graid_bdd_{candidate['source_shard']}_{candidate['source_row']}"
        else:
            raise ValueError(f"Unknown candidate source: {candidate['source']}")
        question += ("\nAnswer with only the option letter inside <answer>...</answer>."
                     if ability == "multiple_choice" else "\nPut the final answer inside <answer>...</answer>.")
        value = {
            "images": [str(image)], "source_image": str(image), "question": question,
            "solution": answer, "bbox": [], "source_index": index, "data_source": source,
            "ability": ability, "split": "train", "uid": uid,
            "original_source": candidate["source"], "image_digest": candidate["image_sha256"],
            "answer_aliases": [answer], "transform": "",
        }
        added.append({field: value.get(field) for field in train.column_names})
    mixed = train.to_pylist() + added
    output_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(mixed, schema=train.schema), output_dir / "train.parquet", compression="zstd")
    for name, rows in (("train.jsonl", mixed), ("added_mme_candidates.jsonl", added),
                       ("selected_candidates.jsonl", [row for _, row in selected])):
        with (output_dir / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("val.parquet", "val.jsonl"):
        shutil.copyfile(base_dir / name, output_dir / name)
    manifest = {
        "base_data_dir": str(base_dir), "candidates_dir": str(candidates_dir), "seed": seed,
        "candidate_count": len(candidates), "added_rows": len(added), "train_rows": len(mixed),
        "validation_rows": pq.ParquetFile(output_dir / "val.parquet").metadata.num_rows,
        "selected_indices": [index for index, _ in selected],
        "added_sources": dict(Counter(row["data_source"] for row in added)),
        "added_question_types": dict(Counter(row["question_type"] for _, row in selected)),
        "sampling_policy": "Uniform random sample without replacement; no additional question filtering.",
        "input_policy": "Original images and questions/options only; no annotations, explanations or tool traces.",
        "split_policy": "Official train candidates only; preserve base validation files byte-for-byte.",
        "resume_policy": "Fresh training data schedule; do not reuse an old data.pt cursor.",
        "candidate_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
        "file_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                        for name in ("train.parquet", "val.parquet")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--count", type=int, default=3000)
    args = parser.parse_args()
    manifest = prepare(args.base_data_dir, args.candidates_dir, args.output_dir, args.seed, args.count)
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_indices"}, indent=2))
