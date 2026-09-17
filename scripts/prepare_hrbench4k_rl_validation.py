#!/usr/bin/env python3
"""Convert local HRBench4K to the mixed RL adapter's validation schema."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def prepare(tsv, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    csv.field_size_limit(sys.maxsize)
    rows, references = [], {}
    with tsv.open() as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            encoded = item["image"]
            if len(encoded) < 64:
                path = references[encoded]
            else:
                raw = base64.b64decode(encoded, validate=True)
                path = images / (hashlib.sha256(raw).hexdigest() + ".jpg")
                if not path.exists():
                    path.write_bytes(raw)
            references[item["index"]] = path
            answer = item["answer"].strip().upper()
            if answer not in "ABCD" or len(answer) != 1:
                raise ValueError(f"Invalid HRBench answer: {answer}")
            question = item["question"] + "\n" + "\n".join(
                f"{letter}. {item[letter]}" for letter in "ABCD"
            ) + "\nAnswer with only the option letter inside <answer>...</answer>."
            rows.append({
                "images": [str(path.resolve())], "question": question,
                "solution": answer, "bbox": [], "source_index": int(item["index"]),
                "source_image": item["index"], "data_source": "visual-agent-hrbench4k",
                "ability": item["category"], "cycle_category": item["cycle_category"],
                "split": "val",
            })
    cycles = Counter(r["cycle_category"] for r in rows)
    if len(set(cycles.values())) != 1:
        raise ValueError("Unequal HRBench cycle sizes: row accuracy would differ from cycle mean")
    pq.write_table(pa.Table.from_pylist(rows), output / "val.parquet")
    summary = {"rows": len(rows), "cycles": dict(cycles), "source": str(tsv.resolve())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.tsv, args.output)
