#!/usr/bin/env python3
"""Create an auditable real-world-perception subset of Kwai-Keye/Thyme-RL.

The raw Thyme-RL parquet files have no task-category field. This tool therefore
uses deliberately conservative text rules: it rejects questions that explicitly
ask for arithmetic, symbolic maths, charts/tables/diagrams, code execution, or
logic puzzles. It keeps all other image questions unchanged.

Default mode is dry-run. Pass --write to materialize a separate parquet snapshot.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "math_reasoning",
        re.compile(
            r"(?:^\s*find\b|\b[A-Z]\s+[A-Z]\s*=|"
            r"\b(?:find|calculate|compute|what is)\s+(?:the\s+)?(?:area|perimeter|volume)\b|"
            r"\\(?:frac|sqrt|angle|triangle|overline)\b|"
            r"\bvalue of\s+[xyz]\b|"
            r"\b(?:mean|median|mode|average|sum|difference|product|quotient|"
            r"calculate|calculation|compute|equation|fraction|ratio|"
            r"percentage|percent(?:age)?|probability|"
            r"algebra|geometry|hypotenuse|parallelogram|quadrilateral|"
            r"polygon|hexagon|theorem|congruent|similar triangles|coordinate plane|"
            r"intercept|radius|diameter|tangent|round to|nearest tenth|"
            r"multiply|divide|subtract|at least|at most)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "chart_or_diagram",
        re.compile(
            r"(?:\b(?:according to|following|data)\s+(?:the\s+)?table\b|"
            r"\b(?:bar chart|pie chart|line graph|scatter plot|chart|graph|"
            r"spreadsheet|schedule|diagram|flowchart|histogram|coordinate|x-axis|"
            r"y-axis)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "code_or_symbolic_task",
        re.compile(
            r"\b(?:write (?:a )?code|program(?:ming)?|python|javascript|sql|"
            r"algorithm|terminal|compile|execute (?:the )?code|write (?:a )?script)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "logic_puzzle",
        re.compile(
            r"\b(?:logic puzzle|riddle|number sequence|next number|sudoku|"
            r"crossword|deduce|deduction|which statement)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify(question: object) -> tuple[bool, str]:
    """Return (keep, reason) for one Thyme question."""
    text = str(question or "").strip()
    if not text:
        return False, "empty_question"
    for reason, pattern in RULES:
        if pattern.search(text):
            return False, reason
    return True, "real_world_perception_candidate"


def add_sample(
    samples: dict[str, list[dict[str, str]]],
    reason: str,
    question: object,
    solution: object,
    limit: int,
    rng: random.Random,
) -> None:
    bucket = samples[reason]
    item = {"question": str(question), "solution": str(solution)}
    if len(bucket) < limit:
        bucket.append(item)
        return
    # Reservoir sampling keeps the audit examples representative without storing
    # all questions in memory.
    index = rng.randrange(len(bucket) + 1)
    if index < limit:
        bucket[index] = item


def scan_questions(
    files: Iterable[Path], sample_limit: int, rng: random.Random
) -> tuple[Counter[str], dict[str, list[dict[str, str]]]]:
    counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in files:
        table = pq.read_table(path, columns=["question", "solution"])
        for row in table.to_pylist():
            keep, reason = classify(row.get("question"))
            counts["kept" if keep else "rejected"] += 1
            counts[f"reason:{reason}"] += 1
            add_sample(samples, reason, row.get("question"), row.get("solution"), sample_limit, rng)
    return counts, samples


def write_filtered_snapshot(
    files: Iterable[Path],
    output_dir: Path,
    sample_limit: int,
    rng: random.Random,
) -> tuple[Counter[str], dict[str, list[dict[str, str]]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for source_path in files:
        target_path = output_dir / source_path.name
        writer: pq.ParquetWriter | None = None
        try:
            parquet_file = pq.ParquetFile(source_path)
            for batch in parquet_file.iter_batches(batch_size=64 * 1024):
                table = pa.Table.from_batches([batch])
                questions = table.column("question").to_pylist()
                solutions = table.column("solution").to_pylist()
                keep_mask: list[bool] = []
                for question, solution in zip(questions, solutions):
                    keep, reason = classify(question)
                    keep_mask.append(keep)
                    counts["kept" if keep else "rejected"] += 1
                    counts[f"reason:{reason}"] += 1
                    add_sample(samples, reason, question, solution, sample_limit, rng)

                filtered = table.filter(pa.array(keep_mask))
                if writer is None:
                    writer = pq.ParquetWriter(target_path, table.schema, compression="zstd")
                if filtered.num_rows:
                    writer.write_table(filtered)
            if writer is None:
                raise RuntimeError(f"empty parquet file: {source_path}")
        finally:
            if writer is not None:
                writer.close()

    return counts, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/thyme_rl_snapshot/data"),
        help="Directory containing the original Thyme train-*.parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/thyme_rl_realworld_perception/data"),
        help="Destination for filtered parquet shards; only used with --write.",
    )
    parser.add_argument("--write", action="store_true", help="Materialize filtered shards instead of only reporting.")
    parser.add_argument("--max-shards", type=int, default=None, help="Process only the first N shards for a quick audit.")
    parser.add_argument("--sample-limit", type=int, default=20, help="Audit examples retained per decision reason.")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("train-*.parquet"))
    if args.max_shards is not None:
        files = files[: args.max_shards]
    if not files:
        raise SystemExit(f"no train-*.parquet files found under {args.input_dir}")

    rng = random.Random(args.seed)
    if args.write:
        counts, samples = write_filtered_snapshot(files, args.output_dir, args.sample_limit, rng)
        summary_path = args.output_dir.parent / "filter_summary.json"
        samples_path = args.output_dir.parent / "filter_audit_samples.json"
        summary = {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "shards": len(files),
            "counts": dict(counts),
            "rules": {reason: pattern.pattern for reason, pattern in RULES},
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        samples_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote filtered shards: {args.output_dir}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote audit samples: {samples_path}")
    else:
        counts, samples = scan_questions(files, args.sample_limit, rng)

    print(json.dumps({"shards": len(files), "counts": dict(counts), "samples": samples}, indent=2))


if __name__ == "__main__":
    main()

