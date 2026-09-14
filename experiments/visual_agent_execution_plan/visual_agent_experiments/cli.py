"""Command line interface for isolated Visual Agent experiment bookkeeping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .analysis import read_jsonl, summarize_utility, validate_prediction_records, write_json


CHECKPOINT_COLUMNS = [
    "run",
    "model_path",
    "revision",
    "model_variant",
    "agent_sft",
    "parent_checkpoint",
    "notes",
]
FAIRNESS_FIELDS = [
    "parent_checkpoint",
    "reference_model",
    "kl_coefficient",
    "freeze_scope",
    "train_manifest_sha256",
    "reward_baseline",
    "budget_accounting",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def command_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = output_dir / "checkpoint_registry.csv"
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_COLUMNS)
        writer.writeheader()
        for run in ("B0", "B1", "B2", "B3", "B4"):
            writer.writerow({"run": run})
    write_json(
        output_dir / "eval_config.json",
        {
            "benchmarks": ["VStarBench", "HRBench4K", "HRBench8K", "MMStar", "POPE", "CV-Bench"],
            "modes": {"native": {"tools_registered": False}, "agent": {"tools_registered": True, "allow_direct_answer": True}},
            "decode": {"seed": 0, "samples_per_condition": 4, "max_generated_tokens": 4096},
            "tool": {"detector": "freeze-before-run", "bbox_coordinate_system": "original-image"},
            "selection_policy": "Select checkpoints on dev only; report test only after configuration lock.",
        },
    )
    print(f"wrote {registry} and {output_dir / 'eval_config.json'}")
    return 0


def command_make_split(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    if not records:
        raise ValueError("input JSONL has no records")
    if args.train_per_10k < 0 or args.dev_per_10k < 0:
        raise ValueError("split sizes must be non-negative")
    if args.train_per_10k + args.dev_per_10k > 10_000:
        raise ValueError("train-per-10k and dev-per-10k must sum to at most 10000")
    seed = str(args.seed)
    dev_boundary = args.train_per_10k + args.dev_per_10k
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_number, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id") or record.get("uid") or "")
        source_group = str(record.get("source_group_id") or record.get("source_image_id") or "")
        if not sample_id or not source_group:
            raise ValueError(f"row {row_number}: require sample_id/uid and source_group_id/source_image_id")
        grouped[source_group].append(sample_id)
    manifest_records = []
    for source_group, sample_ids in sorted(grouped.items()):
        bucket = int(hashlib.sha256(f"{seed}:{source_group}".encode()).hexdigest(), 16) % 10_000
        split = "train" if bucket < args.train_per_10k else "dev" if bucket < dev_boundary else "test"
        for sample_id in sorted(sample_ids):
            manifest_records.append({"sample_id": sample_id, "source_group_id": source_group, "split": split})
    manifest = {"seed": args.seed, "records": manifest_records}
    write_json(args.output, manifest)
    counts = {split: sum(row["split"] == split for row in manifest_records) for split in ("train", "dev", "test")}
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def command_validate_log(args: argparse.Namespace) -> int:
    errors = validate_prediction_records(read_jsonl(args.input))
    if errors:
        raise ValueError("\n".join(errors))
    print("prediction log is valid")
    return 0


def command_summarize_utility(args: argparse.Namespace) -> int:
    summary = summarize_utility(read_jsonl(args.input), bootstrap_repeats=args.bootstrap_repeats, seed=args.seed)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_check_fairness(args: argparse.Namespace) -> int:
    agent = _load_json(args.agent_config)
    dual = _load_json(args.dual_config)
    differences = {field: {"agent": agent.get(field), "dual": dual.get(field)} for field in FAIRNESS_FIELDS if agent.get(field) != dual.get(field)}
    if differences:
        raise ValueError(json.dumps({"incomparable_configuration": differences}, ensure_ascii=False, indent=2))
    if agent.get("stream") != "agent" or dual.get("stream") != "dual":
        raise ValueError("configs must identify stream='agent' and stream='dual'")
    print("A/D fairness fields match")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="create checkpoint and evaluation templates")
    init.add_argument("--output-dir", required=True)
    init.set_defaults(func=command_init)
    split = subparsers.add_parser("make-split", help="freeze source-image grouped splits")
    split.add_argument("--input", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train-per-10k", type=int, default=8000)
    split.add_argument("--dev-per-10k", type=int, default=1000)
    split.set_defaults(func=command_make_split)
    validate = subparsers.add_parser("validate-log", help="validate JSONL prediction logs")
    validate.add_argument("--input", required=True)
    validate.set_defaults(func=command_validate_log)
    summary = subparsers.add_parser("summarize-utility", help="calculate paired ON/OFF utility")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--bootstrap-repeats", type=int, default=2000)
    summary.add_argument("--seed", type=int, default=0)
    summary.set_defaults(func=command_summarize_utility)
    fairness = subparsers.add_parser("check-fairness", help="check single-stream and dual-stream comparability")
    fairness.add_argument("--agent-config", required=True)
    fairness.add_argument("--dual-config", required=True)
    fairness.set_defaults(func=command_check_fairness)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
