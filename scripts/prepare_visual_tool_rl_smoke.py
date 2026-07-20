#!/usr/bin/env python3
"""Build a tiny rule-reward visual-tool RL dataset from the local SFT smoke set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import datasets


SYSTEM_PROMPT = (
    "You are a visual agent. Answer the user's question from the supplied image. "
    "Use the available visual tools when counting, localization, segmentation, or "
    "closer inspection is useful. After tool results are returned, continue reasoning "
    "and put the final answer inside <answer>...</answer>."
)


def resolve_images(repo_root: Path, dataset: Path, sample: dict[str, Any]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in sample.get("images") or [sample.get("image")]:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        candidates = [
            path,
            dataset.parent / path,
            repo_root / path,
            repo_root / "Visual_Agent" / "training_trajectories_natural" / path,
        ]
        match = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"Could not resolve sample image: {raw_path}")
        resolved.append(match)
    if not resolved:
        raise ValueError(f"Sample {sample.get('uid')} has no images")
    return resolved


def load_samples(path: Path, count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= count:
                break
    if len(samples) < count:
        raise ValueError(f"Requested {count} samples but only found {len(samples)} in {path}")
    return samples


def make_row(
    repo_root: Path,
    dataset: Path,
    sample: dict[str, Any],
    *,
    split: str,
    index: int,
) -> dict[str, Any]:
    images = resolve_images(repo_root, dataset, sample)
    encoded_images = [
        {"bytes": image.read_bytes(), "path": image.name}
        for image in images
    ]
    question = str(sample["question"]).strip()
    answer = str(sample["answer"]).strip()
    placeholders = "\n".join("<image>" for _ in images)
    return {
        # vstar-test uses DeepEyesV2's rule-first <answer> exact-match reward.
        "data_source": "vstar-test",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{placeholders}\n{question}"},
        ],
        "images": encoded_images,
        "ability": "perception",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "uid": sample.get("uid"),
            "question": question,
            "answer": answer,
            "required_tools": list(sample.get("required_tools") or []),
        },
    }


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_list(rows)
    dataset.to_parquet(str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/visual_tool_sft_v0.smoke.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visual_tool_rl_smoke"),
    )
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else repo_root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    total = args.train_samples + args.val_samples
    if args.train_samples < 1 or args.val_samples < 1:
        raise ValueError("train-samples and val-samples must both be positive")

    samples = load_samples(input_path, total)
    train_rows = [
        make_row(repo_root, input_path, sample, split="train", index=index)
        for index, sample in enumerate(samples[: args.train_samples])
    ]
    val_rows = [
        make_row(repo_root, input_path, sample, split="val", index=index)
        for index, sample in enumerate(samples[args.train_samples :])
    ]
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    write_parquet(train_rows, train_path)
    write_parquet(val_rows, val_path)
    print(json.dumps({
        "input": str(input_path),
        "train": str(train_path),
        "train_samples": len(train_rows),
        "val": str(val_path),
        "val_samples": len(val_rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
