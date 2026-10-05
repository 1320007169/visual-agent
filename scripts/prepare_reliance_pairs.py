#!/usr/bin/env python3
"""Replace part of an RL training set with factual/counterfactual tool-observation pairs.

Each pair repeats a training question with a fixed prefix: the first tool call
of an existing rollout, followed by either its real observation (factual) or a
faulted copy (counterfactual). The policy continues after the prefix, so the
prefix earns no credit and each branch forms its own GRPO group. Counterfactual
rows store their fault so that repeating the same call returns it again.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil

FAULTS_FILE = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/tool_faults.py"
_spec = importlib.util.spec_from_file_location("verl_tool_faults", FAULTS_FILE)
tool_faults = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_faults)

RELIANCE_FIELDS = ("reliance_prefix", "reliance_fault", "reliance_branch", "reliance_pair_id")


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


def question_key(data_source, source_image, question) -> tuple[str, str, str]:
    return str(data_source), str(source_image), str(question).strip()


def prefix_candidates(paths: list[Path], keys: set, stats: Counter) -> dict[tuple, list[dict]]:
    """First successful, faultable first-turn tool call of each rollout, grouped by question."""
    candidates = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                meta = record.get("source_metadata") or {}
                if meta.get("reliance_branch"):
                    stats["skipped_prefixed_rollout"] += 1
                    continue
                key = question_key(meta.get("data_source"), meta.get("source_image"), meta.get("question"))
                if key not in keys:
                    stats["unmatched_question"] += 1
                    continue
                calls = (record.get("rollout_trace") or {}).get("tool_calls") or []
                call = calls[0] if calls else {}
                if (call.get("model_turn") != 1 or call.get("status") != "success"
                        or call.get("tool") not in tool_faults.FAULT_TOOLS or not isinstance(call.get("arguments"), dict)):
                    stats["first_action_not_faultable"] += 1
                    continue
                try:
                    observation = json.loads(call.get("model_observation") or "")
                except json.JSONDecodeError:
                    observation = None
                if not isinstance(observation, dict):
                    stats["unreadable_observation"] += 1
                    continue
                candidates[key].append({"tool": call["tool"], "arguments": call["arguments"], "observation": observation})
    return candidates


def prefix_messages(tool: str, arguments: dict, observation: dict) -> str:
    return json.dumps([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]},
        {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(observation, ensure_ascii=False)},
    ], ensure_ascii=False)


def build_pair(row: dict, call: dict, rng: random.Random, pair_id: str) -> tuple[dict, dict] | None:
    faulty = tool_faults.inject_fault(call["tool"], call["observation"], rng, row.get("solution"))
    if faulty is None:
        return None
    fault = {"tool": call["tool"], "arguments": call["arguments"], "observation": json.dumps(faulty, ensure_ascii=False)}
    factual = {**row, "reliance_prefix": prefix_messages(call["tool"], call["arguments"], call["observation"]),
               "reliance_fault": "", "reliance_branch": "factual", "reliance_pair_id": pair_id}
    counterfactual = {**row, "reliance_prefix": prefix_messages(call["tool"], call["arguments"], faulty),
                      "reliance_fault": json.dumps(fault, ensure_ascii=False),
                      "reliance_branch": "counterfactual", "reliance_pair_id": pair_id}
    return factual, counterfactual


def select_pairs(rows_by_key: dict, candidates: dict, count: int, profile: dict | None,
                 rng: random.Random, stats: Counter) -> list[tuple[dict, dict]]:
    """Sample questions without replacement, weighted by the profile's fault probability."""
    ranked = []
    for key in sorted(candidates):
        call = rng.choice(candidates[key])
        weight = tool_faults.fault_probability(profile, key[0], call["tool"])
        if weight > 0:
            # Efraimidis-Spirakis keys give weighted sampling without replacement.
            ranked.append((rng.random() ** (1 / weight), key, call))
    ranked.sort(reverse=True)
    pairs = []
    for _, key, call in ranked:
        if len(pairs) == count:
            break
        pair = build_pair(rows_by_key[key], call, rng, f"pair_{len(pairs):06d}")
        if pair is None:
            stats["unfaultable_observation"] += 1
            continue
        pairs.append(pair)
    if len(pairs) < count:
        raise ValueError(f"Only {len(pairs)} pairs are available for the requested {count}")
    return pairs


def prepare(base_dir: Path, rollout_dirs: list[Path], output_dir: Path, steps: tuple[int, int],
            fraction: float = 0.2, profile_path: Path | None = None, seed: int = 20261005,
            factual_only: bool = False) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    train = pq.read_table(base_dir / "train.parquet")
    if set(RELIANCE_FIELDS) & set(train.column_names):
        raise ValueError("Base training data already contains reliance pairs")
    rows = [{**row, **{field: "" for field in RELIANCE_FIELDS}} for row in train.to_pylist()]
    rows_by_key = {}
    for row in rows:
        rows_by_key.setdefault(question_key(row["data_source"], row["source_image"], row["question"]), row)
    paths = sorted(
        (path for directory in rollout_dirs for path in directory.glob("*.jsonl")
         if path.stem.isdigit() and steps[0] <= int(path.stem) <= steps[1]),
        key=lambda path: int(path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"No rollout steps {steps[0]}-{steps[1]} under {rollout_dirs}")
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path else None
    rng = random.Random(seed)
    stats = Counter()
    candidates = prefix_candidates(paths, set(rows_by_key), stats)
    replaced_count = int(len(rows) * fraction) // 2 * 2
    if factual_only:
        # Ablation: the same number of replaced rows, each a factual prefix of a distinct question.
        pairs = select_pairs(rows_by_key, candidates, replaced_count, profile, rng, stats)
        added = [factual for factual, _ in pairs]
    else:
        pairs = select_pairs(rows_by_key, candidates, replaced_count // 2, profile, rng, stats)
        added = [row for pair in pairs for row in pair]
    replaced = sorted(rng.sample(range(len(rows)), len(added)))
    mixed = list(rows)
    for index, row in zip(replaced, added, strict=True):
        mixed[index] = row
    rng.shuffle(mixed)

    schema = train.schema
    for field in RELIANCE_FIELDS:
        schema = schema.append(pa.field(field, pa.string()))
    output_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(mixed, schema=schema), output_dir / "train.parquet", compression="zstd")
    write_jsonl(output_dir / "train.jsonl", mixed)
    write_jsonl(output_dir / "added_reliance_pairs.jsonl", added)
    for filename in ("val.parquet", "val.jsonl"):
        if (base_dir / filename).exists():
            shutil.copyfile(base_dir / filename, output_dir / filename)
    manifest = {
        "base_data_dir": str(base_dir), "seed": seed, "replaced_fraction": fraction, "factual_only": factual_only,
        "train_rows": len(mixed), "pairs": len(pairs), "replaced_rows": len(added),
        "replaced_indices_before_shuffle": replaced,
        "prefix_source_rollouts": [str(path) for path in paths],
        "allocation": "matched to fault profile" if profile else "uniform over eligible questions",
        "profile": str(profile_path) if profile_path else None,
        "questions_with_prefix": len(candidates),
        "pairs_by_source_and_tool": dict(Counter(
            f"{factual['data_source']}|{json.loads(factual['reliance_prefix'])[0]['tool_calls'][0]['function']['name']}"
            for factual, _ in pairs
        )),
        "train_sources_after": dict(Counter(row["data_source"] for row in mixed)),
        "skip_counts": dict(stats),
        "base_train_sha256": sha256(base_dir / "train.parquet"),
        "train_sha256": sha256(output_dir / "train.parquet"),
    }
    if profile_path:
        manifest["profile_sha256"] = sha256(profile_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, action="append", required=True)
    parser.add_argument("--steps", default="1-10", help="Inclusive rollout step range used as prefixes")
    parser.add_argument("--fraction", type=float, default=0.2, help="Share of training rows replaced by pair rows")
    parser.add_argument("--profile", type=Path, help="Fault profile from analyze_tool_reliance.py; uniform when omitted")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--factual-only", action="store_true", help="Ablation: replace rows with factual prefixes only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    low, high = (int(value) for value in args.steps.split("-"))
    manifest = prepare(args.base_data_dir.resolve(), [path.resolve() for path in args.rollout_dir],
                       args.output_dir.resolve(), (low, high), args.fraction, args.profile, args.seed,
                       args.factual_only)
    print(json.dumps({key: value for key, value in manifest.items() if key != "replaced_indices_before_shuffle"}, indent=2))


if __name__ == "__main__":
    main()
