#!/usr/bin/env python3
"""Export unchanged rollout candidates approved by recorded visual reviews."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


QUALITY_STATUS = "agent_visual_reviewed_not_human_verified"
REVIEW_CHECKS = ("tool_grounding", "answer_supported", "query_quality")


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_bytes())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("manifest must contain a cases list")
    digest = manifest.get("candidate_file_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("manifest must contain candidate_file_sha256")
    seen = set()
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("manifest cases must be objects")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("manifest case_id must be a nonempty string")
        if case_id in seen:
            raise ValueError(f"duplicate manifest case_id: {case_id}")
        seen.add(case_id)
        if type(case.get("candidate_line")) is not int or case["candidate_line"] < 1:
            raise ValueError(f"invalid candidate_line for {case_id}")
        if type(case.get("source_index")) is not int:
            raise ValueError(f"invalid source_index for {case_id}")
        row_digest = case.get("row_sha256")
        if not isinstance(row_digest, str) or len(row_digest) != 64:
            raise ValueError(f"invalid row_sha256 for {case_id}")
    return manifest


def read_candidates(path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases_by_line: dict[int, list[dict[str, Any]]] = {}
    for case in manifest["cases"]:
        cases_by_line.setdefault(case["candidate_line"], []).append(case)
    digest = hashlib.sha256()
    candidates = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if line_number not in cases_by_line:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise ValueError(f"candidate line {line_number} must have metadata")
            row_digest = hashlib.sha256(raw).hexdigest()
            for case in cases_by_line[line_number]:
                case_id = case["case_id"]
                if row_digest != case["row_sha256"]:
                    raise ValueError(f"row hash mismatch for {case_id}")
                source_index = row["metadata"].get("source_index")
                if type(source_index) is not int or source_index != case["source_index"]:
                    raise ValueError(f"source_index mismatch for {case_id}")
                candidates[case_id] = row
    if digest.hexdigest() != manifest["candidate_file_sha256"]:
        raise ValueError("candidate file hash mismatch")
    missing = sorted(case["case_id"] for case in manifest["cases"] if case["case_id"] not in candidates)
    if missing:
        raise ValueError(f"candidate lines missing for cases: {', '.join(missing)}")
    return candidates


def read_decisions(paths: list[Path], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = {case["case_id"] for case in manifest["cases"]}
    decisions = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                decision = json.loads(raw)
                if not isinstance(decision, dict):
                    raise ValueError(f"decision must be an object: {path}:{line_number}")
                case_id = decision.get("case_id")
                if not isinstance(case_id, str) or case_id not in expected:
                    raise ValueError(f"unknown decision case_id: {case_id!r}")
                if case_id in decisions:
                    raise ValueError(f"duplicate decision for {case_id}")
                verdict = decision.get("verdict")
                if verdict not in ("keep", "reject", "review"):
                    raise ValueError(f"invalid verdict for {case_id}")
                for key in ("reason", "reviewer"):
                    if not isinstance(decision.get(key), str) or not decision[key].strip():
                        raise ValueError(f"missing {key} for {case_id}")
                for key in REVIEW_CHECKS:
                    if decision.get(key) not in ("pass", "fail", "uncertain"):
                        raise ValueError(f"invalid {key} for {case_id}")
                inspected = decision.get("inspected_images")
                if not isinstance(inspected, list) or any(
                    not isinstance(image, str) or not Path(image).is_absolute() for image in inspected
                ):
                    raise ValueError(f"inspected_images must be absolute paths for {case_id}")
                if verdict == "keep" and (
                    not inspected or any(decision[key] != "pass" for key in REVIEW_CHECKS)
                ):
                    raise ValueError(f"keep requires three passing checks and inspected images for {case_id}")
                decisions[case_id] = decision
    missing = sorted(expected - decisions.keys())
    if missing:
        raise ValueError(f"missing decisions for cases: {', '.join(missing)}")
    return decisions


def export_reviewed(
    candidates_path: Path, manifest_path: Path, decision_paths: list[Path], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    manifest = read_manifest(manifest_path)
    candidates = read_candidates(candidates_path, manifest)
    decisions = read_decisions(decision_paths, manifest)
    approved = []
    ordered_decisions = []
    counts = {"keep": 0, "reject": 0, "review": 0}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        decision = decisions[case_id]
        ordered_decisions.append(decision)
        counts[decision["verdict"]] += 1
        if decision["verdict"] != "keep":
            continue
        row = copy.deepcopy(candidates[case_id])
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ValueError(f"kept candidate has no messages: {case_id}")
        if not isinstance(row.get("images"), list) or not row["images"]:
            raise ValueError(f"kept candidate has no images: {case_id}")
        row["metadata"]["visual_review"] = copy.deepcopy(decision)
        row["metadata"]["quality_status"] = QUALITY_STATUS
        approved.append(row)

    summary = {
        "candidate_file": str(candidates_path.resolve()),
        "candidate_file_sha256": manifest["candidate_file_sha256"],
        "manifest_file": str(manifest_path.resolve()),
        "decision_files": [str(path.resolve()) for path in decision_paths],
        "reviewed_cases": len(manifest["cases"]),
        "verdict_counts": counts,
        "exported_sft_rows": len(approved),
        "quality_status": QUALITY_STATUS,
        "messages_and_images_preserved": True,
    }
    # Validate and serialize every row before creating any output files.
    sft_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in approved)
    decision_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_decisions)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (
        ("sft.jsonl", sft_text),
        ("review_decisions.jsonl", decision_text),
        ("summary.json", summary_text),
    ):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = export_reviewed(args.candidates, args.manifest, args.decisions, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
