#!/usr/bin/env python3
"""Compare identical OCRBench questions and fixed OCR-error cohorts across checkpoints."""

import argparse
import ast
import json
from pathlib import Path

HME = "Handwritten Mathematical Expression Recognition"
CONDITIONS = ("clean", "training", "ocr_confusable")


def normalize(text, category):
    text = str(text).strip().replace("\n", " ")
    return text.replace(" ", "") if category == HME else text.lower()


def matches(text, answers, category):
    return any(normalize(answer, category) in normalize(text, category) for answer in answers)


def summarize(roots):
    arms = {}
    for root in roots:
        for path in root.glob("*/*/dino_latest/VisualAgent-vllm/*/*_OCRBench.json"):
            label, condition = path.relative_to(root).parts[:2]
            if condition not in CONDITIONS:
                continue
            key = (label, condition)
            if key in arms:
                raise ValueError(f"Duplicate prediction set: {key}")
            records = {}
            rows = json.loads(path.read_text())
            for row in rows:
                index, category = int(row["index"]), row["category"]
                answers = ast.literal_eval(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
                prediction = str(row["prediction"])
                trace = json.loads(row.get("raw_response") or "{}")
                calls = trace.get("tool_calls", [])
                fault_index = next((i for i, c in enumerate(calls) if "fault" in c), None)
                fault_call = calls[fault_index] if fault_index is not None else None
                first_ocr = next((c for c in calls if c.get("name") == "ocr_read"), None)
                natural_error = False
                if first_ocr and category.endswith("Recognition"):
                    args = first_ocr.get("canonical_arguments", first_ocr["arguments"])
                    result = first_ocr.get("result", {})
                    if (args.get("target_image") == 0 and args.get("mode", "text") == "text"
                            and args.get("bbox_2d", [0, 0, 1000, 1000]) == [0, 0, 1000, 1000]
                            and isinstance(result, dict) and isinstance(result.get("text"), str)
                            and result.get("status") not in {"error", "failed"}):
                        natural_error = not matches(result["text"], answers, category)
                signature, load_bearing, copied = None, False, False
                if fault_call:
                    fault = fault_call["fault"]
                    original, injected = fault["original"].get("text", ""), fault["injected"].get("text", "")
                    arguments = {"mode": "text", "bbox_2d": [0, 0, 1000, 1000],
                                 **fault_call.get("canonical_arguments", fault_call["arguments"])}
                    signature = json.dumps([arguments, original, injected], sort_keys=True, ensure_ascii=False)
                    load_bearing = (category.endswith("Recognition") and matches(original, answers, category)
                                    and not matches(injected, answers, category))
                    answer_text = normalize(prediction, category)
                    copied = (bool(answer_text) and answer_text in normalize(injected, category)
                              and answer_text not in normalize(original, category))
                records[index] = {
                    "question": row["question"], "category": category, "answers": answers,
                    "prediction": prediction, "api_failed": "Failed to obtain answer via API" in prediction,
                    "correct": "Failed to obtain answer via API" not in prediction and matches(prediction, answers, category),
                    "natural_error": natural_error, "fault_signature": signature, "load_bearing": load_bearing,
                    "copied_fault_proxy": copied, "rechecked_after_fault": fault_index is not None and len(calls) > fault_index + 1,
                    "replay_calls": sum(bool(c.get("replayed_fault")) for c in calls),
                }
            if len(rows) != 1000 or set(records) != set(range(1000)):
                raise ValueError(f"Expected exactly the 1000 OCRBench indices for {key}")
            arms[key] = records
    if ("base", "clean") not in arms:
        raise ValueError("Include the base/clean results to fix the natural-error cohort")
    baseline = arms[("base", "clean")]
    labels = sorted({label for label, _ in arms})
    for label in labels:
        for condition in CONDITIONS:
            if (label, condition) not in arms:
                raise ValueError(f"Missing condition: {label}/{condition}")
    for key, records in arms.items():
        for i, row in records.items():
            if any(row[field] != baseline[i][field] for field in ["question", "category", "answers"]):
                raise ValueError(f"Unpaired sample {i} in {key}")
    cohorts = {
        "all": list(range(1000)),
        "recognition": [i for i, r in baseline.items() if r["category"].endswith("Recognition")],
        "hme": [i for i, r in baseline.items() if r["category"] == HME],
        "natural_ocr_error_fixed_from_base_clean": [i for i, r in baseline.items() if r["natural_error"]],
    }
    report = {"cohorts": cohorts, "arms": {}, "matched_faults": {}}
    for (label, condition), records in arms.items():
        report["arms"][f"{label}/{condition}"] = {}
        for name, indices in cohorts.items():
            selected = [records[i] for i in indices]
            report["arms"][f"{label}/{condition}"][name] = {
                "n": len(selected), "correct": sum(r["correct"] for r in selected),
                "accuracy": sum(r["correct"] for r in selected) / len(selected) if selected else None,
                "api_failed": sum(r["api_failed"] for r in selected),
                "fault_exposed": sum(r["fault_signature"] is not None for r in selected),
                "rechecked_after_fault": sum(r["rechecked_after_fault"] for r in selected),
                "copied_fault_proxy": sum(r["copied_fault_proxy"] for r in selected),
                "replay_calls": sum(r["replay_calls"] for r in selected),
            }
    for condition in CONDITIONS[1:]:
        indices = [i for i in cohorts["recognition"]
                   if all(arms[label, condition][i]["load_bearing"] for label in labels)
                   and len({arms[label, condition][i]["fault_signature"] for label in labels}) == 1]
        report["matched_faults"][condition] = {
            "indices": indices, "n": len(indices),
            "correct": {label: sum(arms[label, condition][i]["correct"] for i in indices) for label in labels},
            "clean_correct_on_same_questions": {label: sum(arms[label, "clean"][i]["correct"] for i in indices) for label in labels},
        }
    report["limitations"] = [
        "Faults are triggered by model tool calls; identical questions and seeds do not guarantee identical exposure.",
        "Matched faults require identical effective arguments, original text and injected text across all checkpoints.",
        "Natural OCR errors are a fixed base/clean cohort of full-image recognition calls under OCRBench answer matching; they are not manually verified semantic errors.",
        "Copying is a lexical proxy. Evaluate coverage and API failures alongside accuracy.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.result_dirs)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}; checkpoints={len(result['arms']) // 3}")
