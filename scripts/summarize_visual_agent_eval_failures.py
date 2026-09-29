#!/usr/bin/env python3
"""Report final API failures separately from benchmark scores."""

import argparse
import json
from pathlib import Path

import openpyxl


def summarize(work_dir):
    reports = []
    seen = set()
    for path in sorted(Path(work_dir).rglob("*.xlsx")):
        if path.resolve() in seen or "exact_matching" in path.stem or path.stem.endswith("_score"):
            continue
        seen.add(path.resolve())
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            headers = next(rows, ())
            if "prediction" not in headers:
                continue
            prediction_index = headers.index("prediction")
            index_column = headers.index("index") if "index" in headers else None
            total = empty = 0
            failed_indices = []
            for row in rows:
                total += 1
                prediction = str(row[prediction_index] or "")
                empty += not bool(prediction.strip())
                if "Failed to obtain answer" in prediction:
                    failed_indices.append(row[index_column] if index_column is not None else total - 1)
            reports.append({
                "prediction_file": str(path), "total": total,
                "api_failed_count": len(failed_indices),
                "api_failed_rate": len(failed_indices) / total if total else 0,
                "empty_count": empty, "failed_indices": failed_indices,
            })
        finally:
            workbook.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = summarize(args.work_dir)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        print(f"{Path(report['prediction_file']).stem}: API failed {report['api_failed_count']}/{report['total']}, empty {report['empty_count']}")


if __name__ == "__main__":
    main()
