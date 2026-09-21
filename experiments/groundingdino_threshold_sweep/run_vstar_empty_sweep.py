#!/usr/bin/env python3
"""Sweep GroundingDINO thresholds over empty VStarBench tool calls."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_BOX_THRESHOLDS = "0.35,0.30,0.25,0.20,0.15,0.10,0.05"
DEFAULT_TEXT_THRESHOLDS = "0.25,0.20,0.15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-thresholds", default=DEFAULT_BOX_THRESHOLDS)
    parser.add_argument("--text-thresholds", default=DEFAULT_TEXT_THRESHOLDS)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item) for item in value.split(",")}, reverse=True)
    if not thresholds or any(item < 0 or item > 1 for item in thresholds):
        raise ValueError(f"Invalid thresholds: {value}")
    return thresholds


def load_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_empty_detection(call: dict[str, Any]) -> bool:
    if call.get("name") != "grounding_detect":
        return False
    result = load_json_object(call.get("result"))
    if result.get("error") or result.get("error_code"):
        return False
    boxes = result.get("boxes")
    detections = result.get("detections")
    return not boxes and not detections


def extract_empty_calls(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw = load_json_object(row.get("raw_response"))
            tool_calls = raw.get("tool_calls") or []
            image_paths = ast.literal_eval(row.get("image_path_list", "[]"))
            if not image_paths:
                continue
            for call_ordinal, call in enumerate(tool_calls, 1):
                if not isinstance(call, dict) or not is_empty_detection(call):
                    continue
                arguments = load_json_object(call.get("arguments"))
                query = str(arguments.get("query", "")).strip()
                target_image = int(arguments.get("target_image", 0))
                if not query or target_image < 0 or target_image >= len(image_paths):
                    continue
                answer_key = str(row.get("answer", ""))
                answer_text = str(row.get(answer_key, ""))
                samples.append(
                    {
                        "sample_key": f"{row.get('index')}_{call_ordinal}_{slugify(query)}",
                        "source_line": line_number,
                        "index": row.get("index"),
                        "call_ordinal": call_ordinal,
                        "query": query,
                        "target_image": target_image,
                        "image_path": str(image_paths[target_image]),
                        "question": row.get("question", ""),
                        "answer": answer_key,
                        "answer_text": answer_text,
                        "prediction": row.get("prediction", ""),
                        "hit": row.get("hit"),
                        "category": row.get("category", ""),
                        "original_result": load_json_object(call.get("result")),
                    }
                )
    return samples


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value[:60] or "query"


def tensor_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return list(value)


def normalize_result(result: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    boxes = tensor_list(result.get("boxes"))
    scores = [float(score) for score in tensor_list(result.get("scores"))]
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels")
    labels = [str(label) for label in tensor_list(labels)]
    boxes_px = [[round(float(item), 2) for item in box] for box in boxes]
    boxes_relative = [
        [
            round(1000 * box[0] / width, 2),
            round(1000 * box[1] / height, 2),
            round(1000 * box[2] / width, 2),
            round(1000 * box[3] / height, 2),
        ]
        for box in boxes_px
    ]
    return {
        "count": len(boxes_px),
        "max_score": max(scores, default=None),
        "boxes_px": boxes_px,
        "boxes_relative": boxes_relative,
        "scores": scores,
        "labels": labels,
    }


def draw_panel(image: Image.Image, result: dict[str, Any], title: str) -> Image.Image:
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 250))
    font = ImageFont.load_default()
    for index, box in enumerate(result["boxes_px"]):
        score = result["scores"][index] if index < len(result["scores"]) else 0.0
        label = result["labels"][index] if index < len(result["labels"]) else "object"
        draw.rectangle(box, outline=(255, 40, 40), width=line_width)
        draw.text((box[0] + 3, box[1] + 3), f"{label} {score:.3f}", fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    rendered.thumbnail((480, 420), Image.Resampling.LANCZOS)
    header_height = 28
    panel = Image.new("RGB", (480, 448), "white")
    panel.paste(rendered, ((480 - rendered.width) // 2, header_height + (420 - rendered.height) // 2))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((8, 8), title, fill="black", font=font)
    return panel


def save_contact_sheet(
    image: Image.Image,
    sample: dict[str, Any],
    box_thresholds: list[float],
    fixed_text_threshold: float,
    grid: dict[tuple[float, float], dict[str, Any]],
    output_path: Path,
) -> None:
    panels = []
    for box_threshold in box_thresholds:
        result = grid[(box_threshold, fixed_text_threshold)]
        panels.append(
            draw_panel(
                image,
                result,
                f"box={box_threshold:.2f}, text={fixed_text_threshold:.2f}, n={result['count']}",
            )
        )
    columns = 2
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * 480, rows * 448 + 44), (235, 235, 235))
    header = ImageDraw.Draw(sheet)
    header.text((8, 8), f"index={sample['index']} query={sample['query']}", fill="black", font=ImageFont.load_default())
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 480, 44 + (index // columns) * 448))
    sheet.save(output_path, quality=92)


def render_review_html(summaries: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for item in summaries:
        cells = "".join(
            f"<td>{html.escape(str(item.get(key, '')))}</td>"
            for key in ("highest_box_at_text_025", "highest_box_at_text_020", "highest_box_at_text_015")
        )
        rows.append(
            "<section>"
            f"<h2>#{item['index']} &mdash; {html.escape(item['query'])}</h2>"
            f"<p><b>Question:</b> {html.escape(str(item['question']))}<br>"
            f"<b>Answer:</b> {html.escape(str(item['answer']))}. {html.escape(str(item['answer_text']))} &nbsp; "
            f"<b>Previous prediction:</b> {html.escape(str(item['prediction']))}</p>"
            "<table><tr><th>text=0.25</th><th>text=0.20</th><th>text=0.15</th></tr>"
            f"<tr>{cells}</tr></table>"
            f"<p><a href=\"{html.escape(item['image_path'])}\">original image</a></p>"
            f"<img loading=\"lazy\" src=\"{html.escape(item['contact_sheet'])}\">"
            "</section>"
        )
    output_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>GroundingDINO threshold sweep</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto}section{border-top:1px solid #aaa;padding:16px 0}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}</style>"
        "<h1>VStarBench empty GroundingDINO calls</h1>" + "".join(rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    box_thresholds = parse_thresholds(args.box_thresholds)
    text_thresholds = parse_thresholds(args.text_thresholds)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    contact_dir = args.output_dir / "contact_sheets"
    contact_dir.mkdir()

    samples = extract_empty_calls(args.result_jsonl)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No empty GroundingDINO calls found")
    for sample in samples:
        if not Path(sample["image_path"]).is_file():
            raise SystemExit(f"Missing source image: {sample['image_path']}")
    with (args.output_dir / "empty_calls_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Loading {args.model} on {args.device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, local_files_only=True)
    model.to(args.device).eval()
    postprocess = processor.post_process_grounded_object_detection
    threshold_argument = "box_threshold" if "box_threshold" in inspect.signature(postprocess).parameters else "threshold"

    details_path = args.output_dir / "threshold_details.jsonl"
    summaries: list[dict[str, Any]] = []
    with details_path.open("w", encoding="utf-8") as details:
        for sample_number, sample in enumerate(samples, 1):
            image = Image.open(sample["image_path"]).convert("RGB")
            prompt = sample["query"].strip().lower()
            if not prompt.endswith("."):
                prompt += "."
            inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)

            grid: dict[tuple[float, float], dict[str, Any]] = {}
            for text_threshold in text_thresholds:
                for box_threshold in box_thresholds:
                    kwargs = {
                        "input_ids": inputs.get("input_ids"),
                        threshold_argument: box_threshold,
                        "text_threshold": text_threshold,
                        "target_sizes": [image.size[::-1]],
                    }
                    result = normalize_result(postprocess(outputs, **kwargs)[0], *image.size)
                    grid[(box_threshold, text_threshold)] = result
                    details.write(
                        json.dumps(
                            {
                                "sample_key": sample["sample_key"],
                                "index": sample["index"],
                                "query": sample["query"],
                                "box_threshold": box_threshold,
                                "text_threshold": text_threshold,
                                **result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            fixed_text = 0.25 if 0.25 in text_thresholds else text_thresholds[0]
            contact_name = f"{sample['sample_key']}.jpg"
            save_contact_sheet(image, sample, box_thresholds, fixed_text, grid, contact_dir / contact_name)
            highest = {
                text_threshold: next(
                    (box_threshold for box_threshold in box_thresholds if grid[(box_threshold, text_threshold)]["count"] > 0),
                    None,
                )
                for text_threshold in text_thresholds
            }
            summary = {
                **{key: value for key, value in sample.items() if key != "original_result"},
                "default_count": grid.get((0.35, 0.25), {}).get("count"),
                "highest_box_at_text_025": highest.get(0.25),
                "highest_box_at_text_020": highest.get(0.20),
                "highest_box_at_text_015": highest.get(0.15),
                "permissive_count": grid[(box_thresholds[-1], text_thresholds[-1])]["count"],
                "permissive_max_score": grid[(box_thresholds[-1], text_thresholds[-1])]["max_score"],
                "contact_sheet": f"contact_sheets/{contact_name}",
            }
            summaries.append(summary)
            print(
                f"[{sample_number:02d}/{len(samples)}] index={sample['index']} query={sample['query']!r} "
                f"first@text0.25={highest.get(0.25)} permissive={summary['permissive_count']}",
                flush=True,
            )
            del outputs, inputs

    csv_fields = [
        "index", "call_ordinal", "query", "category", "question", "answer", "answer_text", "prediction", "hit",
        "default_count", "highest_box_at_text_025", "highest_box_at_text_020", "highest_box_at_text_015",
        "permissive_count", "permissive_max_score", "image_path", "contact_sheet",
    ]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_review_html(summaries, args.output_dir / "review.html")
    recovered = sum(item["highest_box_at_text_025"] is not None for item in summaries)
    print(f"Finished: recovered {recovered}/{len(summaries)} at text_threshold=0.25", flush=True)
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
