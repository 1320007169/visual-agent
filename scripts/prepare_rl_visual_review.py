#!/usr/bin/env python3
"""Prepare a small, image-disjoint tool-trajectory review batch, without a judge."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
FONT = Path("/opt/huawei/modelarts-dev/modelarts-sdk/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")
COLORS = ["#d12035", "#007b43", "#195ece", "#a322be", "#bd5700", "#008d91"]


def tools_from_row(row):
    result = []
    messages = row["messages"]
    for index in range(2, len(messages) - 1, 2):
        call = CALL.fullmatch(messages[index]["content"])
        response = RESPONSE.search(messages[index + 1]["content"])
        if not call or not response:
            raise ValueError("Unpaired tool call in candidate")
        result.append({"turn": len(result) + 1, **json.loads(call.group(1)),
                       "response": json.loads(response.group(1))})
    return result


def select_cases(entries, count, seed, anchors):
    eligible = {entry["line"]: entry for entry in entries if 1 <= len(entry["tools"]) <= 6}
    selected, seen_images = [], set()

    def add(entry):
        image = str(Path(entry["row"]["images"][0]).resolve())
        if image in seen_images or len(selected) >= count:
            return False
        selected.append(entry)
        seen_images.add(image)
        return True

    for line in anchors:
        if line not in eligible:
            raise ValueError(f"Anchor is not an eligible tool trajectory: {line}")
        add(eligible[line])

    groups = defaultdict(lambda: defaultdict(list))
    for entry in eligible.values():
        row = entry["row"]
        label = row["metadata"]["answer_assessment"]["ground_truth"]
        has_crop = any(tool["name"] == "crop_zoom" for tool in entry["tools"])
        stratum = (has_crop, bool(row["metadata"]["review_flags"]),
                   min((row["metadata"]["rollout_step"] - 1) // 55, 2))
        groups[label][stratum].append(entry)

    queues = {}
    for label, strata in groups.items():
        subqueues = []
        for key in sorted(strata):
            values = sorted(strata[key], key=lambda item: hashlib.sha256(
                f"{seed}:{item['line']}".encode()).hexdigest())
            subqueues.append(deque(values))
        interleaved = deque()
        while any(subqueues):
            for queue in subqueues:
                if queue:
                    interleaved.append(queue.popleft())
        queues[label] = interleaved
    while len(selected) < count and any(queues.values()):
        for label in sorted(queues):
            while queues[label]:
                if add(queues[label].popleft()):
                    break
            if len(selected) == count:
                break
    return selected


def font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.is_file() else ImageFont.load_default()


def panel(path, title, boxes=(), labels=()):
    canvas = Image.new("RGB", (740, 640), "white")
    draw = ImageDraw.Draw(canvas)
    title = title.encode("ascii", "backslashreplace").decode("ascii")
    lines = textwrap.wrap(title, width=67)[:5]
    for index, line in enumerate(lines):
        draw.text((12, 8 + index * 24), line, fill="black", font=font(18))
    with Image.open(path) as opened:
        image = ImageOps.contain(opened.convert("RGB"), (716, 475))
    x, y = (740 - image.width) // 2, 140 + (475 - image.height) // 2
    canvas.paste(image, (x, y))
    for index, box in enumerate(boxes):
        color = COLORS[index % len(COLORS)]
        rect = [x + box[0] * image.width / 1000, y + box[1] * image.height / 1000,
                x + box[2] * image.width / 1000, y + box[3] * image.height / 1000]
        draw.rectangle(rect, outline=color, width=3)
        draw.text((rect[0] + 3, max(y, rect[1] - 22)), f"#{index + 1}", fill=color, font=font(18))
    return canvas


def render_case(case_dir, entry):
    row, tools = entry["row"], entry["tools"]
    question = row["messages"][1]["content"].replace("<image>", "").strip()
    clean_path = case_dir / "original.jpg"
    original = panel(row["images"][0], question.split("\n")[0])
    original.save(clean_path, quality=95)
    panels = [panel(row["images"][0], "Original image 0 (unmarked)")]
    for tool in tools:
        args, response = tool["arguments"], tool["response"]
        target = args["target_image"]
        if tool["name"] == "grounding_detect":
            title = f"Turn {tool['turn']} image {target}: {args['query']}\nLabels: {response['labels']}"
            panels.append(panel(row["images"][target], title, response["boxes"], response["labels"]))
        else:
            crop = response["crop_zoom"]
            title = f"Turn {tool['turn']} crop image {target}: {args.get('label', '')}; requested red, actual green"
            panels.append(panel(row["images"][target], title, [args["bbox_2d"], crop["bbox_2d"]]))
            output_index = crop["target_image"]
            panels.append(panel(row["images"][output_index], f"Turn {tool['turn']} actual returned image {output_index}"))
    page_paths = []
    for offset in range(0, len(panels), 4):
        page = Image.new("RGB", (1480, 1280), "#e4e4e4")
        for index, rendered in enumerate(panels[offset:offset + 4]):
            page.paste(rendered, ((index % 2) * 740, (index // 2) * 640))
        page_path = case_dir / f"tools_{offset // 4 + 1:02d}.jpg"
        page.save(page_path, quality=95)
        page_paths.append(str(page_path))
    packet = {
        "case_id": case_dir.name, "candidate_line": entry["line"],
        "source_index": row["metadata"]["source_index"], "question": question,
        "images": row["images"], "tools": tools,
        "original_preview": str(clean_path), "tool_pages": page_paths,
        "instructions": "Inspect original/question before outcome.json. Inspect every tool page and actual crops. "
                        "Use original files for small details. A matching answer or label does not validate a box. "
                        "This is agent visual screening, not human ground truth or proof of tool necessity.",
    }
    (case_dir / "case.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    outcome = {"final": row["messages"][-1]["content"],
               "original_final": row["metadata"]["original_final"],
               "assessment": row["metadata"]["answer_assessment"]}
    (case_dir / "outcome.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--anchor-lines", type=int, nargs="*", default=[2, 5, 29, 86, 318])
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    raw_file = args.candidates.read_bytes()
    entries = []
    for line, raw in enumerate(raw_file.splitlines(keepends=True), 1):
        row = json.loads(raw)
        entries.append({"line": line, "row": row, "tools": tools_from_row(row),
                        "row_sha256": hashlib.sha256(raw).hexdigest()})
    selected = select_cases(entries, args.count, args.seed, args.anchor_lines)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    cases, assignments = [], defaultdict(list)
    labels, paths = Counter(), Counter()
    for index, entry in enumerate(selected):
        case_id = f"case_{entry['line']:05d}"
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True)
        packet = render_case(case_dir, entry)
        record = {"case_id": case_id, "candidate_line": entry["line"],
                  "row_sha256": entry["row_sha256"], "source_index": packet["source_index"],
                  "case_path": str(case_dir / "case.json"),
                  "outcome_path": str(case_dir / "outcome.json")}
        cases.append(record)
        assignments["abcd"[index % 4]].append(record)
        labels[entry["row"]["metadata"]["answer_assessment"]["ground_truth"]] += 1
        paths["crop" if any(tool["name"] == "crop_zoom" for tool in entry["tools"]) else "detect_only"] += 1
    manifest = {"candidate_file": str(args.candidates.resolve()),
                "candidate_file_sha256": hashlib.sha256(raw_file).hexdigest(),
                "seed": args.seed, "cases": cases,
                "sampling": "Purposive diagnostic batch: anchors plus label/tool-path/flag/step strata, unique original images, 1-6 calls. Not a population quality estimate.",
                "counts": {"cases": len(cases), "labels": dict(labels), "paths": dict(paths)}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, batch in assignments.items():
        (output / f"batch_{name}.json").write_text(json.dumps(batch, indent=2) + "\n")
    print(json.dumps({"output": str(output), **manifest["counts"]}))


if __name__ == "__main__":
    main()
