#!/usr/bin/env python3
"""Prepare depth, counting, and OCR teacher-training QA from local source data."""

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import re
import sys

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_ocr_chart_rl import image_digest
from prepare_raw_depth_camera_rl import CAMERA_QUESTION, relative_box
from prepare_rl_ocr_chart_replacement import chart_candidates, hme_candidates, image_file_digests
from prepare_visual_agent_multitool_rl import tally_rows


SEED = 20261005
DISTANCE_QUESTION = re.compile(
    r"^Estimate the real-world distances between objects in this image\. "
    r"Which object is closer to the (?P<anchor>.+?) \[(?P<box_anchor>\d+, \d+, \d+, \d+)\], "
    r"the (?P<object_a>.+?) \[(?P<box_a>\d+, \d+, \d+, \d+)\] or "
    r"the (?P<object_b>.+?) \[(?P<box_b>\d+, \d+, \d+, \d+)\]\?$"
)
BENCHMARKS = (
    "VStarBench", "HRBench4K", "HRBench8K", "OCRBench", "MME-RealWorld-Lite",
    "MME-RealWorld-CN", "CV-Bench-2D", "CV-Bench-3D", "ChartQA_TEST",
)
SCHEMA = pa.schema([
    ("images", pa.list_(pa.string())), ("question", pa.string()),
    ("solution", pa.string()), ("bbox", pa.list_(pa.float64())),
    ("source_index", pa.int64()), ("source_image", pa.string()),
    ("data_source", pa.string()), ("ability", pa.string()), ("split", pa.string()),
    ("image_digest", pa.string()), ("uid", pa.string()),
    ("original_source", pa.string()), ("answer_aliases", pa.list_(pa.string())),
    ("count_complexity", pa.string()), ("transform", pa.string()),
])


def file_digest(path):
    with Image.open(path) as image:
        return image_digest(image)


def row(image, question, answer, uid, source, data_source, ability, index, **extra):
    return {
        "images": [str(image)], "question": question.strip(), "solution": str(answer).strip(),
        "bbox": [], "source_index": index, "source_image": str(image),
        "data_source": data_source, "ability": ability, "split": "train",
        "image_digest": "", "uid": uid, "original_source": source,
        "answer_aliases": [str(answer).strip()], "count_complexity": "", "transform": "",
        **extra,
    }


def split_for(digest):
    return "val" if int(hashlib.sha256(f"{SEED}:{digest}".encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def select(candidates, quotas, blocked, stats, fixed_split=False):
    selected = {"train": [], "val": []}
    counts = Counter()
    seen = set()
    answers = defaultdict(set)
    for item in candidates:
        key = (item["image_digest"] or item["images"][0], item["question"])
        answers[key].add(item["solution"])
    random.Random(SEED).shuffle(candidates)
    digests = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for start in range(0, len(candidates), 128):
            batch = candidates[start:start + 128]
            paths = list(dict.fromkeys(item["images"][0] for item in batch
                                       if not item["image_digest"] and item["images"][0] not in digests))
            digests.update(zip(paths, executor.map(file_digest, paths), strict=True))
            for item in batch:
                source_key = (item["image_digest"] or item["images"][0], item["question"])
                if len(answers[source_key]) > 1:
                    stats["conflicting_answer_exclusions"] += 1
                    continue
                digest = item["image_digest"] or digests[item["images"][0]]
                item["image_digest"] = digest
                if digest in blocked:
                    stats["heldout_image_exclusions"] += 1
                    continue
                key = (digest, item["question"])
                if key in seen:
                    stats["duplicate_qa_exclusions"] += 1
                    continue
                seen.add(key)
                split = item["split"] if fixed_split else split_for(digest)
                group = item["count_complexity"] or item["original_source"]
                if counts[split, group] >= quotas.get((split, group), 0):
                    continue
                item["split"] = split
                selected[split].append(item)
                counts[split, group] += 1
                if all(counts[key] == count for key, count in quotas.items()):
                    return selected
    if any(counts[key] != count for key, count in quotas.items()):
        raise ValueError(f"Insufficient eligible samples: {dict(counts)}; required {quotas}")
    return selected


def write_dataset(root, splits, stats, sources, tools):
    overlap = {r["image_digest"] for r in splits["train"]} & {r["image_digest"] for r in splits["val"]}
    if overlap:
        raise ValueError(f"Train/val image overlap: {len(overlap)}")
    manifests = {}
    for split, rows in splits.items():
        rows = [{name: item.get(name) for name in SCHEMA.names} for item in rows]
        for item in rows:
            item["answer_aliases"] = item["answer_aliases"] or [item["solution"]]
        rows.sort(key=lambda item: hashlib.sha256(f"{SEED}:{item['uid']}".encode()).digest())
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), root / f"{split}.parquet", compression="zstd")
        with (root / f"{split}.jsonl").open("w") as stream:
            for item in rows:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifests[split] = {
            "rows": len(rows), "unique_images": len({r["image_digest"] for r in rows}),
            "sources": dict(Counter(r["original_source"] for r in rows)),
            "data_sources": dict(Counter(r["data_source"] for r in rows)),
            "abilities": dict(Counter(r["ability"] for r in rows)),
            "groups": dict(Counter(r["count_complexity"] for r in rows if r["count_complexity"])),
            "sha256": hashlib.sha256((root / f"{split}.parquet").read_bytes()).hexdigest(),
        }
    manifest = {
        "seed": SEED, "splits": manifests, "exclusions": dict(stats),
        "source_files": [str(p) for p in sources], "teacher_tools": tools,
        "train_val_rgb_overlap": 0, "input_policy": "Original full image and clean question; no boxes, points, density maps, CoT or tool trajectories.",
        "format": "ZwzDeepEyesV2Dataset-compatible RL QA; these are not SFT teacher trajectories or OPD token probabilities.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(root.name, manifests, flush=True)


def prepare(args):
    repo = Path(__file__).resolve().parents[1]
    base = repo.parent
    min_root = base.parent
    output = args.output.resolve()
    if any((output / expert / "train.parquet").exists() for expert in ("depth", "count", "ocr")):
        raise FileExistsError(f"Training files already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for expert in ("depth", "count", "ocr"):
        (output / expert / "images").mkdir(parents=True, exist_ok=True)
    eval_root = base / "DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
    common_val = repo / "data/zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004/val.parquet"
    fsc = base / "datasets/fsc147"
    fsc_splits = json.loads((fsc / "Train_Test_Val_FSC_147.json").read_text())
    blocked = set()
    exclusion_counts = {}
    exclusion_path = output / "exclusions.json"
    if exclusion_path.exists():
        blocked = set(json.loads(exclusion_path.read_text())["blocked_rgb_sha256"])
        print("Using previously computed benchmark image exclusions.", flush=True)
    else:
        print("Hashing benchmark and existing validation images.", flush=True)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for name in BENCHMARKS:
                folder = eval_root / "images" / name
                paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
                if not paths:
                    raise FileNotFoundError(f"No benchmark images: {folder}")
                digests = set(executor.map(file_digest, paths))
                blocked.update(digests)
                exclusion_counts[name] = {"files": len(paths), "unique_rgb_images": len(digests)}
                print("Hashed", name, len(paths), flush=True)
            paths = [fsc / "images_384_VarV2" / name for name in fsc_splits["test"]]
            digests = set(executor.map(file_digest, paths))
            blocked.update(digests)
            exclusion_counts["FSC147_TEST"] = {"files": len(paths), "unique_rgb_images": len(digests)}
            paths = sorted({image for item in pq.read_table(common_val).to_pylist() for image in item["images"]})
            blocked.update(executor.map(file_digest, paths))
            exclusion_counts["existing_rl_validation"] = {"files": len(paths)}
        exclusion_path.write_text(json.dumps({
            "sources": exclusion_counts, "blocked_rgb_sha256": sorted(blocked),
            "method": "SHA256 of dimensions and exact decoded RGB pixels; does not detect recompressed, resized or cropped near-duplicates.",
        }, indent=2) + "\n")

    prepare_depth(output, base, min_root, blocked)
    prepare_count(output, base, blocked)
    prepare_ocr(output, base, min_root, blocked)


def prepare_depth(output, base, min_root, blocked):
    print("Extracting original CA-VQA camera-depth and object-distance QA.", flush=True)
    sys.path.insert(0, str(min_root / "groundingdino_offline_pipeline/scripts"))
    from prepare_vts_sources import decode_ca_vqa_record, read_tfrecord

    depth_root = base / "datasets/ca_vqa/train/cavqa_multichoice/1.0.0"
    candidates = []
    stats = Counter()
    record_index = 0
    shards = sorted(depth_root.glob("*.tfrecord-*"))
    for shard in shards:
        for record in read_tfrecord(shard):
            source = decode_ca_vqa_record(record)
            parsed = []
            objects = defaultdict(set)
            for index, (question, answer) in enumerate(zip(source["questions"], source["answers"], strict=True)):
                question = question.decode().splitlines()[0]
                match = CAMERA_QUESTION.fullmatch(question) or DISTANCE_QUESTION.fullmatch(question)
                if match and answer.decode() in {"A", "B"}:
                    parsed.append((index, match, answer.decode()))
                    objects[match["object_a"]].add(match["box_a"])
                    objects[match["object_b"]].add(match["box_b"])
                    if "anchor" in match.groupdict():
                        objects[match["anchor"]].add(match["box_anchor"])
            if parsed:
                raw = source["images"][-1]
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    digest = image_digest(image)
                if digest not in blocked:
                    scene = []
                    for index, match, answer in parsed:
                        a, b = match["object_a"], match["object_b"]
                        anchor = match.groupdict().get("anchor")
                        names = [a, b] + ([anchor] if anchor else [])
                        if len(set(names)) != len(names) or any(len(objects[name]) != 1 for name in names):
                            stats["ambiguous_object_exclusions"] += 1
                            continue
                        try:
                            relative_box(match["box_a"], width, height)
                            relative_box(match["box_b"], width, height)
                            if anchor:
                                relative_box(match["box_anchor"], width, height)
                        except ValueError as exc:
                            stats[str(exc)] += 1
                            continue
                        scene.append((index, a, b, anchor, answer))
                    random.Random(SEED + record_index).shuffle(scene)
                    camera = [item for item in scene if item[3] is None][:8]
                    distance = [item for item in scene if item[3] is not None][:8]
                    scene = camera + distance
                    if scene:
                        image = base / f"datasets/ca_vqa/extracted_images/multichoice/{record_index:06d}_{len(source['images']) - 1}.png"
                        if not image.is_file():
                            image = output / "depth/images" / f"cavqa_{record_index:06d}.png"
                            image.write_bytes(raw)
                        elif hashlib.sha256(image.read_bytes()).digest() != hashlib.sha256(raw).digest():
                            raise ValueError(f"CA-VQA extracted image mismatch: {image}")
                        for index, a, b, anchor, answer in scene:
                            ability = "object_distance_compare" if anchor else "camera_depth_compare"
                            question = (f"Estimate the real-world distances in image 0. Which object is closer to the {anchor}?" if anchor
                                        else "Which object is closer to the camera in image 0?")
                            candidates.append(row(image, f"{question}\nA. {a}\nB. {b}\nAnswer with A or B.", answer,
                                f"ca_vqa_multichoice_{record_index:06d}_{index:04d}", "ca_vqa_multichoice", "visual-agent-depth-raw", ability, index,
                                image_digest=digest, count_complexity=f"{ability}:{answer}"))
            record_index += 1
        print("Scanned", shard.name, "scenes", record_index, "depth candidates", len(candidates), flush=True)
    quotas = {(split, f"{ability}:{answer}"): count
              for split, count in (("train", 2000), ("val", 100))
              for ability in ("camera_depth_compare", "object_distance_compare") for answer in ("A", "B")}
    depth = select(candidates, quotas, blocked, stats)
    for rows in depth.values():
        for item in rows:
            item["count_complexity"] = ""
    write_dataset(output / "depth", depth, stats, shards, ["grounding_detect", "crop_zoom", "depth_measure"])

def prepare_count(output, base, blocked):
    fsc = base / "datasets/fsc147"
    fsc_splits = json.loads((fsc / "Train_Test_Val_FSC_147.json").read_text())
    print("Selecting official TallyQA and FSC147 counting QA.", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        fsc_val_digests = set(executor.map(file_digest, [fsc / "images_384_VarV2" / name for name in fsc_splits["val"]]))
    tally_root = base / "datasets/tallyqa"
    tally = tally_rows(tally_root / "manifest/tallyqa_train_10k.jsonl", tally_root / "raw/train.json")
    stats = Counter()
    # Existing TallyQA validation images stay outside every training set.
    tally_blocked = blocked | fsc_val_digests | {file_digest(item["images"][0]) for item in tally["val"]}
    count = select(tally["train"], {("train", "simple"): 2000, ("train", "complex"): 3000, ("val", "simple"): 80, ("val", "complex"): 120}, tally_blocked, stats)
    annotations = json.loads((fsc / "annotation_FSC147_384.json").read_text())
    classes = dict(line.split("\t", 1) for line in (fsc / "ImageClasses_FSC147.txt").read_text().splitlines() if line.strip())
    candidates = []
    paths = [fsc / "images_384_VarV2" / name for split in ("train", "val") for name in fsc_splits[split]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        fsc_digests = dict(zip(paths, executor.map(file_digest, paths), strict=True))
    train_blocked = fsc_val_digests | {item["image_digest"] for item in count["val"]}
    for split, names in (("train", fsc_splits["train"]), ("val", fsc_splits["val"])):
        for index, name in enumerate(names):
            image = fsc / "images_384_VarV2" / name
            digest = fsc_digests[image]
            if split == "train" and digest in train_blocked:
                stats["fsc_official_validation_image_exclusions"] += 1
                continue
            candidates.append(row(image, f"Count every instance of {classes[name]} in the image. Respond with only the nonnegative integer count.",
                len(annotations[name]["points"]), f"fsc147_{split}_{Path(name).stem}", "fsc147", "visual-agent-fsc147", "counting", index, split=split, image_digest=digest))
    fsc_selected = select(candidates, {("train", "fsc147"): 3000, ("val", "fsc147"): 200}, blocked, stats, fixed_split=True)
    for split in count:
        count[split].extend(fsc_selected[split])
    write_dataset(output / "count", count, stats, [tally_root / "raw/train.json", tally_root / "manifest/tallyqa_train_10k.jsonl", fsc / "annotation_FSC147_384.json", fsc / "Train_Test_Val_FSC_147.json"],
        ["grounding_detect", "crop_zoom", "object_count"])

def prepare_ocr(output, base, min_root, blocked):
    print("Selecting OCR QA and handwritten formulas.", flush=True)
    stats = Counter()
    candidates = []
    ocr_quotas = {}
    sources = []
    for name, quota in (("textvqa", 1500), ("docvqa", 1500), ("infographicsvqa", 700), ("sroie", 300)):
        path = min_root / f"Visual-CoT/metadata/{name}_cot_train.jsonl"
        sources.append(path)
        with path.open() as stream:
            for index, line in enumerate(stream):
                source = json.loads(line)
                if source["split"] != "train" or not source["question"].strip() or not source["answer"].strip():
                    continue
                if "<" in source["answer"] or ">" in source["answer"]:
                    stats["ocr_answer_protocol_exclusions"] += 1
                    continue
                image = min_root / f"Visual-CoT/cot_image_data/cot_image_data/{name}" / source["image"]
                candidates.append(row(image, source["question"], source["answer"], f"{name}_{index}", name, "visual-agent-ocr", "text_reading", index,
                    answer_aliases=list(dict.fromkeys([source["answer"], *source.get("possible_answers", [])]))))
        ocr_quotas["train", name] = quota
        ocr_quotas["val", name] = quota // 20
    ocr = select(candidates, ocr_quotas, blocked, stats)
    hme_root = base / "datasets/hme100k"
    hme = hme_candidates(hme_root, 4200, random.Random(SEED), blocked, stats)
    candidates = []
    for source in hme:
        digest = source["image_digest"]
        image = output / "ocr/images" / f"hme100k_{source['source_index']:06d}.png"
        candidates.append(row(image, "Transcribe the handwritten mathematical expression in the image into LaTeX. Respond with only the expression; use a single backslash for each LaTeX command.",
            source["solution"], f"hme100k_train_{source['source_index']}", "hme100k", "visual-agent-ocr", "formula_reading", source["source_index"], image_digest=digest))
    hme_selected = select(candidates, {("train", "hme100k"): 3000, ("val", "hme100k"): 150}, blocked, stats)
    raw_by_index = {r["source_index"]: r["image_bytes"] for r in hme}
    for split in ocr:
        for item in hme_selected[split]:
            Path(item["images"][0]).write_bytes(raw_by_index[item["source_index"]])
        ocr[split].extend(hme_selected[split])
    sources.extend(sorted((hme_root / "data").glob("train-*.parquet")))
    print("Selecting official ChartQA training QA for the OCR expert.", flush=True)
    chart_root = base / "datasets/chartqa"
    eval_root = base / "DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
    paths = list((eval_root / "images/ChartQA_TEST").glob("*"))
    for split in ("val", "test"):
        paths.extend((chart_root / f"ChartQA Dataset/{split}/png").glob("*.png"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        chart_blocked = {digest for _, digest in executor.map(image_file_digests, paths)}
    (output / "ocr/chart_exclusions.json").write_text(json.dumps(sorted(chart_blocked)) + "\n")
    charts = chart_candidates(chart_root, 1500, random.Random(SEED), blocked, chart_blocked, stats)
    candidates = [row(source["path"], source["question"], source["solution"],
                      f"chartqa_train_{source['annotation_kind']}_{source['source_index']}",
                      "chartqa", "visual-agent-chartqa", "chart_reasoning", source["source_index"],
                      image_digest=file_digest(source["path"])) for source in charts]
    chart_selected = select(candidates, {("train", "chartqa"): 1000, ("val", "chartqa"): 50}, blocked, stats)
    for split in ocr:
        ocr[split].extend(chart_selected[split])
    sources.extend(chart_root / f"ChartQA Dataset/train/train_{kind}.json" for kind in ("human", "augmented"))
    write_dataset(output / "ocr", ocr, stats, sources, ["grounding_detect", "crop_zoom", "ocr_read"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
