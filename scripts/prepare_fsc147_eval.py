#!/usr/bin/env python3
"""Prepare the text-specified FSC147 test split from cached FSCD dot labels."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


IMAGE_URL = "https://huggingface.co/Hzzone/PseCo/resolve/main/data/fsc147/images_384_VarV2.zip"
IMAGE_SHA256 = "19beaf3cf1b934fb9627fce34467030556519bee47afa8bb8b49189e0f362cc3"


def build_rows(annotation_file, image_root):
    data = json.loads(Path(annotation_file).read_text())
    categories = {row["id"]: row["name"] for row in data["categories"]}
    images = {row["id"]: row for row in data["images"]}
    if not images or len(images) != len(data["images"]):
        raise ValueError("FSC147 image IDs must be nonempty and unique")
    counts = Counter()
    image_categories = defaultdict(set)
    annotation_ids = set()
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in images or annotation["id"] in annotation_ids:
            raise ValueError("Unknown image or duplicate FSC147 annotation ID")
        annotation_ids.add(annotation["id"])
        counts[image_id] += 1
        image_categories[image_id].add(annotation["category_id"])
    rows = []
    filenames = set()
    for image_id, image in images.items():
        filename = image["file_name"]
        if Path(filename).name != filename or filename in filenames:
            raise ValueError("FSC147 image filenames must be unique basenames")
        filenames.add(filename)
        if len(image_categories[image_id]) != 1:
            raise ValueError(f"Expected one counted category per image: {filename}")
        category = categories[next(iter(image_categories[image_id]))]
        rows.append({
            "index": image_id, "image_path": str((Path(image_root) / filename).resolve()),
            "question": f"Count every instance of {category} in the image. Respond with only the nonnegative integer count.",
            "answer": counts[image_id], "category": category, "split": "test",
        })
    return rows


def extract_images(archive, image_root, filenames):
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = {}
        for member in source.infolist():
            name = Path(member.filename).name
            if not member.is_dir() and name in filenames:
                if name in members:
                    raise ValueError(f"Duplicate image in FSC147 archive: {name}")
                members[name] = member
        missing = set(filenames) - members.keys()
        if missing:
            raise ValueError(f"FSC147 archive is missing {len(missing)} test images")
        for name, member in members.items():
            destination = image_root / name
            if destination.is_file() and destination.stat().st_size:
                continue
            with tempfile.NamedTemporaryFile(dir=image_root, delete=False) as target:
                temporary = Path(target.name)
                try:
                    with source.open(member) as image:
                        shutil.copyfileobj(image, target)
                    target.flush()
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)


def prepare(annotation_file, image_root, output, download_images=True, image_url=IMAGE_URL):
    rows = build_rows(annotation_file, image_root)
    filenames = {Path(row["image_path"]).name for row in rows}
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()
               or not Path(row["image_path"]).stat().st_size]
    if missing:
        if not download_images:
            raise FileNotFoundError(f"Missing {len(missing)} FSC147 test images; set FSC147_IMAGE_ROOT or allow download")
        image_root = Path(image_root)
        image_root.mkdir(parents=True, exist_ok=True)
        print(f"Downloading FSC147 image archive (~204 MB): {image_url}", flush=True)
        with tempfile.TemporaryDirectory(dir=image_root.parent) as temporary:
            archive = Path(temporary) / "images.zip"
            digest = hashlib.sha256()
            with urllib.request.urlopen(image_url, timeout=120) as response, archive.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    target.write(chunk)
            if digest.hexdigest() != IMAGE_SHA256:
                raise ValueError("FSC147 image archive SHA256 mismatch")
            extract_images(archive, image_root, filenames)
    from PIL import Image

    for row in rows:
        with Image.open(row["image_path"]) as image:
            image.verify()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as target:
        temporary = Path(target.name)
        try:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Prepared FSC147_TEST: {len(rows)} images; {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-images", choices=("0", "1"), default="1")
    parser.add_argument("--image-url", default=IMAGE_URL)
    args = parser.parse_args()
    prepare(args.annotation_file, args.image_root, args.output, args.download_images == "1", args.image_url)


if __name__ == "__main__":
    main()
