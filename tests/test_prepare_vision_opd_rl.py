import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_vision_opd_rl.py"
SPEC = importlib.util.spec_from_file_location("prepare_vision_opd_rl", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


class PrepareVisionOpdRlTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("images", "teacher_images", "original_images"):
            (self.root / directory).mkdir()
        self.input_path = self.root / "train.jsonl"
        self.output_dir = self.root / "rl"

    def make_row(self, index: int, **updates):
        names = {
            "images": f"images/boxed-{index}.png",
            "teacher_images": f"teacher_images/crop-{index}.png",
            "original_images": f"original_images/original-{index}.jpg",
        }
        for value in names.values():
            (self.root / value).write_bytes(b"image")
        row = {
            "images": [names["images"]],
            "teacher_images": [names["teacher_images"]],
            "original_images": [names["original_images"]],
            "bbox": [10, 20, 30, 40],
            "problem": "<image>\nIgnore everything except the red bounding box.\n\nA. red\nB. blue\nC. green\nD. black",
            "answer": "B",
            "extra_info": {
                "answer": "B",
                "question": (
                    "What color is the object?\n\nA. red\nB. blue\nC. green\nD. black\n\n"
                    "Answer with the option's letter from the given choices."
                ),
            },
        }
        row.update(updates)
        return row

    def write_rows(self, rows):
        self.input_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_conversion_uses_clean_question_and_unmarked_original(self):
        rows = [self.make_row(index) for index in range(20)]
        self.write_rows(rows)
        summary = converter.prepare_dataset(
            self.input_path, self.output_dir, val_percent=50, smoke_train=1, smoke_val=1
        )
        converted = (
            pq.read_table(self.output_dir / "train.parquet").to_pylist()
            + pq.read_table(self.output_dir / "val.parquet").to_pylist()
        )
        self.assertEqual(len(converted), 20)
        first = next(row for row in converted if row["source_index"] == 0)
        self.assertEqual(first["images"], [str((self.root / "original_images/original-0.jpg").resolve())])
        self.assertEqual(first["solution"], "B")
        self.assertEqual(first["answer_text"], "blue")
        self.assertEqual(first["options"], {"A": "red", "B": "blue", "C": "green", "D": "black"})
        self.assertNotIn("red bounding box", first["question"])
        self.assertEqual(first["data_source"], "visual-agent-vision-opd")
        self.assertEqual(summary["total_rows"], 20)
        self.assertFalse(summary["bbox_exposed_to_policy"])

    def test_split_is_stable_from_relative_source_image(self):
        rows = [self.make_row(index) for index in range(20)]
        self.write_rows(rows)
        converter.prepare_dataset(
            self.input_path, self.output_dir, val_percent=50, smoke_train=1, smoke_val=1
        )
        for split in ("train", "val"):
            for row in pq.read_table(self.output_dir / f"{split}.parquet").to_pylist():
                self.assertEqual(converter.image_split(row["source_image"], 50), split)

    def test_default_keeps_every_row_in_train_without_val_files(self):
        rows = [self.make_row(index) for index in range(4)]
        self.write_rows(rows)
        summary = converter.prepare_dataset(
            self.input_path, self.output_dir, smoke_train=1, smoke_val=1
        )
        train = pq.read_table(self.output_dir / "train.parquet").to_pylist()
        self.assertEqual(len(train), 4)
        self.assertEqual(summary["train_rows"], 4)
        self.assertEqual(summary["val_rows"], 0)
        self.assertEqual(summary["smoke_val_rows"], 0)
        self.assertFalse((self.output_dir / "val.parquet").exists())
        self.assertFalse((self.output_dir / "smoke_val.parquet").exists())

    def test_rejects_bounding_box_leak_in_clean_question(self):
        row = self.make_row(0)
        row["extra_info"]["question"] = "Only focus on the red bounding box.\nA. x\nB. y\nC. z\nD. q"
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, "leaks image or bounding-box markup"):
            converter.load_and_validate(self.input_path)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_answer_mismatch_and_invalid_bbox(self):
        row = self.make_row(0)
        row["extra_info"]["answer"] = "A"
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, "answer disagrees"):
            converter.load_and_validate(self.input_path)
        row = self.make_row(1, bbox=[10, 20, 10, 40])
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, "bbox must be valid"):
            converter.load_and_validate(self.input_path)

    def test_rejects_missing_original_image(self):
        row = self.make_row(0)
        (self.root / row["original_images"][0]).unlink()
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, "missing original image"):
            converter.load_and_validate(self.input_path)

    def test_existing_output_is_not_modified(self):
        self.write_rows([self.make_row(index) for index in range(20)])
        self.output_dir.mkdir()
        sentinel = self.output_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            converter.prepare_dataset(
                self.input_path, self.output_dir, val_percent=50, smoke_train=1, smoke_val=1
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
