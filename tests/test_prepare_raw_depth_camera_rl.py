import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_raw_depth_camera_rl.py"
SPEC = importlib.util.spec_from_file_location("prepare_raw_depth_camera_rl", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class CameraDepthPreparationTest(unittest.TestCase):
    def test_reference_image_and_box_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "reference.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 1000, 500))
            row = {
                "images": ["support.png", str(image)],
                "question": (
                    "Which object is closer to the camera taking this photo, the "
                    "chair [100, 50, 500, 250] or the table [500, 100, 900, 400]?\n"
                    "A. chair\nB. table"
                ),
                "solution": "B", "uid": "one", "source": "ca_vqa_multichoice",
            }
            item = prepare.curate(row)
            self.assertEqual(item["images"], [str(image)])
            self.assertEqual(item["bbox_a"], [100, 100, 500, 500])
            self.assertEqual(item["bbox_b"], [500, 200, 900, 800])
            self.assertIn("image 0", item["question"])
            self.assertNotIn("[100, 100, 500, 500]", item["question"])

    def test_scene_split_and_small_box_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = []
            for index in range(40):
                image = root / f"{index}.png"
                image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 1000, 1000))
                for suffix in ("a", "b"):
                    rows.append({
                        "images": [str(image)],
                        "question": "Which object is closer to the camera taking this photo, the chair [10, 10, 110, 110] or the table [200, 200, 400, 400]?",
                        "solution": "A", "uid": f"{index}-{suffix}", "source": "ca_vqa_multichoice",
                    })
            rows.append({**rows[0], "uid": "tiny", "question": "Which object is closer to the camera taking this photo, the chair [10, 10, 12, 12] or the table [200, 200, 400, 400]?"})
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            output = root / "curated"
            summary = prepare.prepare(source, output, val_percent=20)
            self.assertEqual(summary["train"] + summary["val"], 80)
            self.assertEqual(summary["excluded"]["small_box"], 1)
            train = pq.read_table(output / "train.parquet").to_pylist()
            val = pq.read_table(output / "val.parquet").to_pylist()
            self.assertFalse({r["reference_image"] for r in train} & {r["reference_image"] for r in val})


if __name__ == "__main__":
    unittest.main()
