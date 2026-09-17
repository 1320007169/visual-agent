import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mixed_prepare", ROOT / "scripts/prepare_zwz_deepeyesv2_mixed_rl.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class MixedPreparationTest(unittest.TestCase):
    def test_images_metadata_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            zwz = root / "zwz"
            source.mkdir()
            zwz.mkdir()
            images = []
            for index in range(16):
                data = io.BytesIO()
                Image.new("RGB", (4, 4), (index, 20, 30)).save(data, format="PNG")
                images.append(data.getvalue())
            for split, index in (("train", 0), ("val", 1)):
                path = root / f"{split}.png"
                path.write_bytes(images[index])
                pq.write_table(pa.Table.from_pylist([{
                    "images": [str(path)], "question": "Where is A relative to B?",
                    "solution": "above", "bbox": [0., 0., 1., 1.],
                    "source_index": index, "source_image": str(path),
                    "data_source": "visual-agent-zwz-relation", "ability": "spatial_relation",
                }]), zwz / f"{split}.parquet")
            pq.write_table(pa.Table.from_pylist([
                {"images": [{"bytes": images[2]}]}
            ]), source / "vstar_test.parquet")
            pq.write_table(pa.Table.from_pylist([{
                "images": [{"bytes": images[index]}],
                "extra_info": {"question": "What color is the car?"},
                "reward_model": {"ground_truth": "The car is RED."},
            } for index in range(1, 16)]), source / "perception_all_1.parquet")
            runs = []
            for name in ("v1", "v2"):
                summary = prepare.prepare(source, zwz, root / name, {"attribute": 5})
                self.assertEqual(summary["train_rows"], 6)
                self.assertEqual(summary["val_rows"], 1)
                rows = sum((pq.read_table(root / name / f"{s}.parquet").to_pylist()
                            for s in ("train", "val")), [])
                added = [r for r in rows if r["data_source"] == "visual-agent-deepeyesv2"]
                self.assertEqual(len(added), 5)
                self.assertTrue(all(r["split"] == "train" for r in added))
                self.assertTrue(all(r["solution"] == "The car is RED." for r in added))
                self.assertTrue(all(Path(r["images"][0]).is_file() for r in rows))
                runs.append(sorted((r["source_index"], r["split"]) for r in added))
            self.assertEqual(runs[0], runs[1])
            with self.assertRaises(FileExistsError):
                prepare.prepare(source, zwz, root / "v1", {"attribute": 5})

    def test_red_box_and_python_questions_are_excluded(self):
        self.assertIsNone(prepare.category("What color is inside the red bounding box?"))
        self.assertIsNone(prepare.category("Use Python to count how many cars."))
        self.assertEqual(prepare.category("What material is the table made of?"), "attribute")
        self.assertIsNone(prepare.category(
            "What is behind the person?\nAnswer using a single word or phrase."
        ))
        self.assertEqual(prepare.category(
            "Which boats are most numerous?\nAnswer with the option's letter."
        ), "counting")
        self.assertEqual(prepare.category("What word is written on the sign?"), "text")
        self.assertIsNone(prepare.category(
            "Please answer the question and provide the correct option letter.\nWhich item melts?"
        ))


if __name__ == "__main__":
    unittest.main()
