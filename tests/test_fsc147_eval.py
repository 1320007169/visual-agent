import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_fsc147_eval", ROOT / "scripts/prepare_fsc147_eval.py")
metrics = load_module("fsc147_metrics", ROOT / "evaluation/VLMEvalKit/vlmeval/dataset/utils/fsc147.py")


class FSC147EvalTests(unittest.TestCase):
    def test_numeric_extraction_handles_dense_counts_and_rejects_ambiguous_answers(self):
        for prediction, expected in [("147", 147), ("<answer>1,234</answer>", 1234),
                                     ("42.0", 42), ("0", 0)]:
            self.assertEqual(metrics.extract_count(prediction), expected)
        for prediction in ("There are 10 or 20", "-1", "3.5", "nan", "Failed to obtain answer via API."):
            self.assertIsNone(metrics.extract_count(prediction))

    def test_counting_metrics_use_absolute_and_squared_errors(self):
        result, counts = metrics.counting_metrics([10, 20], ["12", "16"])
        self.assertEqual(counts, [12, 16])
        self.assertEqual(result["MAE"], 3)
        self.assertAlmostEqual(result["RMSE"], math.sqrt(10))
        self.assertEqual(result["valid_rate"], 100)

    def test_failed_predictions_do_not_improve_full_split_mae(self):
        result, _ = metrics.counting_metrics([10, 20], ["10", "Failed to obtain answer via API."])
        self.assertTrue(math.isnan(result["MAE"]))
        self.assertTrue(math.isnan(result["RMSE"]))
        self.assertEqual(result["MAE_valid"], 0)
        self.assertEqual(result["exact_accuracy"], 50)
        self.assertEqual(result["valid_rate"], 50)
        self.assertEqual(result["invalid_count"], 1)
        self.assertEqual(result["api_failed_count"], 1)

    def annotation(self, root):
        path = root / "annotations.json"
        (root / "points.json").write_text(json.dumps({
            "one.jpg": {"points": [[10, 10], [20, 20]]},
            "two.jpg": {"points": [[10, 10]]},
        }))
        path.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "one.jpg"}, {"id": 2, "file_name": "two.jpg"}],
            "categories": [{"id": 0, "name": "apple"}],
            "annotations": [{"id": i, "image_id": image, "category_id": 0}
                            for i, image in enumerate([1, 1, 2, 1, 1, 1, 2, 2, 2])],
        }))
        return path

    def test_rows_count_each_dot_and_do_not_expose_answer_in_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = prepare.build_rows(self.annotation(root), root / "images", root / "points.json")
            self.assertEqual([row["answer"] for row in rows], [2, 1])
            self.assertIn("apple", rows[0]["question"])
            self.assertNotIn("2", rows[0]["question"])
            self.assertEqual(rows[0]["image_path"], str(root / "images/one.jpg"))

    def test_missing_images_fail_without_network_when_download_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "Missing 2 FSC147 test images"):
                prepare.prepare(self.annotation(root), root / "images", root / "FSC147_TEST.tsv",
                                root / "points.json", False)
            self.assertFalse((root / "FSC147_TEST.tsv").exists())

    def test_archive_extracts_only_requested_images_to_the_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("images_384_VarV2/one.jpg", b"one")
                target.writestr("images_384_VarV2/two.jpg", b"two")
                target.writestr("../unrelated.txt", b"unrelated")
            prepare.extract_images(archive, root / "images", {"one.jpg"})
            self.assertEqual((root / "images/one.jpg").read_bytes(), b"one")
            self.assertFalse((root / "images/two.jpg").exists())
            self.assertFalse((root / "unrelated.txt").exists())


if __name__ == "__main__":
    unittest.main()
