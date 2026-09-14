import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from visual_agent_experiments.cli import main


class CliTest(unittest.TestCase):
    def test_split_keeps_source_group_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '\n'.join(
                    [
                        json.dumps({"uid": "a1", "source_image_id": "same-image"}),
                        json.dumps({"uid": "a2", "source_image_id": "same-image"}),
                        json.dumps({"uid": "b1", "source_image_id": "other-image"}),
                    ]
                )
                + '\n',
                encoding="utf-8",
            )
            output_path = root / "manifest.json"
            with mock.patch("sys.argv", ["plan", "make-split", "--input", str(input_path), "--output", str(output_path)]):
                self.assertEqual(main(), 0)
            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            assignments = {row["sample_id"]: row["split"] for row in manifest["records"]}
            self.assertEqual(assignments["a1"], assignments["a2"])

    def test_split_uses_dev_size_after_train_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps({"uid": f"sample-{index}", "source_image_id": f"image-{index}"})
                    for index in range(200)
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "manifest.json"
            argv = [
                "plan",
                "make-split",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--train-per-10k",
                "8000",
                "--dev-per-10k",
                "2000",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            splits = {row["split"] for row in json.loads(output_path.read_text())["records"]}
            self.assertIn("dev", splits)
            self.assertNotIn("test", splits)

    def test_split_rejects_sizes_over_ten_thousand(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps({"uid": "sample", "source_image_id": "image"}) + "\n",
                encoding="utf-8",
            )
            argv = [
                "plan",
                "make-split",
                "--input",
                str(input_path),
                "--output",
                str(root / "manifest.json"),
                "--train-per-10k",
                "9000",
                "--dev-per-10k",
                "2000",
            ]
            with mock.patch("sys.argv", argv), self.assertRaisesRegex(SystemExit, "at most 10000"):
                main()
