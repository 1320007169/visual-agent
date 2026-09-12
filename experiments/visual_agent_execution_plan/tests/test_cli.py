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
