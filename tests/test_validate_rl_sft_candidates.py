from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts/validate_rl_sft_candidates.py"
SPEC = importlib.util.spec_from_file_location("validate_rl_sft_candidates", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def tool_turn(name, arguments, response, suffix=""):
    return [
        {"role": "assistant", "content": f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'},
        {"role": "user", "content": f"<tool_response>{json.dumps(response)}</tool_response>{suffix}"},
    ]


class CandidateValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original.png"
        self.crop = self.root / "crop.png"
        Image.new("RGB", (32, 24), color="white").save(self.original)
        Image.new("RGB", (16, 12), color="red").save(self.crop)
        self.train = self.root / "train.parquet"
        self.val = self.root / "val.parquet"
        self.source = {
            "source_index": 55062,
            "source_image": "original.png",
            "images": [str(self.original)],
            "question": "Where is the cup relative to the plate?",
            "solution": "to the left of",
        }
        self.validation_source = {
            "source_index": 99000,
            "source_image": "validation.png",
            "images": [str(self.root / "validation.png")],
            "question": "Where is the plate?",
            "solution": "right of",
        }
        self.write_sources()
        detection = {
            "query": "cup", "boxes": [[0, 0, 300, 500]], "labels": ["cup"],
            "confidence": [0.9], "count": 1,
            "source": "groundingdino", "coordinate_space": "relative_0_1000",
        }
        crop_uri = "tool://images/1/crop.png"
        outputs = [{"target_image": 1, "path": crop_uri, "width": 16, "height": 12, "source": "crop_zoom"}]
        crop_response = {
            "source": "crop_zoom", "target_image": 0, "coordinate_space": "relative_0_1000",
            "crop_zoom": {
                "target_image": 1, "crop_path": crop_uri,
                "bbox_2d": [0, 0, 500, 500], "requested_bbox_2d": [0, 0, 500, 500],
                "coordinate_space": "relative_0_1000", "image_outputs": outputs,
            },
            "image_outputs": outputs,
        }
        self.row = {
            "messages": [
                {"role": "system", "content": "Answer the visual question."},
                {"role": "user", "content": f"<image>\n{self.source['question']}"},
                *tool_turn("grounding_detect", {"query": "cup", "target_image": 0}, detection),
                *tool_turn("crop_zoom", {"bbox_2d": [0, 0, 500, 500], "target_image": 0}, crop_response,
                           "\ncrop_zoom returned a crop: <image>"),
                *tool_turn("grounding_detect", {"query": "cup", "target_image": 1}, detection),
                {"role": "assistant", "content": "<answer>left of</answer>"},
            ],
            "images": [str(self.original), str(self.crop)],
            "metadata": {
                "source_index": 55062, "source_image": "original.png",
                "rollout_file": "40.jsonl", "rollout_line": 12, "rollout_step": 40,
                "review_flags": [], "original_final": "<answer>left of</answer>",
                "quality_status": "automatically_filtered_requires_visual_review",
            },
        }

    def write_sources(self):
        pq.write_table(pa.Table.from_pylist([self.source]), self.train)
        pq.write_table(pa.Table.from_pylist([self.validation_source]), self.val)

    def run_validation(self, *rows):
        path = self.root / "candidates.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return validator.validate(path, self.train, self.val)

    def assert_error(self, row, code):
        report = self.run_validation(row)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["errors"], {code: 1})

    def test_valid_crop_and_detection_on_returned_image(self):
        report = self.run_validation(self.row)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["counts"]["verified_crop_files"], 1)
        self.assertEqual(report["counts"]["invalid_rows"], 0)
        self.assertEqual(report["tool_counts"], {"crop_zoom": 1, "grounding_detect": 2})
        self.assertEqual(report["label_counts"], {"left of": 1})

    def test_direct_answer_is_valid_and_needs_no_crop(self):
        row = deepcopy(self.row)
        row["images"] = row["images"][:1]
        row["messages"] = row["messages"][:2] + row["messages"][-1:]
        report = self.run_validation(row)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["counts"]["direct_answer_rows"], 1)
        self.assertEqual(report["counts"]["verified_crop_files"], 0)

    def test_repeated_source_index_is_rejected(self):
        report = self.run_validation(self.row, self.row)
        self.assertEqual(report["errors"], {"duplicate_source_index": 1})
        self.assertEqual(report["counts"]["valid_rows"], 1)

    def test_wrong_source_image_question_and_answer_are_rejected(self):
        for kind, error in (("image", "original_image_mismatch"), ("question", "question_mismatch"),
                            ("answer", "ground_truth_mismatch"), ("noncanonical", "noncanonical_answer")):
            with self.subTest(kind=kind):
                row = deepcopy(self.row)
                if kind == "image":
                    row["images"][0] = str(self.crop)
                elif kind == "question":
                    row["messages"][1]["content"] = "<image>\nA different question"
                else:
                    value = "right of" if kind == "answer" else "on"
                    row["messages"][-1]["content"] = f"<answer>{value}</answer>"
                self.assert_error(row, error)

    def test_missing_or_future_target_image_is_rejected(self):
        for value in (None, 1, True):
            with self.subTest(value=value):
                row = deepcopy(self.row)
                arguments = {"query": "cup"}
                if value is not None:
                    arguments["target_image"] = value
                row["messages"][2]["content"] = f'<tool_call>{json.dumps({"name": "grounding_detect", "arguments": arguments})}</tool_call>'
                self.assert_error(row, "invalid_target_image")

    def test_crop_dimensions_and_matching_query_are_checked(self):
        row = deepcopy(self.row)
        row["messages"][5]["content"] = row["messages"][5]["content"].replace('"width": 16', '"width": 17')
        self.assert_error(row, "crop_dimensions_mismatch")
        row = deepcopy(self.row)
        row["messages"][3]["content"] = row["messages"][3]["content"].replace('"query": "cup"', '"query": "plate"')
        self.assert_error(row, "detection_query_mismatch")

    def test_role_and_placeholder_counts_are_checked(self):
        row = deepcopy(self.row)
        row["messages"][3]["role"] = "assistant"
        self.assert_error(row, "invalid_message_role")
        row = deepcopy(self.row)
        row["messages"][5]["content"] = row["messages"][5]["content"].replace("<image>", "")
        self.assert_error(row, "image_placeholder_count_mismatch")

    def test_validation_overlap_is_checked_by_id_source_name_and_path(self):
        for field, code in (("source_index", "validation_source_index_overlap"),
                            ("source_image", "validation_source_image_overlap"),
                            ("images", "validation_original_image_overlap")):
            with self.subTest(field=field):
                previous = self.validation_source[field]
                self.validation_source[field] = self.source[field]
                self.write_sources()
                self.assert_error(self.row, code)
                self.validation_source[field] = previous

    def test_missing_metadata_and_missing_image_are_rejected(self):
        row = deepcopy(self.row)
        del row["metadata"]["rollout_line"]
        self.assert_error(row, "missing_provenance")
        self.crop.unlink()
        self.assert_error(self.row, "missing_image")

    def test_corrupt_crop_is_rejected(self):
        self.crop.write_bytes(b"not an image")
        self.assert_error(self.row, "invalid_crop_image")

    def test_empty_input_is_not_a_success(self):
        report = self.run_validation()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["errors"], {"empty_input": 1})

    def test_path_cache_preserves_symlink_resolution_and_invalid_path_errors(self):
        link = self.root / "original-link.png"
        link.symlink_to(self.original)
        validator._resolved_image_path.cache_clear()
        self.addCleanup(validator._resolved_image_path.cache_clear)
        original_resolve = validator.Path.resolve
        with patch.object(validator.Path, "resolve", autospec=True, side_effect=original_resolve) as resolve:
            for _ in range(3):
                self.assertEqual(validator.image_path(str(link)), str(self.original))
            self.assertEqual(resolve.call_count, 1)
        with self.assertRaises(validator.InvalidCandidate) as error:
            validator.image_path([str(self.original)])
        self.assertEqual(error.exception.code, "invalid_image_path")


if __name__ == "__main__":
    unittest.main()
