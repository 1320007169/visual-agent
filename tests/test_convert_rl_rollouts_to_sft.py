import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


PATH = Path(__file__).parents[1] / "scripts/convert_rl_rollouts_to_sft.py"
SPEC = importlib.util.spec_from_file_location("rollout_sft_converter_test", PATH)
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def source(index=7, solution="below"):
    return {
        "source_index": index, "source_image": f"{index}.png",
        "question": "Where is the cup relative to the plate?",
        "solution": solution, "images": [f"/{index}.png"],
    }


def detection(query="cup", boxes=None, target=0):
    boxes = [[100, 300, 200, 400]] if boxes is None else boxes
    arguments = {"query": query}
    if target is not None:
        arguments["target_image"] = target
    return (
        {"name": "grounding_detect", "arguments": arguments},
        {"query": query, "boxes": boxes, "labels": [query] * len(boxes),
         "confidence": [0.8] * len(boxes), "count": len(boxes),
         "coordinate_space": "relative_0_1000"},
    )


def crop(box=None):
    box = [100, 100, 600, 600] if box is None else box
    return (
        {"name": "crop_zoom", "arguments": {"bbox_2d": box, "target_image": 0}},
        {"coordinate_space": "relative_0_1000", "target_image": 0,
         "crop_zoom": {"target_image": 1, "bbox_2d": box, "requested_bbox_2d": box,
                       "coordinate_space": "relative_0_1000",
                       "image_outputs": [{"width": 50, "height": 50, "target_image": 1}]}},
    )


def rollout(answer="below", pairs=(), acc=1):
    output = ""
    for call, response in pairs:
        output += (
            f"<tool_call>{json.dumps(call)}</tool_call>\nuser\n"
            f"<tool_response>{json.dumps(response)}</tool_response>\nassistant\n"
        )
    output += f"The cup is lower.\n<answer>{answer}</answer>"
    return {
        "input": f"system\nHistorical system and tool protocol.\nuser\n{source()['question']}\nassistant\n",
        "output": output, "acc": acc, "format": 1, "tool_used": int(bool(pairs)), "step": 1,
    }


class RolloutSftConverterTest(unittest.TestCase):
    def parse(self, row, sources=None):
        sources = sources or [source()]
        return converter.parse_candidate(row, {source()["question"]: sources}, 12)

    def test_logged_reward_does_not_override_gold(self):
        with self.assertRaisesRegex(ValueError, "invalid_relation_label"):
            self.parse(rollout(answer="on", acc=1))
        self.assertEqual(self.parse(rollout(answer="under", acc=0)).answer, "below")

    def test_label_agreement_does_not_assign_semantic_correctness(self):
        for prediction, expected in [("below", "label_match"), ("under", "label_match"),
                                     ("above", "different_label"), ("on", "unmapped_answer"),
                                     (None, "missing_answer")]:
            with self.subTest(prediction=prediction):
                assessment = converter.answer_assessment(prediction, "below")
                self.assertEqual(assessment["label_status"], expected)
                self.assertIsNone(assessment["semantic_verdict"])

    def test_question_must_be_unique_before_prediction_is_used(self):
        with self.assertRaisesRegex(ValueError, "ambiguous_source_mapping"):
            self.parse(rollout(), [source(), source(8, "above")])

    def test_explicit_source_identity_disambiguates_and_checks_gold(self):
        row = rollout()
        row["source_metadata"] = {"source_index": 7, "ground_truth": "below", "dataset_split": "train"}
        self.assertEqual(self.parse(row, [source(), source(8, "above")]).source["source_index"], 7)
        row["source_metadata"]["ground_truth"] = "above"
        with self.assertRaisesRegex(ValueError, "source_metadata_mismatch"):
            self.parse(row)
        row["source_metadata"] = {"source_index": 7, "dataset_split": "validation"}
        with self.assertRaisesRegex(ValueError, "non_training_source"):
            self.parse(row)

    def test_missing_target_defaults_to_original_but_bad_types_fail(self):
        candidate = self.parse(rollout(pairs=[detection(target=None)]))
        self.assertEqual(candidate.calls[0]["arguments"]["target_image"], 0)
        for target in (True, "0", -1, 1):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "invalid_target_image"):
                self.parse(rollout(pairs=[detection(target=target)]))

    def test_query_filters_handle_relations_cjk_and_valid_identifiers(self):
        for query, reason in [
            ("cup to the left of the plate", "relational_grounding_query"),
            ("\u9ed1\u8272\u4e0d\u9508\u94a2\u70e4\u7bb1\u4f4d\u4e8e\u6728\u8d28\u67b6\u5b50\u4e0b\u65b9\u4e2d\u95f4\u4f4d\u7f6e", "long_grounding_query"),
        ]:
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, reason):
                self.parse(rollout(pairs=[detection(query=query)]))
        self.assertTrue(self.parse(rollout(pairs=[detection(query="right hand")])).calls)

    def test_duplicates_fail_but_detection_on_new_crop_is_allowed(self):
        failed = detection(boxes=[])
        with self.assertRaisesRegex(ValueError, "duplicate_tool_call"):
            self.parse(rollout(pairs=[failed, copy.deepcopy(failed), crop()]))
        candidate = self.parse(rollout(pairs=[failed, crop(), detection(target=1)]))
        self.assertEqual(len(candidate.calls), 3)
        self.assertIn("empty_detection", candidate.review_flags)

    def test_all_empty_without_recovery_and_full_image_crop_fail(self):
        with self.assertRaisesRegex(ValueError, "all_detections_empty_without_recovery"):
            self.parse(rollout(pairs=[detection(boxes=[])]))
        with self.assertRaisesRegex(ValueError, "full_image_crop"):
            self.parse(rollout(pairs=[crop([0, 0, 1000, 1000])]))

    def test_unexecuted_tool_call_cannot_become_direct_answer(self):
        for markup in ('<tool_call>{}</tool_call>', '<tool_response>{}</tool_response>'):
            row = rollout()
            row["output"] = markup + row["output"]
            with self.subTest(markup=markup), self.assertRaisesRegex(ValueError, "unfinished_tool_turn"):
                self.parse(row)

    def test_crop_metadata_must_match_call_and_coordinate_system(self):
        rounded_pair = crop()
        rounded_pair[1]["crop_zoom"]["requested_bbox_2d"] = [100.0001, 100, 600, 599.9999]
        self.assertTrue(self.parse(rollout(pairs=[rounded_pair])).calls)
        for field, value, reason in [
            ("coordinate_space", "pixels", "unknown_coordinate_space"),
            ("requested_bbox_2d", [0, 0, 1000, 1000], "crop_response_call_mismatch"),
            ("image_outputs", [{"target_image": 1, "width": 0, "height": 50}], "invalid_crop_image_metadata"),
        ]:
            pair = crop()
            pair[1]["crop_zoom"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, reason):
                self.parse(rollout(pairs=[pair]))

    def test_invalid_or_mismatched_observations_fail(self):
        for key, value, reason in [
            ("query", "other", "tool_response_query_mismatch"),
            ("count", 2, "invalid_detection_response"),
            ("coordinate_space", "pixels", "unknown_coordinate_space"),
            ("confidence", [float("nan")], "invalid_detection_confidence"),
            ("boxes", [[0, 0, 1001, 1000]], "invalid_bbox"),
        ]:
            pair = detection()
            pair[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, reason):
                self.parse(rollout(pairs=[pair]))

    def test_export_preserves_system_images_identity_and_canonical_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image_path = directory / "original.png"
            Image.new("RGB", (100, 100), "red").save(image_path)
            candidate = self.parse(rollout(answer="under", pairs=[detection(target=None), crop()]))
            candidate.source["images"] = [str(image_path)]
            item = converter.convert_candidate(candidate, directory)
            self.assertEqual(item["messages"][0], {"role": "system", "content": "Historical system and tool protocol."})
            self.assertEqual(item["messages"][-1]["content"], "<answer>below</answer>")
            self.assertIn('"target_image": 0', item["messages"][2]["content"])
            self.assertEqual(len(item["images"]), 2)
            self.assertEqual(item["metadata"]["source_index"], 7)
            self.assertIn("<answer>under</answer>", item["metadata"]["original_final"])
            with Image.open(item["images"][1]) as image:
                self.assertEqual(image.size, (50, 50))

    def test_detection_only_export_does_not_decode_original_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image_path = directory / "original.png"
            Image.new("RGB", (10, 10), "red").save(image_path)
            candidate = self.parse(rollout(pairs=[detection()]))
            candidate.source["images"] = [str(image_path)]
            with patch.object(converter.Image, "open", side_effect=AssertionError("unexpected decode")):
                item = converter.convert_candidate(candidate, directory)
            self.assertEqual(item["images"], [str(image_path)])
            self.assertIsNone(item["metadata"]["answer_assessment"]["semantic_verdict"])

    def test_audit_rescores_without_writing_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pq.write_table(pa.Table.from_pylist([source()]), directory / "source.parquet")
            (directory / "1.jsonl").write_text(
                "\n".join(json.dumps(row) for row in [rollout(answer="on"), rollout(acc=0)]) + "\n"
            )
            args = SimpleNamespace(
                rollout_dir=directory, rollout_glob="[0-9]*.jsonl",
                train_parquet=directory / "source.parquet", output=directory / "sft.jsonl",
                crop_dir=None, max_query_words=12, max_samples=0, min_step=0, max_step=100,
                audit_only=True,
            )
            report = converter.convert(args)
            self.assertEqual(report["stats"]["logged_correct_without_label_match"], 1)
            self.assertEqual(report["stats"]["label_match"], 1)
            self.assertEqual(report["selected_direct_answers"], 1)
            self.assertFalse(args.output.exists())
            self.assertFalse((directory / "sft_crops").exists())

    def test_unmapped_answers_are_preserved_in_deduplicated_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pq.write_table(pa.Table.from_pylist([source()]), directory / "source.parquet")
            rows = [rollout(answer="on"), rollout(answer="on"), rollout(answer="above")]
            (directory / "1.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = SimpleNamespace(
                rollout_dir=directory, rollout_glob="[0-9]*.jsonl",
                train_parquet=directory / "source.parquet", output=directory / "sft.jsonl",
                crop_dir=None, max_query_words=12, max_samples=0, min_step=0, max_step=100,
                audit_only=True, semantic_review_jsonl=directory / "reviews.jsonl",
                review_sample_jsonl=directory / "sample.jsonl", review_sample_size=200,
            )
            report = converter.convert(args)
            reviews = [json.loads(line) for line in args.semantic_review_jsonl.read_text().splitlines()]
            self.assertEqual(len(reviews), 2)
            on_review = next(row for row in reviews if row["raw_answer"] == "on")
            self.assertEqual(on_review["trajectory_count"], 2)
            self.assertEqual(on_review["label_status"], "unmapped_answer")
            self.assertIsNone(on_review["semantic_verdict"])
            self.assertIn("<answer>on</answer>", on_review["raw_output"])
            self.assertEqual(report["review_sample_count"], 1)
            with self.assertRaisesRegex(FileExistsError, "review output already exists"):
                converter.convert(args)


if __name__ == "__main__":
    unittest.main()
