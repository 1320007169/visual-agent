import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/export_reviewed_rl_sft.py"
SPEC = importlib.util.spec_from_file_location("export_reviewed_rl_sft_test", SCRIPT_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportReviewedRlSftTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.candidates_path = self.root / "candidates.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.decisions_path = self.root / "decisions.jsonl"
        self.output_dir = self.root / "exported"
        self.rows = [
            {
                "messages": [
                    {"role": "user", "content": "<image> Where is the cup relative to the plate?"},
                    {"role": "assistant", "content": '<tool_call>{"name":"grounding_detect","arguments":{"query":"cup","target_image":0}}</tool_call>'},
                    {"role": "user", "content": '<tool_response>{"boxes":[[100,100,200,200]]}</tool_response>'},
                    {"role": "assistant", "content": "<answer>above</answer>"},
                ],
                "images": [f"/original/image{index}.jpg"],
                "metadata": {"source_index": index, "original_final": "An unchanged original explanation.", "quality_status": "candidate"},
                "other_field": {"preserve": index},
            }
            for index in range(3)
        ]
        raw_lines = [(json.dumps(row) + "\n").encode("utf-8") for row in self.rows]
        self.candidates_path.write_bytes(b"".join(raw_lines))
        self.original_candidates = self.candidates_path.read_bytes()
        self.manifest = {
            "candidate_file_sha256": hashlib.sha256(self.original_candidates).hexdigest(),
            "cases": [
                {"case_id": f"arbitrary_case_{index}", "candidate_line": index + 1,
                 "row_sha256": hashlib.sha256(raw).hexdigest(), "source_index": index}
                for index, raw in enumerate(raw_lines)
            ],
        }
        self.decisions = [
            {"case_id": f"arbitrary_case_{index}", "verdict": verdict, "reason": "Reviewed image and trajectory.",
             "reviewer": "test_agent", "tool_grounding": "pass", "answer_supported": "pass",
             "query_quality": "pass", "inspected_images": [f"/review/image{index}.jpg"]}
            for index, verdict in enumerate(("keep", "reject", "review"))
        ]
        self.write_inputs()

    def write_inputs(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.decisions_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.decisions), encoding="utf-8"
        )

    def export(self, decision_paths=None):
        return exporter.export_reviewed(
            self.candidates_path, self.manifest_path,
            decision_paths or [self.decisions_path], self.output_dir,
        )

    def assert_invalid(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.export()
        self.assertFalse(self.output_dir.exists())

    def test_export_preserves_messages_images_and_original_metadata(self):
        summary = self.export()
        exported = [json.loads(line) for line in (self.output_dir / "sft.jsonl").read_text().splitlines()]
        self.assertEqual(len(exported), 1)
        row = exported[0]
        self.assertEqual(row["messages"], self.rows[0]["messages"])
        self.assertEqual(row["images"], self.rows[0]["images"])
        self.assertEqual(row["other_field"], self.rows[0]["other_field"])
        self.assertEqual(row["metadata"]["original_final"], self.rows[0]["metadata"]["original_final"])
        self.assertEqual(row["metadata"]["visual_review"], self.decisions[0])
        self.assertEqual(row["metadata"]["quality_status"], "agent_visual_reviewed_not_human_verified")
        self.assertEqual(summary["verdict_counts"], {"keep": 1, "reject": 1, "review": 1})
        self.assertEqual(summary["exported_sft_rows"], 1)
        self.assertEqual(self.candidates_path.read_bytes(), self.original_candidates)
        self.assertEqual(json.loads((self.output_dir / "summary.json").read_text()), summary)
        self.assertEqual(
            [json.loads(line) for line in (self.output_dir / "review_decisions.jsonl").read_text().splitlines()],
            self.decisions,
        )

    def test_candidate_file_hash_mismatch_rejected(self):
        self.manifest["candidate_file_sha256"] = "0" * 64
        self.write_inputs()
        self.assert_invalid("candidate file hash mismatch")

    def test_row_hash_includes_newline(self):
        self.manifest["cases"][0]["row_sha256"] = hashlib.sha256(
            self.original_candidates.splitlines()[0]
        ).hexdigest()
        self.write_inputs()
        self.assert_invalid("row hash mismatch")

    def test_source_index_mismatch_rejected(self):
        self.manifest["cases"][0]["source_index"] = 42
        self.write_inputs()
        self.assert_invalid("source_index mismatch")

    def test_missing_decision_rejected(self):
        self.decisions.pop()
        self.write_inputs()
        self.assert_invalid("missing decisions")

    def test_duplicate_decision_rejected_across_files(self):
        second = self.root / "second_decisions.jsonl"
        second.write_text(json.dumps(self.decisions[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate decision"):
            self.export([self.decisions_path, second])
        self.assertFalse(self.output_dir.exists())

    def test_unknown_decision_rejected(self):
        self.decisions[0]["case_id"] = "unknown"
        self.write_inputs()
        self.assert_invalid("unknown decision")

    def test_keep_requires_every_check_to_pass(self):
        for field in exporter.REVIEW_CHECKS:
            for value in ("uncertain", "fail"):
                with self.subTest(field=field, value=value):
                    self.decisions[0][field] = value
                    self.write_inputs()
                    self.assert_invalid("keep requires")
                    self.decisions[0][field] = "pass"

    def test_keep_requires_inspected_images(self):
        self.decisions[0]["inspected_images"] = []
        self.write_inputs()
        self.assert_invalid("keep requires")

    def test_review_images_must_be_absolute(self):
        self.decisions[0]["inspected_images"] = ["relative/image.jpg"]
        self.write_inputs()
        self.assert_invalid("absolute paths")

    def test_multiple_decision_files_can_cover_the_manifest(self):
        second = self.root / "second_decisions.jsonl"
        second.write_text(json.dumps(self.decisions[-1]) + "\n", encoding="utf-8")
        self.decisions = self.decisions[:-1]
        self.write_inputs()
        summary = self.export([self.decisions_path, second])
        self.assertEqual(summary["reviewed_cases"], 3)

    def test_existing_output_directory_is_not_modified(self):
        self.output_dir.mkdir()
        sentinel = self.output_dir / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual(list(self.output_dir.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
