import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_tool_reliance as audit  # noqa: E402


def rollout(source, truth, answer, acc, calls, index=0):
    return {
        "output": f"<answer>{answer}</answer>",
        "acc": acc,
        "source_metadata": {"data_source": source, "ground_truth": truth, "source_index": index, "question": "q"},
        "rollout_trace": {"tool_calls": [
            {"tool": tool, "model_turn": turn, "status": "success", "model_observation": json.dumps(observed)}
            for tool, turn, observed in calls
        ]},
    }


class ToolRelianceAuditTest(unittest.TestCase):
    def test_rollout_audit_and_matched_profile(self):
        rows = [
            rollout("visual-agent-tallyqa", "5", "7", 0, [("object_count", 1, {"count": 7})]),
            rollout("visual-agent-tallyqa", "5", "5", 1, [
                ("object_count", 1, {"count": 5}), ("grounding_detect", 2, {"boxes": [], "labels": []}),
            ]),
            rollout("visual-agent-tallyqa", "5", "5", 1, []),
            rollout("visual-agent-tallyqa", "5", "5", 1, [("object_count", 1, {"count": 7})]),
            rollout("visual-agent-ocr", "Total", "Totel", 0, [("ocr_read", 1, {"text": "Totel 12"})], index=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "3.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = audit.audit_rollouts([path])

        count = report["visual-agent-tallyqa"]["tools"]["object_count"]
        self.assertAlmostEqual(count["call_rate"], 3 / 4)
        self.assertAlmostEqual(count["tool_error_rate"], 2 / 3)
        self.assertAlmostEqual(count["copy_rate"], 2 / 3)
        self.assertAlmostEqual(count["wrong_copy_rate"], 1 / 2)
        self.assertAlmostEqual(count["wrong_correction_rate"], 1 / 2)
        self.assertEqual(report["visual-agent-tallyqa"]["tools"]["grounding_detect"]["empty_rate"], 1.0)
        effect = report["visual-agent-tallyqa"]["first_action_effect"]["object_count"]
        self.assertEqual(effect["paired_questions"], 1)
        self.assertAlmostEqual(effect["acc_gap_vs_answer"], 2 / 3 - 1)
        ocr = report["visual-agent-ocr"]["tools"]["ocr_read"]
        self.assertEqual((ocr["tool_error_rate"], ocr["wrong_copy_rate"], ocr["wrong_correction_rate"]), (1.0, 1.0, 0.0))

        profile = audit.matched_profile(report, target_rate=0.5)
        self.assertAlmostEqual(profile["by_source"]["visual-agent-tallyqa"]["object_count"], 0.4)
        self.assertAlmostEqual(profile["by_source"]["visual-agent-ocr"]["ocr_read"], 0.6)
        self.assertEqual(profile["default"], 0.5)

    def test_reliance_branches_are_reported_separately(self):
        rows = [rollout("visual-agent-ocr", "OPEN", "OPEN", 1, [("ocr_read", 2, {"text": "OPEN"})])]
        rows.append({**rows[0], "source_metadata": {**rows[0]["source_metadata"], "reliance_branch": "counterfactual"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = audit.audit_rollouts([path])
        self.assertEqual(set(report), {"visual-agent-ocr", "visual-agent-ocr#counterfactual"})
        self.assertEqual(report["visual-agent-ocr#counterfactual"]["tools"]["ocr_read"]["checked"], 1)

    def test_intermediate_tool_values_are_not_tool_errors(self):
        # Three bars counted correctly for a sum of 30 is a step, not a wrong count.
        rows = [
            rollout("visual-agent-chartqa", "30", "30", 1, [("object_count", 1, {"count": 3})]),
            rollout("visual-agent-chartqa", "42", "40", 0, [("ocr_read", 1, {"text": "A 10 B 30"})], index=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = audit.audit_rollouts([path])
        tools = report["visual-agent-chartqa"]["tools"]
        self.assertEqual((tools["object_count"]["checked"], tools["object_count"]["tool_error_rate"]), (0, None))
        self.assertEqual((tools["ocr_read"]["checked"], tools["ocr_read"]["tool_error_rate"]), (0, None))
        with self.assertRaises(ValueError):
            audit.matched_profile(report, target_rate=0.5)

    def test_fault_audit_counts_copied_and_kept_values(self):
        def prediction(answer, raw):
            return {"prediction": answer, "raw_response": raw}

        fault = {"original": {"count": 5}, "injected": {"count": 9}}
        trace = json.dumps({"tool_calls": [{"name": "object_count", "result": {"count": 5}, "fault": fault}]})
        rows = [prediction("9", trace), prediction("5", trace), prediction("5", "{")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FSC147.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            report = audit.audit_faults([path])
        self.assertEqual(report["FSC147.json"]["object_count"], {"faulted": 2, "copied_fault": 1, "kept_original": 1})
        self.assertEqual(report["FSC147.json"]["_"], {"unreadable": 1})


if __name__ == "__main__":
    unittest.main()
