import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis/rollout_decisions.py"
spec = importlib.util.spec_from_file_location("rollout_decisions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(tool_calls, score, source="visual-agent-ocr", forced=None):
    trace = {"tool_calls": tool_calls}
    if forced is not None:
        trace["forced_decision"] = forced
    return {"score": score, "rollout_trace": trace, "source_metadata": {"data_source": source}}


class RolloutDecisionsTest(unittest.TestCase):
    def test_first_turn_decisions_exclude_forced_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step2 = [
                sample([{"tool": "ocr_read", "model_turn": 1}, {"tool": "crop_zoom", "model_turn": 2}], 1.0),
                sample([{"tool": "crop_zoom", "model_turn": 2}], 0.0, source="visual-agent-tallyqa"),
                sample([], 1.0),
                sample([{"tool": "object_count", "model_turn": 1}], 0.0, forced="object_count"),
            ]
            (root / "2.jsonl").write_text("\n".join(json.dumps(row) for row in step2) + "\n")
            (root / "10.jsonl").write_text(json.dumps(sample([], 0.0)) + "\n")
            rows = module.summarize(root)
        self.assertEqual([row["step"] for row in rows], [2, 10])
        self.assertEqual(rows[0]["samples"], 3)
        self.assertEqual(rows[0]["rate"], {"direct": 2 / 3, "ocr_read": 1 / 3})
        self.assertEqual(rows[0]["score"], {"direct": 0.5, "ocr_read": 1.0})
        self.assertEqual(rows[0]["rate_by_source"]["visual-agent-tallyqa"], {"direct": 1.0})
        self.assertEqual(rows[1]["rate"], {"direct": 1.0})

    def test_missing_trace_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1.jsonl").write_text(json.dumps({"score": 0.0}) + "\n")
            with self.assertRaisesRegex(ValueError, "rollout_trace"):
                module.summarize(root)


if __name__ == "__main__":
    unittest.main()
