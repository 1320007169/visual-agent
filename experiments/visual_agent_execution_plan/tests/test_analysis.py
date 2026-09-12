import unittest

from visual_agent_experiments.analysis import summarize_utility, validate_prediction_records


def record(sample_id, condition, correct, tool_calls=None):
    return {
        "run_id": "dev-001",
        "checkpoint": "B1",
        "split": "dev",
        "sample_id": sample_id,
        "source_image_id": f"image-{sample_id}",
        "mode": "agent",
        "condition": condition,
        "seed": 0,
        "correct": correct,
        "tool_calls": tool_calls or [],
        "finish_reason": "normal-finish",
        "generated_tokens": 12,
        "visual_tokens": 5,
        "elapsed_seconds": 0.3,
    }


class AnalysisTest(unittest.TestCase):
    def test_summary_is_paired_by_sample(self):
        calls = [{"name": "crop_zoom", "status": "ok", "image_id": "image-a", "bbox": [0, 0, 10, 10], "coordinate_system": "original-image"}]
        records = [
            record("a", "on", True, calls),
            record("a", "off", False),
            record("b", "on", False),
            record("b", "off", False),
        ]
        summary = summarize_utility(records, bootstrap_repeats=20, seed=3)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["delta_tool"], 0.5)
        self.assertEqual(summary["tool_rate"], 0.5)
        self.assertEqual(summary["effective_call_rate"], 1.0)

    def test_native_tool_call_is_invalid(self):
        invalid = record("a", "off", False, [{"name": "crop_zoom"}])
        invalid["mode"] = "native"
        errors = validate_prediction_records([invalid])
        self.assertTrue(any("native mode" in error for error in errors))

    def test_missing_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both ON and OFF"):
            summarize_utility([record("only-on", "on", True)])
