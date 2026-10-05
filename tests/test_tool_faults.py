import importlib.util
import random
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/tool_faults.py"
SPEC = importlib.util.spec_from_file_location("tool_faults", MODULE_PATH)
tool_faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_faults)


class ToolFaultTest(unittest.TestCase):
    def test_count_fault_avoids_original_and_ground_truth(self):
        for seed in range(50):
            faulty = tool_faults.inject_fault("object_count", {"count": 4}, random.Random(seed), ground_truth="6")
            self.assertNotIn(faulty["count"], {4, 6})
            self.assertGreaterEqual(faulty["count"], 0)
        self.assertGreater(tool_faults.inject_fault("object_count", {"count": 0}, random.Random(0))["count"], 0)
        self.assertIsNone(tool_faults.inject_fault("object_count", {"count": True}, random.Random(0)))

    def test_detection_fault_empties_or_moves_boxes(self):
        observation = {"boxes": [[100, 100, 200, 300]], "confidence": [0.9], "labels": ["cup"]}
        outcomes = {
            bool(tool_faults.inject_fault("grounding_detect", observation, random.Random(seed))["boxes"])
            for seed in range(20)
        }
        self.assertEqual(outcomes, {True, False})
        for seed in range(20):
            boxes = tool_faults.inject_fault("grounding_detect", observation, random.Random(seed))["boxes"]
            for x1, y1, x2, y2 in boxes:
                self.assertEqual((x2 - x1, y2 - y1), (100, 200))
                self.assertTrue(0 <= x1 and x2 <= 1000 and 0 <= y1 and y2 <= 1000)
        self.assertIsNone(tool_faults.inject_fault("grounding_detect", {"boxes": []}, random.Random(0)))

    def test_depth_fault_reverses_order_or_scales_single_region(self):
        observation = {"regions": [{"bbox_2d": [0, 0, 10, 10], "depth_m": 1.5}, {"bbox_2d": [5, 5, 9, 9], "depth_m": 3.0}]}
        faulty = tool_faults.inject_fault("depth_measure", observation, random.Random(0))
        self.assertEqual([region["depth_m"] for region in faulty["regions"]], [3.0, 1.5])
        self.assertEqual(faulty["regions"][0]["bbox_2d"], [0, 0, 10, 10])
        single = tool_faults.inject_fault("depth_measure", {"regions": [{"depth_m": 2.0}]}, random.Random(0))
        self.assertIn(single["regions"][0]["depth_m"], {1.0, 4.0})

    def test_ocr_fault_corrupts_answer_span(self):
        observation = {"text": "Total: 128.50 USD", "truncated": False}
        for seed in range(30):
            faulty = tool_faults.inject_fault("ocr_read", observation, random.Random(seed), ground_truth="128.50")
            self.assertNotIn("128.50", faulty["text"])
            self.assertTrue(faulty["text"].startswith("Total: ") and faulty["text"].endswith(" USD"))
            self.assertFalse(faulty["truncated"])
        self.assertIsNone(tool_faults.inject_fault("ocr_read", {"text": "+ -"}, random.Random(0)))

    def test_errors_and_unknown_tools_are_not_faulted(self):
        self.assertIsNone(tool_faults.inject_fault("object_count", {"status": "error", "count": 3}, random.Random(0)))
        self.assertIsNone(tool_faults.inject_fault("crop_zoom", {"target_image": 1}, random.Random(0)))
        self.assertIsNone(tool_faults.inject_fault("ocr_read", "plain text", random.Random(0)))

    def test_same_seed_gives_same_fault(self):
        observation = {"count": 12}
        first = tool_faults.inject_fault("object_count", observation, random.Random("q1:object_count"))
        second = tool_faults.inject_fault("object_count", observation, random.Random("q1:object_count"))
        self.assertEqual(first, second)

    def test_fault_probability_profile(self):
        profile = {"default": 0.2, "by_tool": {"ocr_read": 0.5}, "by_source": {"visual-agent-fsc147": {"object_count": 0.9}}}
        self.assertEqual(tool_faults.fault_probability(None, "any", "ocr_read"), 1.0)
        self.assertEqual(tool_faults.fault_probability(profile, "visual-agent-fsc147", "object_count"), 0.9)
        self.assertEqual(tool_faults.fault_probability(profile, "visual-agent-ocr", "ocr_read"), 0.5)
        self.assertEqual(tool_faults.fault_probability(profile, "visual-agent-ocr", "depth_measure"), 0.2)


if __name__ == "__main__":
    unittest.main()
