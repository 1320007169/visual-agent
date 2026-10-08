import importlib.util
import random
import re
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/tool_faults.py"
SPEC = importlib.util.spec_from_file_location("tool_faults", MODULE_PATH)
tool_faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_faults)


class ToolFaultTest(unittest.TestCase):
    def test_hme_and_confusable_faults_cannot_extend_latex_command_names(self):
        for variant in ("hme", "ocr_confusable"):
            with self.subTest(variant=variant):
                self.assertIsNone(tool_faults.inject_fault(
                    "ocr_read", {"text": r"\neq0"}, random.Random(0), variant=variant))
                original = r"\neq0 + \frac{1}{0}"
                valid = []
                for seed in range(20):
                    faulty = tool_faults.inject_fault(
                        "ocr_read", {"text": original}, random.Random(seed), variant=variant)
                    if faulty is not None:
                        valid.append(faulty)
                        self.assertEqual(re.findall(r"\\[a-zA-Z]+", faulty["text"]),
                                         re.findall(r"\\[a-zA-Z]+", original))
                        self.assertNotEqual(faulty["text"], original)
                self.assertTrue(valid)

    def test_replay_normalizes_defaults_and_duplicate_image_indices(self):
        images = ["same-image", "other-image", "same-image"]
        expected = {"target_image": 0, "mode": "text", "bbox_2d": [0, 0, 1000, 1000]}
        for args in ({"target_image": 2}, {"target_image": "2"}, {"target_image": 2.0}, {}):
            self.assertEqual(tool_faults.normalize_replay_arguments("ocr_read", args, images, {}), expected)
        self.assertNotEqual(tool_faults.normalize_replay_arguments("ocr_read", {"target_image": 1}, images, {}), expected)

    def test_full_image_crop_aliases_follow_actual_crop_and_chain(self):
        images, aliases = ["original"], {}
        for requested in ([0, 0, 1000, 1000], [50, 50, 950, 950]):
            args = {"target_image": len(images) - 1, "bbox_2d": requested}
            result = {"crop_zoom": {"target_image": len(images), "bbox_2d": [0, 0, 1000, 1000]}}
            returned = [f"reencoded-{len(images)}"]
            tool_faults.record_image_aliases("crop_zoom", args, result, images, returned, aliases)
            images.extend(returned)
        canonical = tool_faults.normalize_replay_arguments("ocr_read", {"target_image": 2}, images, aliases)
        self.assertEqual(canonical["target_image"], 0)
        tool_faults.record_image_aliases("crop_zoom", {"target_image": 0},
                                       {"crop_zoom": {"bbox_2d": [0, 0, 500, 500]}}, images, ["region"], aliases)
        images.append("region")
        self.assertEqual(tool_faults.normalize_replay_arguments("ocr_read", {"target_image": 3}, images, aliases)["target_image"], 3)

    def test_count_replay_uses_effective_query_and_ignores_unused_fields(self):
        canonical = tool_faults.normalize_replay_arguments("object_count", {
            "target_image": 0, "query": " apples ", "bbox_2d": [0, 0, 1000, 1000],
        }, ["original"], {})
        self.assertEqual(canonical, {"target_image": 0, "query": "apples"})

    def test_hme_fault_preserves_latex_commands_and_changes_visible_symbols(self):
        observation = {"text": r"\mathrm{0}+\frac{x}{1}", "truncated": False}
        for seed in range(20):
            faulty = tool_faults.inject_fault("ocr_read", observation, random.Random(seed), variant="hme")
            self.assertIsNotNone(faulty)
            self.assertIn(r"\mathrm", faulty["text"])
            self.assertIn(r"\frac", faulty["text"])
            self.assertNotEqual(faulty["text"], observation["text"])
            self.assertEqual(faulty["truncated"], False)
        self.assertIsNone(tool_faults.inject_fault("ocr_read", {"text": r"\frac{}{}"}, random.Random(0), variant="hme"))

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

    def test_confusable_ocr_faults_preserve_schema_and_support_multichar_substitutions(self):
        for text, expected in {"0": "O", "O": "0", "1": "l", "l": "1", "rn": "m", "m": "rn"}.items():
            faulty = tool_faults.inject_fault("ocr_read", {"text": text, "truncated": False},
                                             random.Random(0), variant="ocr_confusable")
            self.assertEqual(faulty, {"text": expected, "truncated": False})
        self.assertIsNone(tool_faults.inject_fault("ocr_read", {"text": "ABC"}, random.Random(0),
                                                  variant="ocr_confusable"))
        self.assertIsNone(tool_faults.inject_fault("object_count", {"count": 3}, random.Random(0),
                                                  variant="ocr_confusable"))
        self.assertIsNone(tool_faults.inject_fault("ocr_read", {"text": r"\mathrm"}, random.Random(0),
                                                  variant="ocr_confusable"))
        self.assertTrue(tool_faults.inject_fault("ocr_read", {"text": r"\mathrm{0}"}, random.Random(0),
                                                 variant="ocr_confusable")["text"].startswith(r"\mathrm{"))
        with self.assertRaises(ValueError):
            tool_faults.inject_fault("ocr_read", {"text": "0"}, random.Random(0), variant="unknown")


if __name__ == "__main__":
    unittest.main()
