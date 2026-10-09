import importlib.util
import re
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/online_tool_faults.py"
SPEC = importlib.util.spec_from_file_location("online_tool_faults", MODULE)
faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(faults)


class OnlineToolFaultTest(unittest.TestCase):
    def test_ocr_levels_change_answer_and_preserve_schema(self):
        original = {"text": "Invoice total 128.50 USD", "truncated": False}
        for level in range(3):
            with self.subTest(level=level):
                spec = {"tool": "ocr_read", "level": level, "seed": 4,
                        "answer": "128.50", "aliases": ["128.50"]}
                changed = faults.inject_fault(original, spec)
                self.assertIsNotNone(changed)
                self.assertEqual(set(changed), set(original))
                self.assertNotIn("12850", faults._normalized(changed["text"]))
                self.assertFalse(changed["truncated"])

    def test_hme_levels_keep_latex_commands(self):
        text = r"\frac{x}{1}+\sqrt{9}"
        original = {"text": text, "truncated": False}
        commands = re.findall(r"\\[A-Za-z]+", text)
        for level in range(3):
            with self.subTest(level=level):
                spec = {"tool": "ocr_read", "level": level, "seed": 2,
                        "answer": text, "aliases": [text], "variant": "hme"}
                changed = faults.inject_fault(original, spec)
                self.assertIsNotNone(changed)
                self.assertEqual(re.findall(r"\\[A-Za-z]+", changed["text"]), commands)
                self.assertNotIn(faults._normalized(text), faults._normalized(changed["text"]))

    def test_ocr_skips_when_an_alias_survives_or_answer_is_absent(self):
        spec = {"tool": "ocr_read", "level": 0, "seed": 1, "answer": "120", "aliases": ["120"]}
        self.assertIsNone(faults.inject_fault({"text": "120 appears twice: 120"}, spec))
        self.assertIsNone(faults.inject_fault({"text": "no matching number"}, spec))
        self.assertIsNone(faults.inject_fault({"text": r"\frac{}{}"},
                                              {**spec, "answer": r"\frac{}{}", "variant": "hme"}))

    def test_count_levels_differ_from_truth_and_observation(self):
        expected = ({10, 11, 13, 14}, {8, 18}, {6, 24}, {0, 75})
        for level, candidates in enumerate(expected):
            with self.subTest(level=level):
                original = {"count": 12, "source": "countgd"}
                changed = faults.inject_fault(original, {"tool": "object_count", "level": level,
                                                           "seed": 3, "count_truth": "15"})
                self.assertIn(changed["count"], candidates)
                self.assertEqual(changed["source"], original["source"])
        self.assertIsNone(faults.inject_fault({"count": True}, {"tool": "object_count", "level": 0,
                                                                "seed": 1, "count_truth": "1"}))

    def test_grounding_levels_and_single_box_skip(self):
        original = {"boxes": [[100, 100, 200, 200], [500, 500, 600, 600]],
                    "labels": ["cup", "plate"], "confidence": [0.8, 0.7]}
        for level in range(3):
            changed = faults.inject_fault(original, {"tool": "grounding_detect", "level": level, "seed": 2})
            self.assertEqual(set(changed), set(original))
            self.assertNotEqual(changed["boxes"], original["boxes"])
        self.assertEqual(faults.inject_fault(original, {"tool": "grounding_detect", "level": 1,
                                                        "seed": 2})["boxes"], original["boxes"][::-1])
        self.assertEqual(faults.inject_fault(original, {"tool": "grounding_detect", "level": 2,
                                                        "seed": 2})["labels"], [])
        self.assertIsNone(faults.inject_fault({**original, "boxes": original["boxes"][:1]},
                                              {"tool": "grounding_detect", "level": 1, "seed": 2}))


if __name__ == "__main__":
    unittest.main()
