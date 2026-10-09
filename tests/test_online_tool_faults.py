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
        expected = ({10, 11, 13, 14}, {8, 18}, {6, 24}, {0, 60})
        for level, candidates in enumerate(expected):
            with self.subTest(level=level):
                original = {"count": 12, "source": "countgd"}
                changed = faults.inject_fault(original, {"tool": "object_count", "level": level,
                                                           "seed": 3, "count_truth": "15"})
                self.assertIn(changed["count"], candidates)
                self.assertEqual(changed["source"], original["source"])
        self.assertIsNone(faults.inject_fault({"count": True}, {"tool": "object_count", "level": 0,
                                                                "seed": 1, "count_truth": "1"}))

    def test_count_level3_uses_observed_count_and_truth_only_excludes(self):
        values = {faults.inject_fault({"count": 12}, {"tool": "object_count", "level": 3, "seed": seed,
                                                      "count_truth": "1000"})["count"] for seed in range(20)}
        self.assertEqual(values, {0, 60})
        values = {faults.inject_fault({"count": 12}, {"tool": "object_count", "level": 3, "seed": seed,
                                                      "count_truth": "60"})["count"] for seed in range(20)}
        self.assertEqual(values, {0})

    def test_grounding_replay_faults_only_refound_instances(self):
        original = {"boxes": [[100, 100, 200, 200]], "labels": ["cup"], "confidence": [0.9]}
        injected = {"boxes": [[140, 140, 240, 240]], "labels": ["cup"], "confidence": [0.9]}
        # IoU 0.822: the same object under another phrasing keeps this call's label and confidence.
        mug = {"boxes": [[105, 105, 205, 205]], "labels": ["mug"], "confidence": [0.7]}
        self.assertEqual(faults.replay_fault("grounding_detect", mug, original, injected),
                         {"boxes": [[140, 140, 240, 240]], "labels": ["mug"], "confidence": [0.7]})
        # Only the re-found cup moves; the plate stays real.
        both = {"boxes": [[500, 500, 600, 600], [105, 105, 205, 205]], "labels": ["plate", "cup"],
                "confidence": [0.6, 0.8], "source": "groundingdino"}
        self.assertEqual(faults.replay_fault("grounding_detect", both, original, injected),
                         {"boxes": [[500, 500, 600, 600], [140, 140, 240, 240]], "labels": ["plate", "cup"],
                          "confidence": [0.6, 0.8], "source": "groundingdino"})
        # IoU 0.667: an overlapping but different object (e.g. a rider on a bike) stays real.
        neighbour = {"boxes": [[120, 100, 220, 200]], "labels": ["person"], "confidence": [0.8]}
        self.assertAlmostEqual(faults.max_iou(neighbour["boxes"], original["boxes"]), 2 / 3)
        self.assertIsNone(faults.replay_fault("grounding_detect", neighbour, original, injected))
        distant = {"boxes": [[150, 100, 250, 200]], "labels": ["plate"], "confidence": [0.8]}
        self.assertIsNone(faults.replay_fault("grounding_detect", distant, original, injected))
        self.assertEqual(faults.max_iou([[1, 2, 3]], original["boxes"]), 0.0)
        self.assertIsNone(faults.replay_fault("grounding_detect", {"boxes": []}, original, injected))
        self.assertIsNone(faults.replay_fault("grounding_detect", {"status": "error"}, original, injected))

    def test_grounding_replay_follows_box_indices_and_l2_drops_matches(self):
        original = {"boxes": [[100, 100, 200, 200], [500, 500, 600, 600]], "labels": ["cup", "plate"],
                    "confidence": [0.9, 0.8]}
        # IoU with the original cup is 0.96; L1 swapped the two boxes.
        real = {"boxes": [[102, 100, 202, 200], [800, 800, 900, 900]], "labels": ["mug", "lamp"],
                "confidence": [0.7, 0.6]}
        swapped = {**original, "boxes": [[500, 500, 600, 600], [100, 100, 200, 200]]}
        self.assertEqual(faults.replay_fault("grounding_detect", real, original, swapped)["boxes"],
                         [[500, 500, 600, 600], [800, 800, 900, 900]])
        cleared = {**original, "boxes": [], "labels": [], "confidence": []}
        self.assertEqual(faults.replay_fault("grounding_detect", real, original, cleared),
                         {"boxes": [[800, 800, 900, 900]], "labels": ["lamp"], "confidence": [0.6]})

    def test_count_replay_requires_the_original_count(self):
        original, injected = {"count": 7}, {"count": 9}
        boxes = [[i * 20, 100, i * 20 + 10, 110] for i in range(7)]
        self.assertEqual(faults.replay_fault("object_count", {"count": 7, "source": "countgd"}, original, injected,
                         original_evidence={"boxes": boxes}, evidence={"boxes": list(reversed(boxes))}),
                         {"count": 9, "source": "countgd"})
        self.assertIsNone(faults.replay_fault("object_count", {"count": 3}, original, injected))
        self.assertIsNone(faults.replay_fault("object_count", {"count": True}, {"count": 1}, injected))
        self.assertIsNone(faults.replay_fault("ocr_read", {"text": "x"}, {"text": "x"}, {"text": "y"}))

    def test_count_replay_requires_matching_instances(self):
        original, injected = {"count": 2}, {"count": 3}
        boxes = [[100, 100, 200, 200], [500, 500, 600, 600]]
        shifted = [[502, 500, 602, 600], [102, 100, 202, 200]]
        self.assertEqual(faults.replay_fault("object_count", original, original, injected,
                         original_evidence={"boxes": boxes}, evidence={"boxes": shifted}), injected)
        for evidence in (None, {"boxes": []}, {"boxes": [[800, 800, 900, 900], boxes[0]]},
                         {"boxes": [boxes[0], boxes[0]]}):
            with self.subTest(evidence=evidence):
                self.assertIsNone(faults.replay_fault("object_count", original, original, injected,
                                  original_evidence={"boxes": boxes}, evidence=evidence))

    def test_count_replay_treats_two_empty_results_as_the_same_evidence(self):
        original, injected = {"count": 0}, {"count": 2}
        empty = {"count": 0, "boxes": []}
        self.assertEqual(faults.replay_fault("object_count", {"count": 0, "source": "countgd"}, original, injected,
                         original_evidence=empty, evidence=empty), {"count": 2, "source": "countgd"})
        # A non-zero real count, or detections despite a zero count, stays real.
        self.assertIsNone(faults.replay_fault("object_count", {"count": 3}, original, injected,
                          original_evidence=empty, evidence={"count": 3, "boxes": [[1, 1, 9, 9]] * 3}))
        self.assertIsNone(faults.replay_fault("object_count", {"count": 0}, original, injected,
                          original_evidence=empty, evidence={"count": 0, "boxes": [[1, 1, 9, 9]]}))
        self.assertIsNone(faults.replay_fault("object_count", {"count": 0}, original, injected,
                          original_evidence=empty, evidence=None))

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
