import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_ocr_reliance_eval import summarize


class OCRRelianceSummaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = {}
        for label in ["base", "v2_step40"]:
            for condition in ["clean", "training", "ocr_confusable"]:
                rows = []
                for i in range(1000):
                    calls = []
                    prediction = "A"
                    if i == 0 and condition == "clean":
                        calls = [{"name": "ocr_read", "arguments": {"target_image": 0},
                                  "result": {"text": "B" if label == "base" else "A"}}]
                        prediction = "B" if label == "base" else "A"
                    if i == 1 and condition != "clean":
                        calls = [{"name": "ocr_read", "arguments": {"target_image": 0},
                                  "fault": {"original": {"text": "A"}, "injected": {"text": "B"}}}]
                    rows.append({"index": i, "question": f"q{i}", "category": "Regular Text Recognition",
                                 "answer": "['A']", "prediction": prediction,
                                 "raw_response": json.dumps({"tool_calls": calls})})
                path = self.root / label / condition / "dino_latest/VisualAgent-vllm/run/model_OCRBench.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(rows))
                self.paths[label, condition] = path

    def test_natural_error_cohort_is_fixed_and_fault_exposure_is_separate(self):
        result = summarize([self.root])
        cohort = "natural_ocr_error_fixed_from_base_clean"
        self.assertEqual(result["cohorts"][cohort], [0])
        self.assertEqual(result["arms"]["base/clean"][cohort]["correct"], 0)
        self.assertEqual(result["arms"]["v2_step40/clean"][cohort]["correct"], 1)
        self.assertEqual(result["matched_faults"]["training"]["indices"], [1])
        self.assertEqual(result["arms"]["base/training"]["all"]["fault_exposed"], 1)
        self.assertEqual(result["arms"]["base/training"]["all"]["n"], 1000)

    def test_different_faults_do_not_count_as_matched_exposure(self):
        path = self.paths["v2_step40", "training"]
        rows = json.loads(path.read_text())
        trace = json.loads(rows[1]["raw_response"])
        trace["tool_calls"][0]["fault"]["injected"]["text"] = "C"
        rows[1]["raw_response"] = json.dumps(trace)
        path.write_text(json.dumps(rows))
        self.assertEqual(summarize([self.root])["matched_faults"]["training"]["n"], 0)

    def test_alias_arguments_keep_faults_paired(self):
        path = self.paths["v2_step40", "training"]
        rows = json.loads(path.read_text())
        trace = json.loads(rows[1]["raw_response"])
        trace["tool_calls"][0]["arguments"] = {"target_image": 2}
        trace["tool_calls"][0]["canonical_arguments"] = {"target_image": 0}
        rows[1]["raw_response"] = json.dumps(trace)
        path.write_text(json.dumps(rows))
        self.assertEqual(summarize([self.root])["matched_faults"]["training"]["indices"], [1])

    def test_mismatched_questions_are_rejected(self):
        path = self.paths["v2_step40", "training"]
        rows = json.loads(path.read_text())
        rows[1]["question"] = "different question"
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "Unpaired sample"):
            summarize([self.root])

    def test_missing_rows_are_rejected(self):
        path = self.paths["v2_step40", "training"]
        rows = json.loads(path.read_text())
        path.write_text(json.dumps(rows[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly the 1000"):
            summarize([self.root])


if __name__ == "__main__":
    unittest.main()
