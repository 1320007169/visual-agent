import importlib.util
from pathlib import Path
import tempfile
import unittest

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("failure_report", ROOT / "scripts/summarize_visual_agent_eval_failures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FailureReportTest(unittest.TestCase):
    def test_report_counts_failures_and_deduplicates_symlinked_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(["index", "prediction"])
            sheet.append([1, "A"])
            sheet.append([2, "Failed to obtain answer via API."])
            sheet.append([3, None])
            workbook.save(root / "VisualAgent-vllm_OCRBench.xlsx")
            (root / "alias").symlink_to(root, target_is_directory=True)
            reports = module.summarize(root)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["total"], 3)
        self.assertEqual(reports[0]["failed_indices"], [2])
        self.assertEqual(reports[0]["empty_count"], 1)


if __name__ == "__main__":
    unittest.main()
