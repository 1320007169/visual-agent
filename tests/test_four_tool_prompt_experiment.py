import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_qwen3_vl_8b_four_tool_prompt_eval_modelarts.sh"
PROMPT = ROOT / "prompts/visual_agent_eval_five_tools.txt"
CONFIG = (
    ROOT
    / "reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_four_capabilities_config.yaml"
)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "prompted_tool_summary", ROOT / "scripts/summarize_four_tool_prompt_eval.py"
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)


class FiveToolPromptExperimentTest(unittest.TestCase):
    def test_prompt_schema_wrapper_and_summary_use_exactly_five_tools(self):
        expected = {
            "crop_zoom",
            "grounding_detect",
            "object_count",
            "ocr_read",
            "ground_depth",
        }
        prompt = PROMPT.read_text()
        config = CONFIG.read_text()
        wrapper = ENTRY.read_text()
        for text in (prompt, config):
            self.assertEqual({name for name in expected if name in text}, expected)
            self.assertNotIn("depth_measure", text)
            self.assertNotIn("sam3_", text)
        self.assertEqual({name for name in expected if name in wrapper}, expected)
        self.assertEqual(SUMMARY.EXPECTED_TOOLS, expected)
        self.assertIn(
            "VISUAL_AGENT_ALLOWED_TOOL_NAMES=crop_zoom,grounding_detect,object_count,ocr_read,ground_depth",
            wrapper,
        )

    def test_wrapper_defaults_to_five_tool_prompt_and_treatment_only(self):
        wrapper = ENTRY.read_text()
        self.assertIn("prompts/visual_agent_eval_five_tools.txt", wrapper)
        self.assertIn('"mode": "five_tools"', wrapper)
        self.assertIn('WORK_ROOT="$GROUP_ROOT/five_tools"', wrapper)
        self.assertNotIn("run_visual_agent_eval_qwen3_compare.sh", wrapper)
        self.assertIn("paddleocr_vl.yaml", wrapper)
        self.assertIn("depth_anything_3.yaml", wrapper)
        self.assertIn("countgd_plusplus.yaml", wrapper)

    def test_behavior_summary_reads_raw_traces_and_skips_broken_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook_path = root / "five_tools/run/VisualAgent-vllm_OCRBench.xlsx"
            workbook_path.parent.mkdir(parents=True)
            (root / "five_tools/VisualAgent-vllm_broken.xlsx").symlink_to(
                root / "missing.xlsx"
            )
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["index", "prediction", "raw_response"])
            sheet.append([
                1,
                "EXIT",
                json.dumps({
                    "turns": 2,
                    "tool_calls": [{
                        "name": "ocr_read",
                        "arguments": {"target_image": 0},
                        "result": {"text": "EXIT"},
                    }],
                }),
            ])
            sheet.append([
                2,
                "API failed after retries",
                json.dumps({
                    "turns": 3,
                    "tool_calls": [{
                        "name": "object_count",
                        "arguments": {"query": "cars"},
                        "error": "timeout",
                    }],
                }),
            ])
            workbook.save(workbook_path)

            result = SUMMARY.summarize(root)
            item = result["workbooks"][0]
            self.assertEqual(item["dataset"], "OCRBench")
            self.assertEqual(item["samples"], 2)
            self.assertEqual(item["api_failures"], 1)
            self.assertEqual(item["samples_using_any_tool"], 2)
            self.assertEqual(item["expected_tool_calls"]["ocr_read"], 1)
            self.assertEqual(item["expected_tool_successes"]["ocr_read"], 1)
            self.assertEqual(item["expected_tool_successes"]["object_count"], 0)
            total = result["mode_totals"]["five_tools"]
            self.assertEqual(total["samples"], 2)
            self.assertEqual(total["tool_use_rate"], 1.0)
            self.assertEqual(total["expected_tool_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
