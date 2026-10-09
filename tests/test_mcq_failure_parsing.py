import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MCQFailureParsingTests(unittest.TestCase):
    def test_mme_rejects_agent_failures_before_extracting_a_choice(self):
        path = ROOT / "evaluation/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py"
        extract = load_function(path, "extract_characters_regex", {"re": re})
        self.assertEqual(extract("Failed to obtain answer via API."), "")
        self.assertEqual(extract("Agent exceeded the maximum of 8 turns"), "")
        self.assertEqual(extract('<tool_call>{"name": "crop_zoom", "arguments": {}}</tool_call>'), "")
        self.assertEqual(extract("<answer>E</answer>"), "E")

    def test_hrbench_rejects_agent_failures_before_option_matching(self):
        path = ROOT / "evaluation/VLMEvalKit/vlmeval/utils/matching_util.py"
        infer = load_function(path, "can_infer", {
            "can_infer_option": lambda answer, choices: "A" if "A" in answer else False,
            "can_infer_text": lambda answer, choices: False,
        })
        self.assertFalse(infer("Failed to obtain answer via API.", {"A": "apple"}))
        self.assertFalse(infer("Agent exceeded the maximum of 8 turns", {"A": "apple"}))
        self.assertFalse(infer('<tool_call>{"name": "crop_zoom"}</tool_call>', {"A": "apple"}))
        self.assertEqual(infer("A", {"A": "apple"}), "A")

    def test_hrbench_does_not_send_agent_failures_to_a_judge(self):
        path = ROOT / "evaluation/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py"
        evaluate = load_function(path, "eval_vanilla", {
            "extract_answer_from_item": lambda model, item, dataset_name=None: {"opt": "A", "log": "A"},
        })
        item = {"prediction": "Failed to obtain answer via API.", "GT": "A"}
        self.assertEqual(evaluate(object(), item, "HRBench8K")["hit"], 0)


if __name__ == "__main__":
    unittest.main()
