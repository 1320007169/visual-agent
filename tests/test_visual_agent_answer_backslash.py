import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/VLMEvalKit"))
from vlmeval.api.visual_agent_api import VisualAgentAPI, _normalize_answer_backslashes


class AnswerBackslashTest(unittest.TestCase):
    def test_restore_escaped_commands(self):
        original = r"\\frac{6.8}{x}=\\frac{1.7}{4}"
        normalized = _normalize_answer_backslashes(original)
        self.assertEqual(normalized, r"\frac{6.8}{x}=\frac{1.7}{4}")
        self.assertEqual(_normalize_answer_backslashes(normalized), normalized)

    def test_preserve_single_mixed_and_linebreaks(self):
        for answer in (
            r"\frac{6.8}{x}=\frac{1.7}{4}",
            r"\frac{U}{R}=10\\Omega",
            r"x=1\\y=2",
            r"\begin{aligned}x&=1\\y&=2\end{aligned}",
            r"\\begin{aligned}x&=1\\y&=2\\end{aligned}",
            r"\\customcommand{x}",
            r"\\\frac{1}{2}",
            r"x=1\\[2pt]y=2",
            "plain text\nnext line",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(_normalize_answer_backslashes(answer), answer)

    def test_switch_changes_only_returned_answer(self):
        api = object.__new__(VisualAgentAPI)
        original = r"\\frac{1}{2}"
        trace = {"response": original, "messages": [{"content": original}]}
        raw_response = json.dumps(trace)
        for enabled in ("0", "1"):
            response = {"response": original, "raw_response": raw_response}
            with patch.object(api, "_generate_inner", return_value=(0, response, trace)):
                with patch.dict(os.environ, {"VISUAL_AGENT_NORMALIZE_ANSWER_BACKSLASH": enabled}):
                    code, actual, actual_trace = api.generate_inner([])
            self.assertEqual(code, 0)
            self.assertEqual(actual["response"], r"\frac{1}{2}" if enabled == "1" else original)
            self.assertEqual(actual["raw_response"], raw_response)
            self.assertEqual(actual_trace["response"], original)


if __name__ == "__main__":
    unittest.main()
