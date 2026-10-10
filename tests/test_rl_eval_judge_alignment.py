import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "aligned_reward", ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py",
)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


class EvalJudgeAlignmentTest(unittest.TestCase):
    def test_ocr_requires_complete_answer_and_preserves_formula_case(self):
        cases = [
            ("cat or dog", "cat", "textvqa", 0),
            ("new  york", "New York", "textvqa", 1),
            (r"\\frac{1}{2}", r"\frac{1}{2}", "hme100k", 1),
            (r"\frac{X}{2}", r"\frac{x}{2}", "hme100k", 0),
        ]
        with patch.object(reward, "judge_match", return_value=False):
            for answer, reference, original_source, expected in cases:
                extra = {"data_source": "visual-agent-ocr", "original_source": original_source}
                with self.subTest(answer=answer):
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", reference, extra)["acc"], expected)

    def test_chart_inexact_numbers_require_contextual_judging(self):
        extra = {"data_source": "visual-agent-chartqa", "question": "What is the value?"}
        with patch.object(reward, "judge_match", return_value=False) as judge:
            for answer, reference, expected in (("95", "100", 0), ("105", "100", 0),
                                                 ("1966", "1965", 0), ("52%", "50%", 0),
                                                 ("2%", "2", 0), ("0", "0", 1)):
                with self.subTest(answer=answer, reference=reference):
                    judge.reset_mock()
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", reference, extra)["acc"], expected)
                    self.assertEqual(judge.called, not expected)
                    if judge.called:
                        self.assertEqual(judge.call_args.kwargs["extra_info"], extra)

    def test_ocr_api_receives_all_aliases_and_strict_transcription_rules(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content=r"Equivalent. \boxed{Yes}"), finish_reason="stop")
        ]
        extra = {"data_source": "visual-agent-ocr", "question": "Which city?",
                 "answer_aliases": ["London", "LONDON CITY"]}
        with patch.object(reward, "_evaluation_judge_template", return_value="Verify.\n"), patch.object(
            reward, "_judge_client_and_model", return_value=(client, "judge")
        ):
            result = reward.compute_score("<answer>The UK capital</answer>", "London", extra)
        self.assertEqual(result["acc"], 1)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(len(request["messages"]), 1)
        prompt = request["messages"][0]["content"]
        self.assertIn("Alternative 1: London\nAlternative 2: LONDON CITY", prompt)
        self.assertIn("Reject contradictory answers or lists of guesses", prompt)
        self.assertIn("preserve words, digits, mathematical symbols and variable case", prompt)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertNotIn("stop", request)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})

    def test_chart_api_limits_tolerance_to_continuous_quantities(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content=r"\boxed{Yes}"), finish_reason="stop")
        ]
        extra = {"data_source": "visual-agent-chartqa", "question": "What percentage?",
                 "answer_aliases": ["2", "two percent"]}
        with patch.object(reward, "_evaluation_judge_template", return_value="Verify.\n"), patch.object(
            reward, "_judge_client_and_model", return_value=(client, "judge")
        ):
            self.assertEqual(reward.compute_score("<answer>2 percent</answer>", "2", extra)["acc"], 1)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("abs(p-r)/abs(r) <= 0.05", prompt)
        self.assertIn("discrete counts must match exactly", prompt)
        self.assertIn("question does not establish a continuous quantity, require exact", prompt)
        self.assertIn("Alternative 2: two percent", prompt)

    def test_unresolved_and_truncated_judgments_do_not_become_wrong_answers(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content=r"\boxed{Yes}"), finish_reason="length")
        ]
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "", "LLM_AS_A_JUDGE_PRIMARY_RETRIES": "1"}), patch.object(
            reward, "_evaluation_judge_template", return_value="Verify.\n"
        ), patch.object(reward, "_judge_client_and_model", return_value=(client, "judge")):
            result = reward.compute_score("<answer>unmatched</answer>", "reference", {"data_source": "visual-agent-ocr"})
            self.assertEqual(result["reward_valid"], 0.0)
            self.assertEqual(result["score"], 0.0)
            self.assertIn("acc", result)

    def test_failure_messages_are_not_sent_to_judge(self):
        with patch.object(reward, "judge_match", side_effect=AssertionError("judge called")):
            for source in ("visual-agent-ocr", "visual-agent-chartqa"):
                for answer in ("", "Failed to obtain answer via API.", "Agent exceeded the maximum of 8 turns"):
                    result = reward.compute_score(f"<answer>{answer}</answer>", "8", {"data_source": source})
                    self.assertEqual(result["acc"], 0)
