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
    def test_ocr_uses_benchmark_substring_rules_and_final_answer_unescaping(self):
        cases = [
            ("The word is London", "London", "textvqa", 1),
            ("new  york", "New York", "textvqa", 0),
            (r"The expression is \\frac{1}{2}", r"\frac{1}{2}", "hme100k", 1),
            (r"\frac{X}{2}", r"\frac{x}{2}", "hme100k", 0),
        ]
        with patch.object(reward, "judge_match", return_value=False):
            for answer, reference, original_source, expected in cases:
                extra = {"data_source": "visual-agent-ocr", "original_source": original_source}
                with self.subTest(answer=answer):
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", reference, extra)["acc"], expected)

    def test_chart_rule_matches_five_percent_including_years_and_percentages(self):
        extra = {"data_source": "visual-agent-chartqa", "question": "What is the value?"}
        with patch.object(reward, "judge_match", return_value=False) as judge:
            for answer, reference, expected in (("95", "100", 1), ("105", "100", 1),
                                                 ("106", "100", 0), ("1966", "1965", 1),
                                                 ("52%", "50%", 1), ("2%", "2", 0),
                                                 ("0", "0", 1), ("1", "0", 0)):
                with self.subTest(answer=answer, reference=reference):
                    judge.reset_mock()
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", reference, extra)["acc"], expected)
                    self.assertEqual(judge.called, not expected)

    def test_ocr_api_uses_original_template_all_aliases_and_boxed_verdict(self):
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
        self.assertEqual(request["messages"], [{"role": "user", "content":
            "Verify.\n【用户问题】:Which city?\n【参考答案】：Alternative 1: London\nAlternative 2: LONDON CITY"
            "\n【模型回答】：The UK capital"}])
        self.assertEqual(request["max_tokens"], 1024)
        self.assertNotIn("stop", request)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})

    def test_chart_api_judges_each_reference_with_numeric_instructions(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=r"\boxed{No}"), finish_reason="stop")]),
            Mock(choices=[Mock(message=Mock(content=r"\boxed{Yes}"), finish_reason="stop")]),
        ]
        extra = {"data_source": "visual-agent-chartqa", "question": "What percentage?",
                 "answer_aliases": ["2", "two percent"]}
        with patch.object(reward, "_evaluation_judge_template", return_value="Verify.\n"), patch.object(
            reward, "_judge_client_and_model", return_value=(client, "judge")
        ):
            result = reward.compute_score("<answer>2 percent</answer>", "2", extra)
        self.assertEqual(result["acc"], 1)
        prompts = [call.kwargs["messages"][0]["content"] for call in client.chat.completions.create.call_args_list]
        self.assertEqual(len(prompts), 2)
        self.assertIn("abs(p-r)/abs(r)", prompts[0])
        self.assertIn("0.05（5%）", prompts[0])
        self.assertIn("不得随意去掉百分号", prompts[0])
        self.assertIn("参考值为 0", prompts[0])
        self.assertNotIn("0.05", prompts[1])
        self.assertNotIn("Alternative 1", prompts[1])

    def test_unresolved_and_truncated_benchmark_judgments_do_not_become_wrong_answers(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content=r"\boxed{Yes}"), finish_reason="length")
        ]
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "", "LLM_AS_A_JUDGE_PRIMARY_RETRIES": "1"}), patch.object(
            reward, "_evaluation_judge_template", return_value="Verify.\n"
        ), patch.object(reward, "_judge_client_and_model", return_value=(client, "judge")):
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                reward.compute_score("<answer>unmatched</answer>", "reference", {"data_source": "visual-agent-ocr"})

    def test_benchmark_failure_messages_are_not_sent_to_judge(self):
        with patch.object(reward, "judge_match", side_effect=AssertionError("judge called")):
            for source in ("visual-agent-ocr", "visual-agent-chartqa"):
                for answer in ("", "Failed to obtain answer via API.", "Agent exceeded the maximum of 8 turns"):
                    result = reward.compute_score(f"<answer>{answer}</answer>", "8", {"data_source": source})
                    self.assertEqual(result["acc"], 0)
