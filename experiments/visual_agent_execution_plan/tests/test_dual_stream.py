import unittest
from pathlib import Path

from visual_agent_experiments.dual_stream import combine_stream_losses, grouped_outcome_advantages


class DualStreamTest(unittest.TestCase):
    def test_native_prompt_matches_strict_reward_contract(self):
        repo_root = Path(__file__).resolve().parents[3]
        prompt = (repo_root / "prompts" / "visual_agent_native_system.txt").read_text(encoding="utf-8")

        self.assertIn("<answer>", prompt)
        self.assertIn("</answer>", prompt)
        self.assertIn("nothing else", prompt)
        self.assertIn("without calling tools", prompt)

    def test_vision_opd_native_prompt_matches_mcq_reward_contract(self):
        repo_root = Path(__file__).resolve().parents[3]
        prompt = (repo_root / "prompts" / "visual_agent_native_system_mcq.txt").read_text(encoding="utf-8")

        self.assertIn("option letter", prompt)
        self.assertIn("<answer>A</answer>", prompt)
        self.assertIn("nothing else", prompt)
        self.assertIn("without calling tools", prompt)

    def test_advantages_stay_inside_each_stream_group(self):
        advantages = grouped_outcome_advantages(
            [0.0, 2.0, 5.0, 9.0],
            ["a-agent-b1", "a-agent-b1", "a-native-b1", "a-native-b1"],
            ["agent", "agent", "native", "native"],
            normalize_std=False,
        )
        self.assertEqual(advantages, [-1.0, 1.0, -2.0, 2.0])

    def test_mixed_stream_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixes streams"):
            grouped_outcome_advantages([0.0, 1.0], ["bad", "bad"], ["agent", "native"])

    def test_losses_are_aggregated_before_one_update(self):
        self.assertEqual(combine_stream_losses(2.0, 4.0, alpha=1.0), 3.0)
        self.assertEqual(combine_stream_losses(2.0, 4.0, alpha=0.5), 8.0 / 3.0)
