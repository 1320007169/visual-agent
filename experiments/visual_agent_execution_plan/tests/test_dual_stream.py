import unittest

from visual_agent_experiments.dual_stream import combine_stream_losses, grouped_outcome_advantages


class DualStreamTest(unittest.TestCase):
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
