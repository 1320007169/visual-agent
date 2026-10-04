import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "decision_branch_advantage", ROOT / "reinforcement_learning/verl/trainer/ppo/decision_branch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compute = module.compute_decision_branch_advantage

# Response layout: tokens 0-1 are the decision prefix, 2-4 the rest of the trajectory.
DECISION = torch.tensor([1, 1, 0, 0, 0])
RESPONSE = torch.ones(5, dtype=torch.long)


def batch(rows):
    """rows: (uid, decision, forced, reward, decision_logp_per_token)."""
    size = len(rows)
    rewards = torch.zeros(size, 5)
    old_log_probs = torch.zeros(size, 5)
    for i, (_, _, _, reward, logp) in enumerate(rows):
        rewards[i, -1] = reward
        old_log_probs[i, :2] = logp
    return dict(
        token_level_rewards=rewards,
        response_mask=RESPONSE.repeat(size, 1),
        decision_mask=DECISION.repeat(size, 1),
        old_log_probs=old_log_probs,
        index=np.array([row[0] for row in rows], dtype=object),
        decision_ids=np.array([row[1] for row in rows], dtype=object),
        decision_forced=np.array([row[2] for row in rows], dtype=bool),
        decisions=["direct", "ocr_read"],
    )


class DecisionBranchAdvantageTest(unittest.TestCase):
    def setUp(self):
        # Forced spans: log pi(direct) = 2 * log(0.8)/2, log pi(ocr) = 2 * log(0.2)/2.
        self.rows = [
            ("x", "direct", True, 0.0, math.log(0.8) / 2),
            ("x", "ocr_read", True, 1.0, math.log(0.2) / 2),
            ("x", "direct", False, 0.0, 0.0),
            ("x", "ocr_read", False, 1.0, 0.0),
            ("x", "direct", False, 1.0, 0.0),
        ]
        rewards = torch.tensor([row[3] for row in self.rows])
        self.scale = rewards.std() + 1e-6
        self.utility = {"direct": 1 / 3, "ocr_read": 1.0}
        self.value = 0.8 * self.utility["direct"] + 0.2 * self.utility["ocr_read"]

    def test_decision_and_execution_tokens_use_separate_baselines(self):
        advantages, _, metrics = compute(**batch(self.rows))
        for i, (_, decision, forced, reward, _) in enumerate(self.rows):
            execution = (reward - self.utility[decision]) / self.scale
            torch.testing.assert_close(advantages[i, 2:], torch.full((3,), float(execution)))
            decision_adv = (self.utility[decision] - self.value) / self.scale
            if forced:
                # Positive forced decisions are reinforced; negative ones are skipped by default.
                decision_adv = decision_adv if decision_adv > 0 else 0.0
            torch.testing.assert_close(advantages[i, :2], torch.full((2,), float(decision_adv)))
        self.assertAlmostEqual(metrics["decision_branch/pi/direct"], 0.8, places=5)
        self.assertAlmostEqual(metrics["decision_branch/value_mean"], self.value, places=5)
        self.assertAlmostEqual(metrics["decision_branch/oracle_gain_mean"], 1.0 - self.value, places=5)
        entropy = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        self.assertAlmostEqual(metrics["decision_branch/entropy_mean"], entropy, places=5)
        # One prompt: the marginal policy equals the conditional one, so I(X;D) = 0.
        self.assertAlmostEqual(metrics["decision_branch/mutual_information"], 0.0, places=5)
        self.assertEqual(metrics["decision_branch/forced_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["decision_branch/sampled_rate/direct"], 2 / 3, places=5)

    def test_forced_weights_and_decision_weight(self):
        advantages, _, _ = compute(**batch(self.rows), forced_positive_weight=0.5,
                                   forced_negative_weight=1.0, decision_weight=2.0)
        forced_direct = (self.utility["direct"] - self.value) / self.scale * 1.0 * 2.0
        forced_ocr = (self.utility["ocr_read"] - self.value) / self.scale * 0.5 * 2.0
        torch.testing.assert_close(advantages[0, :2], torch.full((2,), float(forced_direct)))
        torch.testing.assert_close(advantages[1, :2], torch.full((2,), float(forced_ocr)))

    def test_incomplete_policy_falls_back_to_mean_baseline(self):
        rows = [row for row in self.rows if not (row[2] and row[1] == "ocr_read")]
        advantages, _, metrics = compute(**batch(rows))
        rewards = torch.tensor([row[3] for row in rows])
        value, scale = rewards.mean(), rewards.std() + 1e-6
        expected = (rewards[rows.index(("x", "ocr_read", False, 1.0, 0.0))] - value) / scale
        torch.testing.assert_close(advantages[2, :2], torch.full((2,), float(expected)))
        self.assertEqual(metrics["decision_branch/pi_coverage_rate"], 0.0)
        self.assertNotIn("decision_branch/entropy_mean", metrics)

    def test_equal_rewards_give_zero_advantages(self):
        # Uneven pi makes float32 sum(pi) != 1; the epsilon scale must not amplify that residue.
        rows = [("z", "direct", True, 1.0, math.log(0.37) / 2), ("z", "ocr_read", True, 1.0, math.log(0.63) / 2),
                ("z", "direct", False, 1.0, 0.0), ("z", "ocr_read", False, 1.0, 0.0)]
        advantages, _, _ = compute(**batch(rows))
        torch.testing.assert_close(advantages, torch.zeros(4, 5), atol=1e-6, rtol=0)

    def test_singleton_decision_gets_no_execution_gradient(self):
        rows = [("y", "direct", True, 0.0, 0.0), ("y", "ocr_read", True, 1.0, 0.0)]
        advantages, _, metrics = compute(**batch(rows))
        torch.testing.assert_close(advantages[:, 2:], torch.zeros(2, 3))
        self.assertEqual(metrics["decision_branch/singleton_decision_rate"], 1.0)

    def test_invalid_rewards_and_masked_rows_are_excluded(self):
        values = batch(self.rows)
        values["response_mask"][4] = 0
        values["decision_mask"][4] = 0
        advantages, _, _ = compute(**values, reward_valid_mask=[True, True, True, False, True])
        torch.testing.assert_close(advantages[3:], torch.zeros(2, 5))
        # Remaining rows: forced direct 0, forced ocr 1, sampled direct 0, so U(direct) = 0.
        torch.testing.assert_close(advantages[2, 2:], torch.zeros(3))
        torch.testing.assert_close(advantages[0, 2:], torch.zeros(3))
        self.assertGreater(float(advantages[1, 0]), 0.0)

    def test_groups_are_independent_and_mutual_information_uses_marginal(self):
        rows = [
            ("a", "direct", True, 1.0, math.log(0.9) / 2), ("a", "ocr_read", True, 0.0, math.log(0.1) / 2),
            ("a", "direct", False, 1.0, 0.0),
            ("b", "direct", True, 0.0, math.log(0.1) / 2), ("b", "ocr_read", True, 1.0, math.log(0.9) / 2),
            ("b", "ocr_read", False, 1.0, 0.0),
        ]
        _, _, metrics = compute(**batch(rows))
        conditional = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        self.assertAlmostEqual(metrics["decision_branch/entropy_mean"], conditional, places=5)
        self.assertAlmostEqual(metrics["decision_branch/mutual_information"], math.log(2) - conditional, places=5)


if __name__ == "__main__":
    unittest.main()
