import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rollout_consistency", ROOT / "reinforcement_learning/verl/workers/rollout/rollout_consistency.py"
)
consistency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consistency)

EOS = 9
HEADER = [7, 8]  # "<|im_start|>assistant\n"
NEWLINE = 6


class RolloutConsistencyTest(unittest.TestCase):
    def test_parse_token_id_logprobs(self):
        payload = SimpleNamespace(content=[SimpleNamespace(token="token_id:5", logprob=-0.5),
                                           SimpleNamespace(token="token_id:9", logprob=-0.1)])
        self.assertEqual(consistency.parse_sampled_logprobs(payload), ([5, 9], [-0.5, -0.1]))
        self.assertIsNone(consistency.parse_sampled_logprobs(None))
        self.assertIsNone(consistency.parse_sampled_logprobs(SimpleNamespace(content=None)))
        text_tokens = SimpleNamespace(content=[SimpleNamespace(token="hello", logprob=-0.5)])
        self.assertIsNone(consistency.parse_sampled_logprobs(text_tokens))

    def test_adjacent_tool_messages_share_one_turn(self):
        messages = [{"role": "assistant"}, {"role": "tool"}, {"role": "tool"}, {"role": "assistant"},
                    {"role": "user"}, {"role": "user"}, {"role": "assistant"}]
        self.assertEqual(consistency.template_turn_roles(messages),
                         ["assistant", "tool", "assistant", "user", "user", "assistant"])

    def test_matching_turns_map_to_training_positions(self):
        # assistant(1,2,EOS) observation(NL,3,EOS) assistant(NL,HEADER,4,5,EOS) NL
        response = [1, 2, EOS, NEWLINE, 3, EOS, NEWLINE, *HEADER, 4, 5, EOS, NEWLINE]
        roles = ["assistant", "tool", "assistant"]
        sampled = [([1, 2, EOS], [-0.1, -0.2, -0.3]), ([4, 5], [-0.4, -0.5])]
        positions, values, stats = consistency.align_sampled_turns(response, roles, EOS, HEADER, sampled)
        self.assertEqual(positions, [0, 1, 2, 9, 10])
        self.assertEqual(values, [-0.1, -0.2, -0.3, -0.4, -0.5])
        self.assertEqual(stats, {"turns": 2, "matched_turns": 2, "sampled_tokens": 5,
                                 "matched_tokens": 5, "unaligned": 0})

    def test_rerendered_turn_is_reported_not_mapped(self):
        response = [1, 2, EOS, NEWLINE, 3, EOS, NEWLINE, *HEADER, 4, 5, EOS]
        roles = ["assistant", "tool", "assistant"]
        # The first turn was re-rendered differently from what vLLM sampled.
        sampled = [([1, 20, EOS], [-0.1, -0.2, -0.3]), ([4, 5, EOS], [-0.4, -0.5, -0.6])]
        positions, values, stats = consistency.align_sampled_turns(response, roles, EOS, HEADER, sampled)
        self.assertEqual(positions, [9, 10, 11])
        self.assertEqual(values, [-0.4, -0.5, -0.6])
        self.assertEqual((stats["matched_turns"], stats["turns"]), (1, 2))
        self.assertEqual((stats["matched_tokens"], stats["sampled_tokens"]), (3, 6))

    def test_budget_dropped_turns_use_sampled_prefix(self):
        response = [1, 2, EOS, NEWLINE, 3, EOS]
        roles = ["assistant", "tool"]
        sampled = [([1, 2, EOS], [-0.1, -0.2, -0.3]), ([4, 5], [-0.4, -0.5])]
        positions, _, stats = consistency.align_sampled_turns(response, roles, EOS, HEADER, sampled)
        self.assertEqual(positions, [0, 1, 2])
        self.assertEqual(stats["turns"], 1)

    def test_missing_logprobs_and_unaligned_terminators(self):
        response = [1, 2, EOS]
        _, _, stats = consistency.align_sampled_turns(response, ["assistant"], EOS, HEADER, [None])
        self.assertEqual((stats["turns"], stats["matched_turns"], stats["sampled_tokens"]), (1, 0, 0))
        _, _, stats = consistency.align_sampled_turns([1, 2], ["assistant"], EOS, HEADER, [([1, 2], [0.0, 0.0])])
        self.assertEqual(stats["unaligned"], 1)


if __name__ == "__main__":
    unittest.main()
