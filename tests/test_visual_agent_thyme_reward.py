import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


REWARD_PATH = (
    Path(__file__).parents[1]
    / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
)
SPEC = importlib.util.spec_from_file_location("visual_agent_thyme_reward", REWARD_PATH)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward)


class VisualAgentThymeRewardTest(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {"LLM_AS_A_JUDGE_BASE": ""})
        self.env_patch.start()
        reward._judge_client_and_model.cache_clear()

    def tearDown(self):
        self.env_patch.stop()
        reward._judge_client_and_model.cache_clear()

    def test_strict_answer_only_format(self):
        self.assertEqual(
            reward.compute_score("<answer>right of</answer>", "right of"),
            {
                "score": 1.0,
                "acc": 1.0,
                "format": 1.0,
                "tool_used": 0.0,
            },
        )

    def test_strict_format_accepts_final_answer_after_tool(self):
        solution = (
            '<tool_call>{"name":"grounding_detect"}</tool_call> '
            'user <tool_response>{"boxes": []}</tool_response> '
            "assistant <answer>below</answer>"
        )

        result = reward.compute_score(solution, "below")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format"], 1.0)
        self.assertEqual(result["tool_used"], 1.0)

    def test_repeated_answers_lose_format_reward(self):
        result = reward.compute_score(
            "<answer>left of</answer> repeated <answer>right of</answer>",
            "right of",
        )

        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format"], 0.0)

    def test_reasoning_before_final_answer_keeps_format_reward(self):
        result = reward.compute_score(
            "The object is lower. <answer>below</answer>", "below"
        )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format"], 1.0)

    def test_text_after_final_answer_loses_format_reward(self):
        result = reward.compute_score(
            "<answer>below</answer> trailing text", "below"
        )

        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format"], 0.0)
