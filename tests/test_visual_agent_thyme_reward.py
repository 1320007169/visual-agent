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

    def test_long_grounding_query_is_penalized(self):
        solution = (
            '<tool_call>{"name":"grounding_detect","arguments":'
            '{"query":"one two three four five six seven eight nine ten eleven twelve thirteen",'
            '"target_image":0}}</tool_call> '
            "user <tool_response>{}</tool_response> "
            "assistant <answer>below</answer>"
        )
        with patch.dict(
            os.environ,
            {
                "GROUNDING_QUERY_MAX_WORDS": "12",
                "GROUNDING_QUERY_PENALTY": "0.1",
                "GROUNDING_QUERY_MAX_PENALTY": "0.2",
            },
        ):
            result = reward.compute_score(solution, "below")

        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["invalid_grounding_queries"], 1.0)
        self.assertEqual(result["query_penalty"], 0.1)

    def test_zwz_relation_scoring_never_calls_open_answer_matchers(self):
        metadata = {"source": "zwz_rl_vqa/original_images"}
        with patch.object(reward, "rule_match", side_effect=AssertionError("open rules called")), patch.object(
            reward, "judge_match", side_effect=AssertionError("judge called")
        ):
            for gold in reward.RELATION_LABELS:
                for prediction in ("on", "in", "over", "above or below", "The answer is above"):
                    with self.subTest(gold=gold, prediction=prediction):
                        result = reward.compute_score(f"<answer>{prediction}</answer>", gold, metadata)
                        self.assertEqual(result["acc"], 0.0)
                        self.assertEqual(result["score"], 0.1)
                with self.subTest(gold=gold, prediction=gold):
                    result = reward.compute_score(f"<answer>{gold}</answer>", gold, metadata)
                    self.assertEqual(result["acc"], 1.0)
                    self.assertEqual(result["score"], 1.0)

    def test_zwz_relation_aliases_are_complete_labels(self):
        examples = [
            ("  UNDER.  ", "below"),
            ("beneath", "below"),
            ("to the left of", "left of"),
            ("on the right of", "right of"),
            ("overlapping", "overlap"),
            ("contains", "contain"),
            ("beside", "next to"),
            ("inside of", "inside"),
        ]
        with patch.object(reward, "judge_match", side_effect=AssertionError("judge called")):
            for prediction, gold in examples:
                with self.subTest(prediction=prediction, gold=gold):
                    self.assertEqual(reward.normalize_relation_answer(prediction), gold)
                    result = reward.compute_score(
                        f"<answer>{prediction}</answer>", gold,
                        {"source": "zwz_rl_vqa/original_images"},
                    )
                    self.assertEqual(result["acc"], 1.0)

    def test_relation_matching_rejects_wrong_and_unknown_labels(self):
        self.assertFalse(reward.relation_match("above", "below"))
        self.assertFalse(reward.relation_match("inside", "contain"))
        self.assertFalse(reward.relation_match("on", "on"))
        self.assertFalse(reward.relation_match("above", "unknown"))
        self.assertIsNone(reward.normalize_relation_answer("above the table"))

    def test_explicit_zwz_markers_enable_closed_relation_scoring(self):
        markers = [
            {"source": "zwz_rl_vqa/original_images"},
            {"data_source": "visual-agent-zwz-relation"},
            {"task_type": "zwz_original_relation"},
        ]
        with patch.object(reward, "judge_match", side_effect=AssertionError("judge called")):
            for metadata in markers:
                with self.subTest(metadata=metadata):
                    result = reward.compute_score("<answer>above</answer>", "below", metadata)
                    self.assertEqual(result["acc"], 0.0)

    def test_open_tasks_keep_semantic_judge_fallback(self):
        markers = [None, {"source": "visual-agent-thyme"}, {"ability": "spatial_relation"}]
        for metadata in markers:
            with self.subTest(metadata=metadata), patch.object(
                reward, "rule_match", return_value=False
            ), patch.object(reward, "judge_match", return_value=True) as judge:
                result = reward.compute_score("<answer>on</answer>", "above", metadata)
                self.assertEqual(result["acc"], 1.0)
                judge.assert_called_once_with("", "on", "above")
