import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


REWARD_PATH = (
    Path(__file__).parents[1]
    / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
)
SPEC = importlib.util.spec_from_file_location("visual_agent_thyme_reward", REWARD_PATH)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward)


class VisualAgentThymeRewardTest(unittest.TestCase):
    def test_think_action_protocol(self):
        final = '<think>The target is below the anchor.</think><answer>below</answer>'
        call = '<think>Inspect the target.</think><tool_call>{"name":"crop_zoom","arguments":{"target_image":0}}</tool_call>'
        transcript = call + '<|im_end|>\n<|im_start|>user\n<tool_response>observation</tool_response><|im_end|>\n<|im_start|>assistant\n' + final + '<|im_end|>'
        with patch.dict(os.environ, {"VISUAL_AGENT_REQUIRE_THINK": "1"}):
            for valid in [final, transcript]:
                self.assertTrue(reward.has_strict_answer_format(valid))
                result = reward.compute_score(valid, "below")
                self.assertEqual(result['format'], 1.0)
                self.assertEqual(result['score'], 1.0)
            for invalid in ['<answer>below</answer>', '<think>only thinking</think>',
                            '<think></think><answer>below</answer>', final + 'extra',
                            final + final, call, transcript.replace('<think>Inspect the target.</think>', ''),
                            final.replace('The target is below the anchor.', '<tool_call>{}</tool_call>')]:
                self.assertFalse(reward.has_strict_answer_format(invalid), invalid)

    def test_initialization_failure_falls_back(self):
        backup = Mock()
        backup.chat.completions.create.return_value.choices = [Mock(message=Mock(content="TRUE"))]
        before = reward.judge_stats_snapshot()
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_PRIMARY_RETRIES": "1", "LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1"}), patch.object(
            reward, "_judge_client_and_model", side_effect=[TimeoutError(), (backup, "model")]
        ):
            self.assertTrue(reward.judge_match("q", "p", "g"))
        stats = reward.judge_batch_summary(before, 1)
        self.assertEqual(stats["primary_timeouts"], 1)
        self.assertEqual(stats["primary_requests"], 0)
        self.assertEqual(stats["fallbacks"], 1)
        self.assertEqual(stats["backup_requests"], 1)

    def test_initialization_failures_are_bounded(self):
        before = reward.judge_stats_snapshot()
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_PRIMARY_RETRIES": "1", "LLM_AS_A_JUDGE_BACKUP_RETRIES": "2", "LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1"}), patch.object(
            reward, "_judge_client_and_model", side_effect=TimeoutError()
        ) as initialize:
            self.assertIsNone(reward.judge_match("q", "p", "g"))
        self.assertEqual(initialize.call_count, 3)
        stats = reward.judge_batch_summary(before, 1)
        self.assertEqual(stats["backup_errors"], 2)
        self.assertEqual(stats["final_unresolved"], 1)

    def test_choice_punctuation_without_judge(self):
        for source in ("visual-agent-hrbench4k", "visual-agent-vision-opd"):
            with patch.object(reward, "judge_match") as judge:
                for answer in ("B", "B.", "B)", "B. text", "B) text"):
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", "B", {"data_source": source})["acc"], 1)
                for answer in ("A.", "A)", "B or C", "blue"):
                    self.assertEqual(reward.compute_score(f"<answer>{answer}</answer>", "B", {"data_source": source})["acc"], 0)
                judge.assert_not_called()

    def test_judge_stats_distinguish_backup_false_from_failure(self):
        primary, backup = Mock(), Mock()
        error = RuntimeError("limited")
        error.status_code = 429
        primary.chat.completions.create.side_effect = error
        backup.chat.completions.create.return_value.choices = [Mock(message=Mock(content="FALSE"))]
        before = reward.judge_stats_snapshot()
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1", "LLM_AS_A_JUDGE_BACKUP_RETRIES": "2"}), patch.object(
            reward, "_judge_client_and_model", side_effect=[(primary, "primary"), (backup, "backup")]
        ):
            self.assertFalse(reward.judge_match("color?", "blue", "red"))
        stats = reward.judge_batch_summary(before, 8)
        self.assertEqual(stats["primary_requests"], 1)
        self.assertEqual(stats["primary_rate_limited"], 1)
        self.assertEqual(stats["backup_requests"], 1)
        self.assertEqual(stats["backup_valid"], 1)
        self.assertEqual(stats["backup_false"], 1)
        self.assertEqual(stats["final_unresolved"], 0)
        self.assertEqual(stats["judge_sample_ratio"], 1 / 8)

    def test_judge_stats_track_empty_response_as_unresolved(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [Mock(message=Mock(content=None))]
        before = reward.judge_stats_snapshot()
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "", "LLM_AS_A_JUDGE_PRIMARY_RETRIES": "1"}), patch.object(
            reward, "_judge_client_and_model", return_value=(client, "model")
        ):
            self.assertIsNone(reward.judge_match("color?", "red", "red"))
        stats = reward.judge_batch_summary(before, 1)
        self.assertEqual(stats["primary_invalid"], 1)
        self.assertEqual(stats["primary_valid"], 0)
        self.assertEqual(stats["final_unresolved"], 1)
        self.assertEqual(stats["final_unresolved_ratio"], 1)

    def test_judge_stats_are_thread_safe_and_batch_local(self):
        from concurrent.futures import ThreadPoolExecutor

        client = Mock()
        client.chat.completions.create.return_value.choices = [Mock(message=Mock(content="TRUE"))]
        before = reward.judge_stats_snapshot()
        with patch.object(reward, "_judge_client_and_model", return_value=(client, "model")):
            with ThreadPoolExecutor(max_workers=32) as pool:
                self.assertTrue(all(pool.map(lambda _: reward.judge_match("q", "a", "a"), range(128))))
        stats = reward.judge_batch_summary(before, 896)
        self.assertEqual(stats["judge_samples"], 128)
        self.assertEqual(stats["primary_requests"], 128)
        self.assertEqual(stats["primary_valid"], 128)
        self.assertEqual(stats["primary_true"], 128)
        self.assertEqual(reward.judge_batch_summary(reward.judge_stats_snapshot(), 896)["judge_samples"], 0)

    def test_judge_falls_back_on_transient_failure(self):
        for status in (None, 408, 429, 500, 503):
            with self.subTest(status=status):
                primary, backup = Mock(), Mock()
                error = RuntimeError("provider failure")
                error.status_code = status
                primary.chat.completions.create.side_effect = error
                backup.chat.completions.create.return_value.choices = [
                    Mock(message=Mock(content="TRUE"))
                ]
                with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1", "LLM_AS_A_JUDGE_BACKUP_RETRIES": "2"}), patch.object(
                    reward, "_judge_client_and_model", side_effect=[(primary, "primary"), (backup, "backup")]
                ):
                    self.assertTrue(reward.judge_match("color?", "red", "red"))
                self.assertEqual(primary.chat.completions.create.call_count, 1)
                self.assertEqual(backup.chat.completions.create.call_count, 1)

    def test_judge_falls_back_on_invalid_response(self):
        primary, backup = Mock(), Mock()
        primary.chat.completions.create.return_value.choices = [Mock(message=Mock(content="maybe"))]
        backup.chat.completions.create.return_value.choices = [Mock(message=Mock(content="TRUE"))]
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1"}), patch.object(
            reward, "_judge_client_and_model", side_effect=[(primary, "primary"), (backup, "backup")]
        ):
            self.assertTrue(reward.judge_match("color?", "red", "red"))
        self.assertEqual(primary.chat.completions.create.call_count, 1)
        self.assertEqual(backup.chat.completions.create.call_count, 1)

    def test_judge_false_does_not_fall_back(self):
        primary = Mock()
        primary.chat.completions.create.return_value.choices = [Mock(message=Mock(content="FALSE"))]
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1"}), patch.object(
            reward, "_judge_client_and_model", return_value=(primary, "primary")
        ) as get_client:
            self.assertFalse(reward.judge_match("color?", "blue", "red"))
            get_client.assert_called_once_with(backup=False)

    def test_judge_both_fail_is_bounded(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError()
        with patch.dict(os.environ, {"LLM_AS_A_JUDGE_BACKUP_BASE": "https://backup/v1", "LLM_AS_A_JUDGE_BACKUP_RETRIES": "2"}), patch.object(
            reward, "_judge_client_and_model", return_value=(client, "model")
        ):
            self.assertIsNone(reward.judge_match("color?", "red", "red"))
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_hrbench_uses_choice_accuracy_without_judge(self):
        extra = {"data_source": "visual-agent-hrbench4k"}
        with patch.object(reward, "judge_match") as judge:
            self.assertEqual(reward.compute_score("<answer>B</answer>", "B", extra)["acc"], 1)
            self.assertEqual(reward.compute_score("<answer>B. 37B</answer>", "B", extra)["acc"], 1)
            self.assertEqual(reward.compute_score("<answer>B) 37B</answer>", "B", extra)["acc"], 1)
            self.assertEqual(reward.compute_score("<answer>A</answer>", "B", extra)["acc"], 0)
            self.assertEqual(reward.compute_score("<answer>A. 37B</answer>", "B", extra)["acc"], 0)
            self.assertEqual(reward.compute_score("<answer>blue</answer>", "B", extra)["acc"], 0)
            self.assertEqual(reward.compute_score("<answer>B or C</answer>", "B", extra)["acc"], 0)
            judge.assert_not_called()

    def test_mixed_deepeyes_free_form_uses_judge_without_changing_relation_reward(self):
        extra = {"source": "deepeyesv2/perception", "question": "What color is the car?"}
        with patch.object(reward, "rule_match", return_value=False), patch.object(
            reward, "judge_match", return_value=True
        ) as judge:
            result = reward.compute_score("<answer>red</answer>", "The car is red.", extra)
            self.assertEqual(result["acc"], 1.0)
            judge.assert_called_once_with("What color is the car?", "red", "The car is red.")
            judge.reset_mock()
            result = reward.compute_score("<answer>on</answer>", "above", {
                "source": "zwz_rl_vqa/original_images"
            })
            self.assertEqual(result["acc"], 0.0)
            judge.assert_not_called()

    def test_unresolved_judge_keeps_format_reward_and_enters_grpo(self):
        extra = {"source": "deepeyesv2/perception", "question": "What color is the car?"}
        with patch.object(reward, "rule_match", return_value=False), patch.object(
            reward, "judge_match", return_value=None
        ):
            result = reward.compute_score("<answer>red</answer>", "The car is red.", extra)
        self.assertEqual(result["score"], 0.1)
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["reward_valid"], 1.0)

    def test_vision_opd_uses_closed_multiple_choice_reward(self):
        extra_info = {
            "source": "vision-opd/original_images",
            "data_source": "visual-agent-vision-opd",
        }
        self.assertEqual(
            reward.compute_score("<answer>D</answer>", "D", extra_info=extra_info),
            {"score": 1.0, "acc": 1.0, "format": 1.0, "tool_used": 0.0, "reward_valid": 1.0},
        )
        self.assertEqual(
            reward.compute_score("<answer>C</answer>", "D", extra_info=extra_info)["acc"],
            0.0,
        )
        self.assertEqual(
            reward.compute_score("<answer>D. Purple</answer>", "D", extra_info=extra_info)["acc"],
            1.0,
        )
        self.assertEqual(
            reward.compute_score("<answer>purple</answer>", "D", extra_info=extra_info)["acc"],
            0.0,
        )

    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {"LLM_AS_A_JUDGE_BASE": "", "LLM_AS_A_JUDGE_BACKUP_BASE": ""},
        )
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
                "reward_valid": 1.0,
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
