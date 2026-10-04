import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "decision_branch_rollout", ROOT / "reinforcement_learning/verl/workers/rollout/decision_branch.py"
)
branch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(branch)

TOOL_PREFIX = '<tool_call>\n{"name": "ocr_read", "arguments":'


def completion(content, finish_reason="stop"):
    return {"choices": [{"index": 0, "finish_reason": finish_reason, "logprobs": {"content": []},
                         "message": {"role": "assistant", "content": content}}]}


class DecisionBranchHelpersTest(unittest.TestCase):
    def test_forced_tool_call_is_parsed_like_hermes(self):
        data = branch.merge_forced_completion(
            completion(' {"target_image": 0, "mode": "chart"}}\n</tool_call>'), TOOL_PREFIX
        )
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["logprobs"])
        self.assertIsNone(choice["message"]["content"])
        [call] = choice["message"]["tool_calls"]
        self.assertEqual(call["function"]["name"], "ocr_read")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"target_image": 0, "mode": "chart"})
        self.assertEqual(call["function"]["arguments"], '{"target_image": 0, "mode": "chart"}')

    def test_unparsable_forced_call_stays_text_like_hermes(self):
        # Invalid JSON, and valid JSON without "arguments" (KeyError in hermes).
        for continuation in (" {bad json\n</tool_call>", '}\n</tool_call>'):
            data = branch.merge_forced_completion(completion(continuation, "length"), '<tool_call>\n{"name": "ocr_read"')
            message = data["choices"][0]["message"]
            self.assertNotIn("tool_calls", message)
            self.assertEqual(message["content"], '<tool_call>\n{"name": "ocr_read"' + continuation)
            self.assertEqual(data["choices"][0]["finish_reason"], "length")

    def test_forced_direct_answer_keeps_text(self):
        data = branch.merge_forced_completion(completion(">B</answer>"), branch.DIRECT_PREFIX)
        self.assertEqual(data["choices"][0]["message"]["content"], "<answer>B</answer>")
        self.assertNotIn("tool_calls", data["choices"][0]["message"])

    def test_first_decision_classification(self):
        decisions = ["direct", "ocr_read", "crop_zoom"]
        native = {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "crop_zoom", "arguments": "{}"}}, {"function": {"name": "ocr_read", "arguments": "{}"}}]}
        self.assertEqual(branch.first_decision([native, {"role": "tool", "content": "x"}], decisions), "crop_zoom")
        xml = {"role": "assistant", "content": '<tool_call>{"name": "ocr_read", "arguments": {}}</tool_call>'}
        self.assertEqual(branch.first_decision([xml], decisions), "ocr_read")
        self.assertEqual(branch.first_decision([{"role": "assistant", "content": "<answer>B</answer>"}], decisions), "direct")
        unknown = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "rotate", "arguments": "{}"}}]}
        self.assertEqual(branch.first_decision([unknown], decisions), "other")
        broken = {"role": "assistant", "content": "<tool_call>{oops</tool_call>"}
        self.assertEqual(branch.first_decision([broken], decisions), "other")
        self.assertEqual(branch.first_decision([], decisions), "other")

    def test_decision_mask_only_searches_the_first_turn(self):
        eos = 9
        ids = torch.tensor([5, 1, 2, 3, eos, 1, 2, 3, eos])
        torch.testing.assert_close(branch.decision_mask(ids, [1, 2], eos), torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0]))
        torch.testing.assert_close(branch.decision_mask(ids, [7], eos), torch.zeros(9, dtype=torch.long))
        torch.testing.assert_close(branch.decision_mask(ids, None, eos), torch.zeros(9, dtype=torch.long))

    def test_forced_assignment_keeps_on_policy_samples(self):
        plan = branch.assign_forced_decisions(8, ["direct", "ocr_read", "crop_zoom"], 2)
        self.assertEqual(plan, ["direct", "direct", "ocr_read", "ocr_read", "crop_zoom", "crop_zoom", None, None])
        with self.assertRaisesRegex(ValueError, "must exceed"):
            branch.assign_forced_decisions(6, ["direct", "ocr_read", "crop_zoom"], 2)


class RealTokenizerDecisionBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        model = Path(os.environ.get("QWEN3_VL_PROCESSOR_PATH", ROOT.parent / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        if not (model / "tokenizer.json").is_file():
            raise unittest.SkipTest("Set QWEN3_VL_PROCESSOR_PATH to local Qwen3-VL tokenizer/processor files")
        cls.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        cls.tokenizer = cls.processor.tokenizer
        cls.tools = ["grounding_detect", "crop_zoom", "depth_measure", "object_count", "ocr_read"]
        cls.prefixes = branch.decision_prefixes(cls.tokenizer, cls.tools)
        cls.prompt = [{"role": "user", "content": "q"}]
        cls.generation_prompt = cls.tokenizer.apply_chat_template(cls.prompt, add_generation_prompt=True, tokenize=False)

    def rendered_turn_ids(self, message):
        text = self.tokenizer.apply_chat_template([*self.prompt, message], tokenize=False)
        return self.tokenizer.encode(text[len(self.generation_prompt):], add_special_tokens=False)

    def test_prefixes_end_on_token_boundaries(self):
        arguments = [{"query": "red cup", "target_image": 0}, {"bbox_2d": [1, 2, 3, 4], "target_image": 1},
                     {"target_image": 0, "mode": "chart"}, {"query": "中文招牌", "target_image": 0}]
        for name in self.tools:
            prefix_ids = self.tokenizer.encode(self.prefixes[name], add_special_tokens=False)
            for args in arguments:
                call = {"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                ids = self.rendered_turn_ids({"role": "assistant", "content": "", "tool_calls": [call]})
                self.assertEqual(ids[: len(prefix_ids)], prefix_ids, (name, args))
        direct_ids = self.tokenizer.encode(branch.DIRECT_PREFIX, add_special_tokens=False)
        for answer in ("B", "3.5", "left of", "12", "是"):
            ids = self.rendered_turn_ids({"role": "assistant", "content": f"<answer>{answer}</answer>"})
            self.assertEqual(ids[: len(direct_ids)], direct_ids, answer)

    def test_forced_and_sampled_calls_render_identically_with_decision_mask(self):
        sys.path.insert(0, str(ROOT / "reinforcement_learning"))
        sys.path.insert(0, str(ROOT / "tests"))
        from test_rl_response_budget import scheduler_definitions

        args = {"target_image": 0, "mode": "chart"}
        forced = branch.merge_forced_completion(
            completion(" " + json.dumps(args) + "}\n</tool_call>"), self.prefixes["ocr_read"]
        )["choices"][0]["message"]
        forced = {key: value for key, value in forced.items() if value is not None}
        forced.setdefault("content", "")
        # What the callback stores for a sampled hermes call of the same tool and arguments.
        sampled = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "chatcmpl-tool-0", "type": "function",
            "function": {"name": "ocr_read", "arguments": json.dumps(args, ensure_ascii=False)}}]}
        observation = {"role": "tool", "content": '{"text": "42", "truncated": false}', "tool_call_id": "x"}
        answer = {"role": "assistant", "content": "<answer>42</answer>"}

        context = scheduler_definitions()
        callback = context["ToolCompletionCallback"]()
        callback.tokenizer, callback.processor, callback.tool_schemas = self.tokenizer, self.processor, None
        callback.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            response_length=512, multi_turn={"overlong_masking": True})))
        callback.scheduler = SimpleNamespace(decision_branch={
            "decisions": ["direct", *self.tools],
            "prefix_ids": {name: self.tokenizer.encode(prefix, add_special_tokens=False) for name, prefix in self.prefixes.items()},
        })
        prompt_ids = self.tokenizer(self.generation_prompt, return_tensors="pt", add_special_tokens=False)
        raw = np.empty(1, dtype=object)
        raw[0] = self.prompt
        batch = SimpleNamespace(batch=TensorDict({
            "input_ids": prompt_ids["input_ids"], "attention_mask": prompt_ids["attention_mask"],
            "position_ids": torch.arange(prompt_ids["input_ids"].shape[-1]).unsqueeze(0)}, [1]),
            non_tensor_batch={"raw_prompt": raw})
        conversations = [
            [*copy.deepcopy(self.prompt), message, observation, answer]
            for message in (forced, sampled, {"role": "assistant", "content": "<answer>7</answer>"})
        ]
        output = callback.postprocess(batch, conversations, n=3, forced_decisions=["ocr_read", None, None])

        responses, masks = output.batch["responses"], output.batch["decision_mask"]
        torch.testing.assert_close(responses[0], responses[1])
        torch.testing.assert_close(masks[0], masks[1])
        prefix_ids = callback.scheduler.decision_branch["prefix_ids"]["ocr_read"]
        self.assertEqual(responses[0][masks[0].bool()].tolist(), prefix_ids)
        self.assertEqual(masks[0][: len(prefix_ids)].tolist(), [1] * len(prefix_ids))
        direct_ids = callback.scheduler.decision_branch["prefix_ids"]["direct"]
        self.assertEqual(responses[2][masks[2].bool()].tolist(), direct_ids)
        self.assertEqual(output.non_tensor_batch["decision_id"].tolist(), ["ocr_read", "ocr_read", "direct"])
        self.assertEqual(output.non_tensor_batch["decision_forced"].tolist(), [True, False, False])
        # Decision tokens are always trainable assistant tokens.
        self.assertTrue(bool((masks <= output.batch["response_mask"]).all()))


if __name__ == "__main__":
    unittest.main()
