"""CPU contract tests for rollout code, without Ray/vLLM runtime imports."""

import ast
import asyncio
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3] / "verl"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name][-1]
    node.returns = None
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class MultiturnTrainingContracts(unittest.TestCase):
    def test_observation_masks(self):
        mask_fn = load_function(ROOT / "workers/rollout/chat_scheduler.py", "_mask_out_tools_calling_tokens", {"itertools": itertools})
        assistant = {"role": "assistant", "content": "<think>Inspect the object.</think><tool_call>CALL</tool_call>"}
        xml = {"role": "user", "content": "<tool_response>RESULT</tool_response>"}
        image = {"role": "user", "content": [{"type": "text", "text": xml["content"]}, {"type": "image_url", "image_url": "crop"}]}
        native = {"role": "tool", "content": "RESULT"}
        for observations, spans in [([xml], 1), ([xml, xml], 2), ([xml, image], 2), ([native, native], 1)]:
            with self.subTest(observations=observations):
                # Each serialized turn has two content tokens and one EOS.
                ids = torch.tensor([[7, 8, 0] * (spans + 2) + [9, 9]])
                attention = torch.ones_like(ids)
                attention[:, -2:] = 0
                result = mask_fn(SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0)), [[]], [[assistant] + observations + [assistant]], ids, attention)
                expected = torch.tensor([[1, 1, 1] + [0] * (3 * spans) + [1, 1, 1, 0, 0]])
                self.assertTrue(torch.equal(result, expected))

    def test_six_assistant_turns(self):
        callback = load_function(ROOT / "workers/rollout/chat_scheduler.py", "__call__", {})
        submissions = []

        async def tool(*args, **kwargs):
            return {"role": "tool", "content": "observation"}

        obj = SimpleNamespace(max_turns=6, _call_tool=tool, scheduler=SimpleNamespace(submit_chat_completions=lambda **kw: submissions.append(kw)))
        completion = SimpleNamespace(id="request", choices=[SimpleNamespace(
            message=SimpleNamespace(model_dump=lambda **kw: {"role": "assistant", "content": ""}, tool_calls=[{}]), finish_reason="tool_calls")])
        messages = [{"role": "system"}, {"role": "user"}]
        info = {}
        callback.__globals__["asyncio"] = asyncio
        for _ in range(6):
            asyncio.run(callback(obj, messages, completion, info))
        self.assertEqual(info["assistant_turns"], 6)
        self.assertEqual(len(submissions), 5)

    def test_think_tool_observation_answer_trajectory(self):
        namespace = {"asyncio": asyncio, "json": json, "re": re,
                     "TOOL_CALL_RE": re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)}
        callback = load_function(ROOT / "workers/rollout/chat_scheduler.py", "__call__", namespace)
        calls, submissions = [], []
        async def tool(call, *args, **kwargs):
            calls.append(call)
            return {"role": "user", "content": "<tool_response>crop</tool_response>"}
        obj = SimpleNamespace(max_turns=6, _call_tool=tool, scheduler=SimpleNamespace(submit_chat_completions=lambda **kw: submissions.append(kw)))
        messages, info = [], {}
        texts = ['<think>Inspect the target.</think><tool_call>{"name":"crop_zoom","arguments":{}}</tool_call>',
                 '<think>The target is visible.</think><answer>below</answer>']
        for text in texts:
            completion = SimpleNamespace(id='test', choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                tool_calls=[], model_dump=lambda **kw: {"role": "assistant", "content": text}))])
            asyncio.run(callback(obj, messages, completion, info))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(submissions), 1)
        self.assertEqual(messages[0]['content'], texts[0])
        self.assertEqual(messages[2]['content'], texts[1])
        self.assertEqual(info['assistant_turns'], 2)

    def test_recompute_image_and_following_text_positions(self):
        rope = load_function(ROOT / "models/transformers/qwen2_vl.py", "get_rope_index", {"torch": torch})
        processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2), tokenizer=SimpleNamespace(
            convert_tokens_to_ids=lambda token: {"<|image_pad|>": 99, "<|video_pad|>": 98, "<|vision_start|>": 97}[token]))
        ids = torch.tensor([[0, 10, 11, 97, 99, 99, 99, 99, 12, 13, 0]])
        attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        batch = SimpleNamespace(batch={"input_ids": ids, "attention_mask": attention, "position_ids": torch.zeros(1, 3, 11)},
                                non_tensor_batch={"multi_modal_inputs": [{"image_grid_thw": torch.tensor([[1, 4, 4]])}]})
        tree = ast.parse((ROOT / "trainer/ppo/ray_trainer.py").read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "batch.batch['position_ids'].dim() == 3")
        # Execute the actual trainer block using the actual local RoPE helper.
        block.body = [n for n in block.body if not isinstance(n, ast.ImportFrom)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "trainer_positions", "exec"),
             {"batch": batch, "self": SimpleNamespace(processor=processor), "torch": torch, "get_rope_index": rope})
        positions = batch.batch["position_ids"][0]
        self.assertEqual(positions[:, 4:8].tolist(), [[3, 3, 3, 3], [3, 3, 4, 4], [3, 4, 3, 4]])
        self.assertEqual(positions[:, 9].tolist(), [6, 6, 6])


if __name__ == "__main__":
    unittest.main()
