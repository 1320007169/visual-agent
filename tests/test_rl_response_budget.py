import ast
import __future__
import asyncio
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
from functools import partial
import importlib.util
import io
import itertools
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "reinforcement_learning/verl"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget_module = load_module("response_budget", VERL / "workers/rollout/response_budget.py")
verification = load_module("verify_rl_padding", ROOT / "scripts/verify_rl_padding.py")


def definitions(path, functions=(), classes=None, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            node.decorator_list = []
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and classes and node.name in classes:
            node.bases = []
            node.decorator_list = []
            node.body = [method for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in classes[node.name]]
            for method in node.body:
                method.decorator_list = []
            nodes.append(node)
    context = {"torch": torch, "np": np, "Optional": Optional, "Dict": Dict, "Any": Any,
               "List": List, "defaultdict": defaultdict, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), context)
    return context


def scheduler_definitions():
    context = definitions(
        VERL / "workers/rollout/chat_scheduler.py",
        functions=("_collect_message_images", "_encode_response_with_images"),
        classes={"ToolCompletionCallback": {"postprocess", "_mask_out_tools_calling_tokens", "__call__"},
                 "ChatCompletionScheduler": {"submit_chat_completions", "_submit_chat_completions_semaphore"}},
        namespace={"TensorDict": TensorDict, "DataProto": SimpleNamespace,
                   "ResponseBudget": budget_module.ResponseBudget, "itertools": itertools,
                   "asyncio": asyncio, "json": json, "time": time, "logger": logging.getLogger(__name__)},
    )
    return context


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m.get('content', '')}</end>" for m in messages)
        return text + ("<assistant>" if kwargs.get("add_generation_prompt") else "")


def text_encode(tokenizer, processor, text, images, cache):
    ids = torch.arange(len(text), dtype=torch.long)
    return ids, torch.ones_like(ids), {}


def distributed_update_worker(rank, rendezvous):
    from datetime import timedelta
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.py_functional import append_to_dict

    torch.set_num_threads(1)
    torch.manual_seed(9)
    torch.distributed.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                                        world_size=2, timeout=timedelta(seconds=60))
    try:
        funcs = verification.load_loss_functions()
        context = definitions(VERL / "workers/actor/dp_actor.py",
            classes={"DataParallelPPOActor": {"update_policy"}},
            namespace={"DataProto": DataProto, "get_device_id": lambda: "cpu", "append_to_dict": append_to_dict,
                       **{name: funcs[name] for name in ("agg_loss", "compute_policy_loss", "kl_penalty")}})
        actor = context["DataParallelPPOActor"]()
        forward, _ = verification.load_actor_forward()
        actor._forward_micro_batch = forward._forward_micro_batch.__get__(actor)
        actor.actor_module = torch.nn.parallel.DistributedDataParallel(verification.TinyCausalModel())
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=0.01, weight_decay=0.1)
        actor.config = OmegaConf.create({"ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1, "use_dynamic_bsz": False, "use_kl_loss": True, "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001, "clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None,
            "entropy_coeff": 0.01, "loss_agg_mode": "token-mean", "crop_common_padding": True})
        actor.device_name = "cpu"
        actor.use_remove_padding = actor.use_fused_kernels = False
        counts = [0]
        actor.actor_module.register_forward_hook(lambda *args: counts.__setitem__(0, counts[0] + 1))
        def optimizer_step():
            norm = torch.nn.utils.clip_grad_norm_(actor.actor_module.parameters(), 1.0)
            actor.actor_optimizer.step()
            return norm
        actor._optimizer_step = optimizer_step
        raw = verification.make_batch(actor.actor_module.module)
        tensors = {k: value.repeat(2, *([1] * (value.ndim - 1))) for k, value in raw.items() if torch.is_tensor(value)}
        shape = tensors["responses"].shape
        tensors.update(old_log_probs=torch.zeros(shape), advantages=torch.ones(shape), ref_log_prob=torch.zeros(shape))
        mask = tensors["attention_mask"].clone()
        mask[:, :-shape[-1]] = 0
        if rank == 0:
            mask[:] = 0
        tensors["loss_mask"] = mask
        mm = np.empty(2, dtype=object)
        mm[:] = [{}, {}]
        batch = DataProto(batch=TensorDict(tensors, [2]), non_tensor_batch={"multi_modal_inputs": mm},
                         meta_info={"multi_turn": True, "temperature": 1.0})
        metrics = actor.update_policy(batch)
        assert metrics["actor/optimizer_step_per_minibatch"] == [1.0]
        assert all(np.isfinite(value).all() for value in metrics.values())
        before = [parameter.detach().clone() for parameter in actor.actor_module.parameters()]
        batch.batch["loss_mask"].zero_()
        metrics = actor.update_policy(batch)
        assert metrics["actor/optimizer_step_per_minibatch"] == [0.0]
        for previous, parameter in zip(before, actor.actor_module.parameters()):
            torch.testing.assert_close(previous, parameter, atol=0, rtol=0)
            gathered = [torch.empty_like(parameter) for _ in range(2)]
            torch.distributed.all_gather(gathered, parameter.detach())
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
        assert counts[0] == 4
    finally:
        torch.distributed.destroy_process_group()


class ResponseBudgetTest(unittest.TestCase):
    def make_budget(self, limit):
        return budget_module.ResponseBudget(TinyTokenizer(), None, [{"role": "user", "content": "q"}],
                                            None, limit, text_encode, lambda messages: [])

    def test_whole_group_removal_and_empty_response(self):
        prompt = {"role": "user", "content": "q"}
        first = {"role": "assistant", "content": "answer"}
        messages = [prompt, first, {"role": "assistant", "content": "tool call"},
                    {"role": "tool", "content": "x" * 1000}]
        budget = self.make_budget(64)
        encoded = budget.enforce(messages)
        self.assertEqual(messages, [prompt, first])
        self.assertLessEqual(encoded[0].numel(), 64)
        self.assertTrue(budget.truncated)
        budget = self.make_budget(1)
        budget.enforce(messages)
        self.assertEqual(messages, [prompt])
        self.assertEqual(budget.encode(messages)[0].numel(), 0)
        self.assertEqual(budget.generation_allowance(messages), 0)

    def test_trace_pruning_marks_executed_discarded_calls(self):
        trace = {"model_calls": [{"latency_ms": 10}, {"latency_ms": 20}],
                 "tool_calls": [{"model_turn": 1, "latency_ms": 5}, {"model_turn": 2, "latency_ms": 8}],
                 "active_latency_ms": 43}
        budget_module.retain_trace(trace, 1, True, "response_length")
        self.assertEqual(trace["model_call_count"], 1)
        self.assertEqual(trace["tool_call_count"], 1)
        self.assertEqual(len(trace["discarded_calls"]), 2)
        self.assertTrue(all(call["discarded"] for call in trace["discarded_calls"]))

    def test_parallel_discarded_calls_preserve_execution_timings(self):
        # Two discarded tools run concurrently for 100 ms after 50 ms of model work.
        trace = {"model_calls": [{"latency_ms": 50}, {"latency_ms": 10}],
                 "tool_calls": [{"model_turn": 2, "latency_ms": 100}] * 2,
                 "active_latency_ms": 160, "queue_latency_ms": 20, "total_latency_ms": 180,
                 "model_latency_ms": 60, "tool_latency_ms": 200,
                 "model_time_ratio": 60 / 160, "tool_time_ratio": 200 / 160,
                 "scheduler_latency_ms": 0, "scheduler_time_ratio": 0}
        execution = {key: value for key, value in trace.items() if not isinstance(value, list)}
        budget_module.retain_trace(trace, 1, True, "response_length")
        for key, value in execution.items():
            self.assertEqual(trace[key], value)
        self.assertEqual(trace["execution_active_latency_ms"], 160)
        self.assertEqual(trace["retained_model_cumulative_latency_ms"], 50)
        self.assertEqual(trace["retained_tool_cumulative_latency_ms"], 0)
        self.assertEqual(trace["retained_call_cumulative_latency_ms"], 50)

    def test_grpo_excludes_masked_and_empty_groups(self):
        grpo = definitions(VERL / "trainer/ppo/core_algos.py", functions=("compute_grpo_outcome_advantage",))["compute_grpo_outcome_advantage"]
        rewards = torch.tensor([[1.0], [0.0], [100.0], [1.0], [0.0], [7.0]])
        mask = torch.tensor([[1], [1], [0], [0], [0], [1]])
        advantages, _ = grpo(rewards, mask, np.array(["a", "a", "a", "b", "b", "c"]))
        expected = torch.tensor([[1 / np.sqrt(2)], [-1 / np.sqrt(2)], [0], [0], [0], [0]], dtype=torch.float32)
        torch.testing.assert_close(advantages, expected, atol=2e-6, rtol=2e-6)
        advantages, _ = grpo(rewards, torch.zeros_like(mask), np.array(["a"] * 6))
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertEqual(advantages.count_nonzero(), 0)

    def test_common_padding_preserves_features_and_prediction_boundary(self):
        _, crop = verification.load_actor_forward()
        batch = verification.make_batch(verification.TinyCausalModel(), crops=2, original_image=True)
        original_mm = batch["multi_modal_inputs"]
        batch["position_ids"] = batch["position_ids"].unsqueeze(1).expand(-1, 3, -1)
        cropped, padding = crop(batch)
        self.assertIs(cropped["multi_modal_inputs"], original_mm)
        self.assertEqual(padding, 13)
        self.assertEqual(cropped["position_ids"].shape[-1], cropped["input_ids"].shape[-1])
        torch.testing.assert_close(cropped["input_ids"], batch["input_ids"][:, 7:-13])
        torch.testing.assert_close(cropped["responses"], batch["responses"][:, :-13])
        # Different rows can have different boundaries; only common padding is cropped.
        two = {k: v.repeat(2, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v for k, v in batch.items()}
        two["attention_mask"][1, 3] = 1
        two["attention_mask"][1, -2] = 1
        cropped, padding = crop(two)
        self.assertEqual(padding, 1)
        self.assertEqual(cropped["input_ids"].shape[-1], two["input_ids"].shape[-1] - 4)

    def test_cpu_two_rank_masked_updates(self):
        import sys
        sys.path.insert(0, str(ROOT / "reinforcement_learning"))
        from verl import DataProto
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            torch.multiprocessing.start_processes(distributed_update_worker,
                args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True, start_method="spawn")

    def test_budget_stops_generation_and_tool_execution(self):
        context = scheduler_definitions()
        prompt = [{"role": "user", "content": "q"}]
        scheduler = context["ChatCompletionScheduler"]()
        budget = self.make_budget(1)
        info = {"response_budget": budget, "__depth__": 0, "__done__": asyncio.Event()}
        scheduler.submit_chat_completions(messages=prompt, request_id=None, info=info)
        self.assertTrue(info["__done__"].is_set())
        callback = context["ToolCompletionCallback"]()
        message = SimpleNamespace(model_dump=lambda **kwargs: {"role": "assistant", "content": "x" * 100}, tool_calls=[])
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
        messages = list(prompt)
        asyncio.run(callback(messages, completion, {"response_budget": self.make_budget(1)}))
        self.assertEqual(messages, prompt)
        # An executed tool can exceed the remaining budget. Its entire group is rolled back.
        callback.max_turns = 8
        callback.scheduler = SimpleNamespace(submit_chat_completions=lambda **kwargs: self.fail("Budget exhausted; generation must stop"))
        async def call_tool(*args, **kwargs):
            return {"role": "tool", "content": "x" * 1000}
        callback._call_tool = call_tool
        message = SimpleNamespace(model_dump=lambda **kwargs: {"role": "assistant", "content": ""}, tool_calls=[{}])
        completion = SimpleNamespace(id="test", choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])
        messages = list(prompt)
        budget = self.make_budget(64)
        asyncio.run(callback(messages, completion, {"response_budget": budget}))
        self.assertTrue(budget.truncated)
        self.assertEqual(messages, prompt)

    def test_empty_loss_metrics_and_loss_modes(self):
        funcs = verification.load_loss_functions()
        values = torch.randn(2, 3, requires_grad=True)
        for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"):
            loss = funcs["agg_loss"](values, torch.zeros_like(values), mode)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(loss.item(), 0)
        context = definitions(VERL / "trainer/ppo/metric_utils.py",
                              functions=("_compute_response_info", "compute_data_metrics", "compute_timing_metrics"))
        zeros = torch.zeros(2, 3)
        batch = SimpleNamespace(batch={"responses": zeros.long(), "attention_mask": torch.tensor([[1, 1, 0, 0, 0]] * 2),
            "token_level_scores": zeros, "token_level_rewards": zeros, "advantages": zeros, "returns": zeros})
        metrics = context["compute_data_metrics"](batch, use_critic=False)
        metrics.update(context["compute_timing_metrics"](batch, {"gen": 1.0}))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_reward_uses_retained_response_even_when_loss_is_masked(self):
        import sys
        sys.path.insert(0, str(ROOT / "reinforcement_learning"))
        from verl import DataProto
        from tqdm import tqdm

        context = definitions(VERL / "workers/reward_manager/naive_async.py",
            functions=("unit_compute_reward_func",),
            classes={"AsyncNaiveRewardManager": {"__init__", "__call__"}},
            namespace={"ThreadPoolExecutor": ThreadPoolExecutor, "partial": partial,
                "tqdm": tqdm, "time": time, "datetime": datetime, "os": os, "json": json,
                "judge_stats_snapshot": lambda: {}, "judge_batch_summary": lambda *args: {}})
        seen = []
        def reward(**kwargs):
            seen.append(kwargs["solution_str"])
            return 1.0
        tokenizer = SimpleNamespace(decode=lambda ids: ",".join(str(i) for i in ids.tolist()))
        manager = context["AsyncNaiveRewardManager"](tokenizer, 0, compute_score=reward)
        manager.reward_kwargs["num_workers"] = 1
        batch = DataProto(batch=TensorDict({
            "prompts": torch.tensor([[11], [11]]), "responses": torch.tensor([[14, 15, 0], [0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]),
            "loss_mask": torch.zeros(2, 4, dtype=torch.long), "truncated": torch.tensor([True, True]),
        }, [2]), non_tensor_batch={
            "data_source": np.array(["test", "test"], dtype=object),
            "reward_model": np.array([{"ground_truth": "a"}, {"ground_truth": "a"}], dtype=object),
        })
        result = manager(batch)
        self.assertEqual(seen, ["14,15", ""])
        torch.testing.assert_close(result, torch.tensor([[0., 1., 0.], [0., 0., 0.]]))


class RealProcessorBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        model = Path(os.environ.get("QWEN3_VL_PROCESSOR_PATH", ROOT.parent / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        if not (model / "tokenizer.json").is_file():
            raise unittest.SkipTest("Set QWEN3_VL_PROCESSOR_PATH to local Qwen3-VL tokenizer/processor files")
        cls.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        cls.processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 4096}
        cls.tokenizer = cls.processor.tokenizer
        cls.context = scheduler_definitions()
        # Import the real image decoding helper under its normal package path.
        import sys
        sys.path.insert(0, str(ROOT / "reinforcement_learning"))

    def image(self, color):
        stream = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()

    def encode(self, text, images):
        return self.context["_encode_response_with_images"](self.tokenizer, self.processor, text, images)

    def budget(self, prompt, limit):
        return budget_module.ResponseBudget(self.tokenizer, self.processor, prompt, None, limit,
                                            self.context["_encode_response_with_images"], self.context["_collect_message_images"])

    def test_image_cache_matches_real_processor(self):
        prompt = [{"role": "user", "content": "q"}]
        images = [self.image("red"), self.image("blue")]
        messages = [*prompt, {"role": "assistant", "content": "inspect"},
                    {"role": "user", "content": [{"type": "text", "text": "<tool_response>crop</tool_response>"},
                        *[{"type": "image_url", "image_url": {"url": image}} for image in images]]}]
        budget = self.budget(prompt, 1024)
        result = budget.enforce(messages)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[len(budget.prompt_text):]
        expected = self.encode(text, images)
        for a, b in zip(result[:2], expected[:2]):
            torch.testing.assert_close(a, b)
        for key in result[2]:
            torch.testing.assert_close(result[2][key], expected[2][key])
        self.assertEqual(len(budget.processor.image_processor.cache), 2)
        messages.append({"role": "assistant", "content": "done"})
        budget.enforce(messages)
        self.assertEqual(len(budget.processor.image_processor.cache), 2)

    def test_discarded_images_are_released_without_reprocessing_retained_images(self):
        prompt = [{"role": "user", "content": "q"}]
        images = [self.image("red"), self.image("blue")]
        def group(image):
            return [{"role": "assistant", "content": "inspect"}, {"role": "user", "content": [
                {"type": "text", "text": "<tool_response>crop</tool_response>"},
                {"type": "image_url", "image_url": {"url": image}}]}]
        messages = [*prompt, *group(images[0]), *group(images[1])]
        budget = self.budget(prompt, 1024)
        retained_width = budget.encode([*prompt, *group(images[0])])[0].numel()
        retained_decoded = budget.image_cache[images[0]]
        retained_features = next(iter(budget.processor.image_processor.cache.values()))[1]
        budget.encode(messages)
        dropped_decoded = weakref.ref(budget.image_cache[images[1]])
        dropped_features = weakref.ref(next(value[1]["pixel_values"] for value in
            budget.processor.image_processor.cache.values() if value[0] is budget.image_cache[images[1]]))
        old_encoding = weakref.ref(budget.last_encoding[2]["pixel_values"])
        budget.limit = retained_width
        result = budget.enforce(messages)
        self.assertEqual(len(budget.image_cache), 1)
        self.assertEqual(len(budget.processor.image_processor.cache), 1)
        self.assertIs(budget.image_cache[images[0]], retained_decoded)
        self.assertIs(next(iter(budget.processor.image_processor.cache.values()))[1], retained_features)
        self.assertIsNone(dropped_decoded())
        self.assertIsNone(dropped_features())
        self.assertIsNone(old_encoding())
        self.assertEqual(result[2]["image_grid_thw"].shape[0], 1)
        budget.limit = 1
        empty = budget.enforce(messages)
        self.assertEqual(empty[0].numel(), 0)
        self.assertEqual(empty[2], {})
        self.assertEqual(budget.image_cache, {})
        self.assertEqual(budget.processor.image_processor.cache, {})
        self.assertIsNone(budget.last_encoding)

    def test_scheduler_releases_intermediate_images_before_batch_finishes(self):
        scheduler = self.context["ChatCompletionScheduler"]()
        scheduler.request_semaphore = asyncio.Semaphore(1)
        scheduler.config = SimpleNamespace(response_length=1024)
        scheduler.completion_callback = SimpleNamespace(tokenizer=self.tokenizer, processor=self.processor, tool_schemas=None)
        messages = [{"role": "user", "content": "q"}]
        def submit(**kwargs):
            kwargs["messages"].extend([{"role": "assistant", "content": "inspect"}, {"role": "user", "content": [
                {"type": "text", "text": "<tool_response>crop</tool_response>"},
                {"type": "image_url", "image_url": {"url": self.image("red")}}]}])
            kwargs["info"]["response_budget"].encode(kwargs["messages"])
            kwargs["info"]["__done__"].set()
        scheduler.submit_chat_completions = submit
        trace = asyncio.run(scheduler._submit_chat_completions_semaphore(messages, None, {}))
        budget = trace["_response_budget"]
        final_encoding = budget.last_encoding
        self.assertEqual(budget.image_cache, {})
        self.assertEqual(budget.processor.image_processor.cache, {})
        self.assertEqual(final_encoding[2]["image_grid_thw"].shape[0], 1)
        self.assertIs(budget.enforce(messages), final_encoding)
        budget.release_cache()
        self.assertIsNone(budget.last_encoding)
        self.assertEqual(final_encoding[2]["image_grid_thw"].shape[0], 1)

    def test_postprocess_budget_masks_and_mrope_alignment(self):
        prompt = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": self.image("green")}},
                                                  {"type": "text", "text": "q"}]}]
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        from verl.utils.dataset.vision_utils import process_image, merge_multi_modal_inputs

        inputs = self.processor(text=[prompt_text], images=[process_image(prompt[0]["content"][0]["image_url"]["url"])], return_tensors="pt")
        raw = np.empty(1, dtype=object)
        raw[0] = prompt
        batch = SimpleNamespace(batch=TensorDict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
            "position_ids": torch.arange(inputs["input_ids"].shape[-1]).view(1, 1, -1).expand(1, 3, -1)}, [1]),
            non_tensor_batch={"raw_prompt": raw})
        first = [{"role": "assistant", "content": "inspect"}, {"role": "user", "content": [
            {"type": "text", "text": "<tool_response>crop</tool_response>"},
            {"type": "image_url", "image_url": {"url": self.image("red")}}]}]
        messages = [*prompt, *first, {"role": "assistant", "content": "inspect again"},
                    {"role": "user", "content": [{"type": "text", "text": "<tool_response>" + "region " * 5000 + "</tool_response>"},
                        {"type": "image_url", "image_url": {"url": self.image("blue")}}]}]
        limit = self.budget(prompt, 1024).encode([*prompt, *first])[0].numel()
        callback = self.context["ToolCompletionCallback"]()
        callback.tokenizer, callback.processor, callback.tool_schemas = self.tokenizer, self.processor, None
        callback.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(response_length=limit, multi_turn={"overlong_masking": True})))
        output = callback.postprocess(batch, [messages], n=1)
        self.assertEqual(messages, [*prompt, *first])
        self.assertLessEqual(output.batch["responses"].shape[-1], limit)
        self.assertTrue(output.batch["truncated"].item())
        self.assertEqual(output.batch["loss_mask"].count_nonzero(), 0)
        self.assertEqual(output.non_tensor_batch["__num_turns__"].tolist(), [1])
        mm = merge_multi_modal_inputs({k: v for k, v in inputs.items() if k in ("pixel_values", "image_grid_thw")}, output.non_tensor_batch["rollout_multi_modal_inputs"][0])
        self.assertEqual(mm["image_grid_thw"].shape[0], 2)
        image_token = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        expected_image_tokens = (mm["image_grid_thw"].prod(-1) // self.processor.image_processor.merge_size ** 2).sum()
        self.assertEqual((output.batch["input_ids"] == image_token).sum(), expected_image_tokens)
        rope = definitions(VERL / "models/transformers/qwen2_vl.py", functions=("get_rope_index",))["get_rope_index"]
        positions = rope(self.processor, output.batch["input_ids"][0], image_grid_thw=mm["image_grid_thw"], attention_mask=output.batch["attention_mask"][0])
        self.assertEqual(positions.shape, (3, output.batch["input_ids"].shape[-1]))
        self.assertTrue(torch.isfinite(positions).all())


if __name__ == "__main__":
    unittest.main()
