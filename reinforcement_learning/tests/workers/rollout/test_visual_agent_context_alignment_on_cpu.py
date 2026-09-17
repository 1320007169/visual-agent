"""Actor/rollout input parity, without model weights or a running server.

Set QWEN3VL_PROCESSOR_PATH to local Qwen3-VL tokenizer/processor assets to run
the integration cases. They never download files or load the 8B model.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.tools.schemas import load_tool_schemas_from_config
from verl.utils.dataset import zwz_original_relation_dataset as zwz
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.dataset.vision_utils import merge_multi_modal_inputs, process_raw_image
from verl.workers.rollout.chat_scheduler import (
    ChatCompletionScheduler,
    ToolCompletionCallback,
    _collect_message_images,
    _image_to_data_url,
)


ROOT = Path(__file__).resolve().parents[4]
TOOL_CONFIG = ROOT / "reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_groundingdino_config.yaml"


def test_dataset_schemas_match_runtime_tool_schemas():
    tools = initialize_tools_from_config(str(TOOL_CONFIG))
    expected = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools]
    assert load_tool_schemas_from_config(str(TOOL_CONFIG)) == expected
    assert load_tool_schemas_from_config(None) == []


def test_dataset_tool_config_follows_rollout_config():
    config = OmegaConf.load(ROOT / "reinforcement_learning/verl/trainer/config/ppo_trainer.yaml")
    assert config.data.tool_config_path is None
    config.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(TOOL_CONFIG)
    assert config.data.tool_config_path == str(TOOL_CONFIG)


def test_schema_loading_does_not_initialize_clients(tmp_path):
    path = tmp_path / "tools.yaml"
    OmegaConf.save(OmegaConf.create({"tools": [{
        "class_name": "unavailable_package.RemoteTool",
        "config": {"api_key": "${oc.env:UNSET_CONTEXT_TEST_SECRET}"},
        "tool_schema": load_tool_schemas_from_config(str(TOOL_CONFIG))[0],
    }]}), path)
    assert len(load_tool_schemas_from_config(str(path))) == 1


@pytest.fixture(scope="module")
def qwen():
    path = os.environ.get("QWEN3VL_PROCESSOR_PATH")
    if not path:
        pytest.skip("Set QWEN3VL_PROCESSOR_PATH for offline processor integration tests")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
    assert type(processor).__name__ == "Qwen3VLProcessor"
    return tokenizer, processor


@pytest.fixture
def dataset_factory(qwen, tmp_path, monkeypatch):
    tokenizer, processor = qwen
    monkeypatch.setenv("VISUAL_AGENT_IMAGE_MAX_PIXELS", "2359296")
    monkeypatch.setenv("VISUAL_AGENT_IMAGE_PATCH_SIZE", "16")
    monkeypatch.setattr(zwz, "DUAL_STREAM_ENABLED", True)
    monkeypatch.setattr(zwz, "SYSTEM_PROMPT", (ROOT / "prompts/visual_agent_rl_system_groundingdino.txt").read_text().strip())
    # Non-constant pixels make lossy encoding or a different resize observable.
    pixels = np.zeros((2048, 4096, 3), dtype=np.uint8)
    pixels[..., 0] = (np.arange(4096) % 256)[None, :]
    pixels[..., 1] = (np.arange(2048) % 256)[:, None]
    image_path = tmp_path / "original.png"
    Image.fromarray(pixels).save(image_path)
    data_path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "images": [str(image_path)], "question": "Where is the sign relative to the car?",
        "solution": "above", "source_image": "original", "bbox": [],
    }]), data_path)

    def create(**overrides):
        config = OmegaConf.create({
            "max_prompt_length": 8192, "return_raw_chat": True,
            "filter_overlong_prompts": False, "filter_overlong_prompts_workers": 1,
            "tool_config_path": str(TOOL_CONFIG), "truncation": "error",
            "cache_dir": str(tmp_path / "cache"), **overrides,
        })
        return zwz.ZwzOriginalRelationDataset(str(data_path), tokenizer, config, processor)

    return create


class CropTool:
    """Deterministic test executor; records which resolution the tool receives."""

    def __init__(self):
        self.instances = {}
        self.sources = []

    async def create(self, **kwargs):
        key = str(len(self.sources))
        self.instances[key] = list(kwargs["images"])
        return key

    async def execute(self, instance_id, arguments):
        images = self.instances[instance_id]
        source = process_raw_image(images[arguments["target_image"]])
        self.sources.append((len(images), source.size))
        crop = source.crop((0, 0, source.width // 2, source.height // 2))
        return json.dumps({"target_image": len(images)}), 0.0, {
            "returned_images": [_image_to_data_url(crop)]
        }

    async def release(self, instance_id):
        self.instances.pop(instance_id)


@pytest.mark.parametrize("transport", ["pil_png", "source_cached"])
@pytest.mark.parametrize("route", ["direct", "crop_twice", "native_crop", "native"])
def test_actor_and_rollout_see_identical_inputs(qwen, dataset_factory, monkeypatch, transport, route):
    tokenizer, processor = qwen
    monkeypatch.setenv("VISUAL_AGENT_IMAGE_TRANSPORT", transport)
    monkeypatch.setattr(zwz, "IMAGE_TRANSPORT_MODE", transport)
    row = dataset_factory()[0]
    if route == "native":
        for key in ("input_ids", "attention_mask", "position_ids", "raw_prompt"):
            row[key] = row[f"native_{key}"]
    batch = DataProto.from_single_dict(collate_fn([row]))
    if route == "native":
        batch.meta_info["disable_tools"] = True
    original_prompt = copy.deepcopy(row["raw_prompt"])
    actor_prompt_ids = row["input_ids"][row["attention_mask"].bool()]
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tokenizer, callback.processor = tokenizer, processor
    callback._tool_schemas = load_tool_schemas_from_config(str(TOOL_CONFIG))
    crop_tool = CropTool()
    callback.tools = {"crop_zoom": crop_tool}
    scheduler = ChatCompletionScheduler.__new__(ChatCompletionScheduler)
    scheduler.config = OmegaConf.create({"temperature": 1.0, "top_p": 1.0, "n": 2,
                                         "multi_turn": {"max_tokens_per_turn": 512}})
    scheduler.model_name = "offline-test"
    scheduler.trace_rollouts = False
    scheduler.completion_callback = callback
    expected_inputs = []

    async def generate(*, messages, images, tools_enabled, **kwargs):
        schemas = callback.tool_schemas if tools_enabled else None
        assert tools_enabled == (route != "native")
        initial_text = tokenizer.apply_chat_template(messages, tools=schemas, add_generation_prompt=True, tokenize=False)
        visible_images = [process_raw_image(value) for value in _collect_message_images(messages)]
        initial = processor(text=[initial_text], images=visible_images, return_tensors="pt")
        assert torch.equal(initial["input_ids"][0], actor_prompt_ids)
        assert torch.equal(initial["image_grid_thw"], row["multi_modal_inputs"]["image_grid_thw"])
        assert torch.equal(initial["pixel_values"], row["multi_modal_inputs"]["pixel_values"])
        assert process_raw_image(images[0]).size == (4096, 2048)
        assert visible_images[0].size == (2144, 1056)
        assert int((initial["input_ids"] == processor.image_token_id).sum()) == 2211

        info = {"images": images}
        if route in {"crop_twice", "native_crop"}:
            for target in range(2 if route == "crop_twice" else 1):
                call = {"name": "crop_zoom", "arguments": {"target_image": target, "bbox_2d": [0, 0, 500, 500]}}
                if route == "native_crop":
                    native_call = SimpleNamespace(id="crop-call", function=SimpleNamespace(name=call["name"], arguments=json.dumps(call["arguments"])))
                    messages.append({"role": "assistant", "content": "", "tool_calls": [{
                        "id": "crop-call", "type": "function", "function": {
                            "name": call["name"], "arguments": json.dumps(call["arguments"]),
                        },
                    }]})
                    observation = await callback._call_tool(native_call, info, xml_mode=False)
                else:
                    messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps(call)}</tool_call>"})
                    observation = await callback._call_tool(call, info, xml_mode=True)
                messages.append(observation)
        messages.append({"role": "assistant", "content": "<answer>above</answer>"})
        full_text = tokenizer.apply_chat_template(messages, tools=schemas, add_generation_prompt=False, tokenize=False)
        full = processor(text=[full_text], images=[process_raw_image(value) for value in _collect_message_images(messages)], return_tensors="pt", add_special_tokens=False)
        expected_inputs.append(full)
        return {}

    scheduler._submit_chat_completions_semaphore = generate
    output = asyncio.run(scheduler.generate_sequences(batch))
    assert len(expected_inputs) == 2
    assert row["raw_prompt"] == original_prompt  # No shared-dict mutation across n or streams.
    for index, expected in enumerate(expected_inputs):
        actual_ids = output.batch["input_ids"][index][output.batch["attention_mask"][index].bool()]
        assert torch.equal(actual_ids, expected["input_ids"][0])
        merged = merge_multi_modal_inputs(row["multi_modal_inputs"], output.non_tensor_batch["rollout_multi_modal_inputs"][index])
        assert torch.equal(merged["image_grid_thw"], expected["image_grid_thw"])
        assert torch.equal(merged["pixel_values"], expected["pixel_values"])
        image_tokens = output.batch["responses"][index] == processor.image_token_id
        assert not output.batch["response_mask"][index][image_tokens].any()
        if route in {"crop_twice", "native_crop"}:
            assert image_tokens.any()
            tool_token = tokenizer.convert_tokens_to_ids("<tool_call>")
            assert output.batch["response_mask"][index][output.batch["responses"][index] == tool_token].all()
    if route == "crop_twice":
        assert crop_tool.sources == [(1, (4096, 2048)), (2, (2048, 1024))] * 2
    elif route == "native_crop":
        assert crop_tool.sources == [(1, (4096, 2048))] * 2


def test_prompt_length_filter_counts_tool_definitions(qwen, dataset_factory):
    _, processor = qwen
    row = dataset_factory()[0]
    no_tools_text = processor.apply_chat_template(row["raw_prompt"], add_generation_prompt=True, tokenize=False)
    no_tools_ids = processor(text=[no_tools_text], images=row["multi_modal_data"]["image"], return_tensors="pt")["input_ids"]
    budget = no_tools_ids.shape[-1] + 10
    assert row["attention_mask"].sum() > budget
    filtered = dataset_factory(max_prompt_length=budget, filter_overlong_prompts=True)
    assert len(filtered) == 0
    unfiltered = dataset_factory(max_prompt_length=budget)
    with pytest.raises((ValueError, RuntimeError)):
        unfiltered[0]


@pytest.mark.parametrize("with_tools", [False, True])
def test_text_dataset_uses_same_schema_contract(qwen, tmp_path, with_tools):
    tokenizer, _ = qwen
    path = tmp_path / "text.parquet"
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    pq.write_table(pa.Table.from_pylist([{"prompt": messages}]), path)
    config = OmegaConf.create({
        "tool_config_path": str(TOOL_CONFIG) if with_tools else None,
        "max_prompt_length": 1024, "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": 1, "return_raw_chat": True,
        "cache_dir": str(tmp_path / "cache"), "truncation": "error",
    })
    dataset = RLHFDataset(str(path), tokenizer, config)
    row = dataset[0]
    expected = tokenizer.apply_chat_template(
        messages, tools=load_tool_schemas_from_config(config.tool_config_path) or None,
        add_generation_prompt=True, tokenize=True,
    )
    assert row["input_ids"][row["attention_mask"].bool()].tolist() == expected
