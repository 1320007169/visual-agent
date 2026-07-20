import asyncio
import json
from types import SimpleNamespace

import torch

from verl.utils.dataset.vision_utils import merge_multi_modal_inputs
from verl.workers.rollout.chat_scheduler import (
    ToolCompletionCallback,
    _collect_message_images,
    _encode_response_with_images,
)


class _CropTool:
    async def create(self, instance_id=None, **kwargs):
        assert kwargs["images"] == ["original-image"]
        return "crop-test"

    async def execute(self, instance_id, parameters, **kwargs):
        assert instance_id == "crop-test"
        assert parameters["query"] == "small sign"
        return (
            json.dumps({"crop_zoom": {"target_image": 1}}),
            0.0,
            {"returned_images": ["data:image/jpeg;base64,AA=="]},
        )

    async def release(self, instance_id):
        assert instance_id == "crop-test"


def _callback():
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tools = {"sam3_crop_zoom": _CropTool()}
    return callback


def test_native_tool_response_includes_returned_crop_image():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="sam3_crop_zoom",
            arguments=json.dumps({"query": "small sign"}),
        ),
    )
    info = {"images": ["original-image"]}

    message = asyncio.run(_callback()._call_tool(tool_call, info, xml_mode=False))

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call-1"
    assert message["content"][0]["type"] == "text"
    assert message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AA=="},
    }
    assert info["images"] == ["original-image", "data:image/jpeg;base64,AA=="]


def test_xml_tool_response_still_includes_returned_crop_image():
    tool_call = {"name": "sam3_crop_zoom", "arguments": {"query": "small sign"}}
    info = {"images": ["original-image"]}

    message = asyncio.run(_callback()._call_tool(tool_call, info, xml_mode=True))

    assert message["role"] == "user"
    assert message["content"][0]["type"] == "text"
    assert "<tool_response>" in message["content"][0]["text"]
    assert message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AA=="},
    }


class _Tokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class _Processor:
    def __call__(self, text, images, return_tensors, add_special_tokens):
        assert text == ["response-with-image"]
        assert images == ["decoded:crop-data"]
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {
            "input_ids": torch.tensor([[1, 99, 99, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


def test_response_images_are_encoded_with_matching_multimodal_inputs(monkeypatch):
    monkeypatch.setattr(
        "verl.utils.dataset.vision_utils.process_image",
        lambda image: f"decoded:{image}",
    )
    input_ids, attention_mask, mm_inputs = _encode_response_with_images(
        _Tokenizer(),
        _Processor(),
        "response-with-image",
        ["crop-data"],
    )

    assert input_ids.tolist() == [1, 99, 99, 2]
    assert attention_mask.tolist() == [1, 1, 1, 1]
    assert mm_inputs["pixel_values"].shape == (4, 3)
    assert mm_inputs["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_collect_and_merge_returned_image_inputs():
    messages = [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "crop"},
                {"type": "image_url", "image_url": {"url": "crop-data"}},
            ],
        }
    ]
    assert _collect_message_images(messages) == ["crop-data"]

    merged = merge_multi_modal_inputs(
        {
            "pixel_values": torch.zeros(2, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "pixel_values": torch.ones(3, 3),
            "image_grid_thw": torch.tensor([[1, 3, 3]]),
        },
    )
    assert merged["pixel_values"].shape == (5, 3)
    assert merged["image_grid_thw"].tolist() == [[1, 2, 2], [1, 3, 3]]

