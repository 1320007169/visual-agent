import asyncio
import base64
import json
from types import SimpleNamespace

import torch

from verl.utils.dataset.vision_utils import merge_multi_modal_inputs
from verl.workers.rollout.chat_scheduler import (
    ToolCompletionCallback,
    _collect_message_images,
    _encode_response_with_images,
    _image_to_data_url,
    _prepare_rollout_images,
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


class _FailingTool:
    def __init__(self):
        self.released = False

    async def create(self, instance_id=None, **kwargs):
        return "failure-test"

    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("no region was detected")

    async def release(self, instance_id):
        self.released = True


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


def test_native_tool_failure_returns_recoverable_observation():
    failing_tool = _FailingTool()
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tools = {"sam3_crop_zoom": failing_tool}
    tool_call = SimpleNamespace(
        id="call-error",
        function=SimpleNamespace(
            name="sam3_crop_zoom",
            arguments=json.dumps({"query": "missing sign"}),
        ),
    )

    message = asyncio.run(callback._call_tool(tool_call, {"images": []}, xml_mode=False))

    payload = json.loads(message["content"])
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call-error"
    assert payload == {
        "status": "error",
        "tool": "sam3_crop_zoom",
        "error_type": "RuntimeError",
        "message": "no region was detected",
        "recoverable": True,
    }
    assert failing_tool.released is True


def test_xml_tool_failure_returns_recoverable_observation():
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tools = {"sam3_crop_zoom": _FailingTool()}
    tool_call = {"name": "sam3_crop_zoom", "arguments": {"query": "missing sign"}}

    message = asyncio.run(callback._call_tool(tool_call, {"images": []}, xml_mode=True))

    assert message["role"] == "user"
    assert "<tool_response>" in message["content"]
    assert '"status": "error"' in message["content"]
    assert '"recoverable": true' in message["content"]


def test_unknown_tool_returns_recoverable_observation():
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tools = {}
    tool_call = {"name": "missing_tool", "arguments": {}}

    message = asyncio.run(callback._call_tool(tool_call, {"images": []}, xml_mode=True))

    assert message["role"] == "user"
    assert '"tool": "missing_tool"' in message["content"]
    assert '"error_type": "KeyError"' in message["content"]


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


def test_source_image_transport_preserves_original_file_bytes(tmp_path):
    original_bytes = b"original-png-file-bytes"
    image_path = tmp_path / "source.png"
    image_path.write_bytes(original_bytes)

    data_url = _image_to_data_url(str(image_path))

    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == original_bytes


def test_source_image_transport_encodes_once_per_source_prompt(tmp_path):
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"first-version")
    cache = {}

    first = _prepare_rollout_images([str(image_path)], 0, "source_cached", cache)
    image_path.write_bytes(b"changed-after-first-encoding")
    second = _prepare_rollout_images([str(image_path)], 0, "source_cached", cache)
    other_source = _prepare_rollout_images([str(image_path)], 1, "source_cached", cache)

    assert first is not second
    assert base64.b64decode(first[0].split(",", 1)[1]) == b"first-version"
    assert base64.b64decode(second[0].split(",", 1)[1]) == b"first-version"
    assert base64.b64decode(other_source[0].split(",", 1)[1]) == b"changed-after-first-encoding"

    first.append("crop-from-first-rollout")
    assert second == [cache[0][0]]


def test_pil_png_transport_keeps_existing_behavior():
    images = [object()]
    cache = {}

    assert _prepare_rollout_images(images, 0, "pil_png", cache) is images
    assert cache == {}
