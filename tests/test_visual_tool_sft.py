import json
from pathlib import Path

import pytest

from scripts.convert_visual_tool_sft import convert_files, convert_item
from scripts.validate_visual_tool_sft import validate_item


def _write_image(path: Path) -> str:
    path.write_bytes(b"not a real image, only path existence matters")
    return str(path)


def test_convert_item_maps_tool_observation_to_user_tool_response(tmp_path: Path):
    image = _write_image(tmp_path / "image.jpg")
    crop = _write_image(tmp_path / "crop.jpg")
    source = {
        "uid": "sample_attribute",
        "data_type": "attribute",
        "image": image,
        "images": [image, crop],
        "question": "What color is the target?",
        "answer": "red",
        "required_tools": ["sam3_crop_zoom"],
        "messages": [
            {"role": "user", "content": "<image>\nWhat color is the target?"},
            {
                "role": "assistant",
                "content": '<think>Need a crop.</think><tool_call>{"name":"sam3_crop_zoom","arguments":{"query":"target","target_image":0,"slack_ratio":0.35}}</tool_call>',
            },
            {"role": "tool", "content": '{"crop_image": 1, "query": "target"}'},
            {"role": "user", "content": "sam3_crop_zoom returned a crop: <image>"},
            {"role": "assistant", "content": "<think>The crop is clear.</think><answer>red</answer>"},
        ],
        "error": None,
        "validation_errors": [],
    }

    converted = convert_item(source, task_type="attribute")

    assert converted["uid"] == "sample_attribute"
    assert converted["task_type"] == "attribute"
    assert converted["images"] == [image, crop]
    assert "error" not in converted
    assert "validation_errors" not in converted
    assert [m["role"] for m in converted["messages"]] == ["user", "assistant", "user", "assistant"]
    assert converted["messages"][2]["content"].startswith("<tool_response>\n")
    assert "\n</tool_response>" in converted["messages"][2]["content"]
    assert "sam3_crop_zoom returned a crop: <image>" in converted["messages"][2]["content"]
    assert converted["messages_raw"][2]["role"] == "tool"


def test_convert_item_does_not_double_wrap_existing_tool_response(tmp_path: Path):
    image = _write_image(tmp_path / "image.jpg")
    source = {
        "uid": "sample_count",
        "data_type": "counting",
        "image": image,
        "images": [image],
        "question": "How many?",
        "answer": "2",
        "required_tools": ["sam3_segment_multi"],
        "messages": [
            {"role": "user", "content": "How many?"},
            {"role": "assistant", "content": "I should count with segmentation."},
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"sam3_segment_multi","arguments":{"queries":[{"role":"target","query":"window"}],"target_image":0}}</tool_call>',
            },
            {"role": "tool", "content": '<tool_response>{"count": 2}</tool_response>'},
            {"role": "assistant", "content": "<answer>2</answer>"},
        ],
    }

    converted = convert_item(source, task_type="counting")

    tool_response_messages = [m for m in converted["messages"] if "<tool_response>" in m["content"]]
    assert len(tool_response_messages) == 1
    assert tool_response_messages[0]["content"].count("<tool_response>") == 1
    assert tool_response_messages[0]["content"] == '<tool_response>{"count": 2}</tool_response>'
    assert [m["role"] for m in converted["messages"]] == ["user", "assistant", "user", "assistant"]


def test_validate_item_accepts_visual_tool_multimage_sample(tmp_path: Path):
    image = _write_image(tmp_path / "image.jpg")
    crop = _write_image(tmp_path / "crop.jpg")
    item = {
        "uid": "sample_attribute",
        "task_type": "attribute",
        "image": image,
        "images": [image, crop],
        "question": "What color is the target?",
        "answer": "red",
        "messages": [
            {"role": "user", "content": "<image>\nWhat color is the target?"},
            {
                "role": "assistant",
                "content": '<think>Need a crop.</think><tool_call>{"name":"sam3_crop_zoom","arguments":{"query":"target","target_image":0,"slack_ratio":0.35}}</tool_call>',
            },
            {
                "role": "user",
                "content": '<tool_response>\n{"crop_image": 1, "query": "target"}\n</tool_response>\nsam3_crop_zoom returned a crop: <image>',
            },
            {"role": "assistant", "content": "<think>The crop is clear.</think><answer>red</answer>"},
        ],
    }

    result = validate_item(item, seen_uids=set())

    assert result.errors == []
    assert result.tools == ["sam3_crop_zoom"]
    assert result.image_count == 2


def test_validate_item_rejects_unknown_tool(tmp_path: Path):
    image = _write_image(tmp_path / "image.jpg")
    item = {
        "uid": "sample_bad",
        "task_type": "counting",
        "image": image,
        "images": [image],
        "question": "How many?",
        "answer": "1",
        "messages": [
            {"role": "user", "content": "<image>\nHow many?"},
            {
                "role": "assistant",
                "content": '<think>Need a tool.</think><tool_call>{"name":"code_interpreter","arguments":{"code":"print(1)"}}</tool_call>',
            },
            {"role": "user", "content": '<tool_response>\n{"count": 1}\n</tool_response>'},
            {"role": "assistant", "content": "<think>Done.</think><answer>1</answer>"},
        ],
    }

    result = validate_item(item, seen_uids=set())

    assert any("unsupported tool" in error for error in result.errors)


def test_convert_files_makes_duplicate_uids_unique(tmp_path: Path):
    image = _write_image(tmp_path / "image.jpg")
    base = {
        "uid": "dupe",
        "image": image,
        "images": [image],
        "question": "How many?",
        "answer": "1",
        "messages": [
            {"role": "user", "content": "<image>\nHow many?"},
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"grounding_detect","arguments":{"query":"person","target_image":0}}</tool_call>',
            },
            {"role": "tool", "content": '{"count": 1}'},
            {"role": "assistant", "content": "<answer>1</answer>"},
        ],
    }
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(base) + "\n")
    second.write_text(json.dumps(base) + "\n")

    output = tmp_path / "out.jsonl"
    convert_files({"counting": first, "attribute": second}, output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["dupe", "attribute:dupe"]
    assert [row["source_uid"] for row in rows] == ["dupe", "dupe"]
