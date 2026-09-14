import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from slime_visual_agent.protocol import (
    append_response_segment,
    extract_latest_tool_call,
    slime_grpo_advantages,
    tool_observation_message,
    validate_sample_alignment,
)
from slime_visual_agent.reward import score_response


FIXTURE = REPO_ROOT / "tests/fixtures/slime_visual_agent_fixed_trajectory.json"


def test_fixed_image_return_trajectory_alignment():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sample = SimpleNamespace(
        tokens=list(fixture["prompt_tokens"]),
        response_length=0,
        loss_mask=[],
        rollout_log_probs=[],
    )
    response_tokens = []
    for segment in fixture["segments"]:
        append_response_segment(
            sample,
            response_tokens,
            segment["tokens"],
            log_probs=segment["log_probs"],
            trainable=segment["kind"] == "generation",
        )

    validate_sample_alignment(sample, len(fixture["prompt_tokens"]))
    assert response_tokens == fixture["expected_response_tokens"]
    assert sample.tokens == fixture["prompt_tokens"] + fixture["expected_response_tokens"]
    assert sample.response_length == len(fixture["expected_response_tokens"])
    assert sample.loss_mask == fixture["expected_loss_mask"]
    assert sample.rollout_log_probs == fixture["expected_rollout_log_probs"]


def test_fixed_group_advantages_match_slime_and_verl_sample_std():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    advantages = slime_grpo_advantages(fixture["group_rewards"])
    assert advantages == pytest.approx([0.7071056701, -0.7071056701])
    assert sum(advantages) == pytest.approx(0.0)


def test_tool_call_and_returned_image_protocol():
    response = (
        '<tool_call>{"name":"crop_zoom","arguments":'
        '{"bbox_2d":[10,20,300,400],"target_image":0}}</tool_call>'
    )
    call = extract_latest_tool_call(response, {"crop_zoom"})
    assert call is not None
    assert call.name == "crop_zoom"
    assert call.arguments["target_image"] == 0

    observation = tool_observation_message('{"crop_image":1}', ["data:image/png;base64,AA=="])
    assert observation["role"] == "user"
    assert observation["content"][0]["type"] == "text"
    assert observation["content"][1] == {
        "type": "image",
        "image": "data:image/png;base64,AA==",
    }


def _load_verl_reward_module():
    path = REPO_ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
    spec = importlib.util.spec_from_file_location("visual_agent_thyme_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("<answer>right of</answer>", "right of"),
        ("reasoning\n<answer>right</answer>", "right of"),
        (
            "assistant <tool_call>{}</tool_call> user <tool_response>{}</tool_response> "
            "assistant <answer>right of</answer>",
            "right of",
        ),
        ("<answer>left of</answer>", "right of"),
        ("right of", "right of"),
        ("<answer>on</answer>", "inside"),
    ],
)
def test_reward_matches_existing_verl_zwz_reward(response, label):
    reference = _load_verl_reward_module().compute_score(
        response,
        label,
        {"source": "zwz_rl_vqa/original_images"},
    )
    assert score_response(response, label) == reference
