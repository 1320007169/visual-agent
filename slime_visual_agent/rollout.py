"""Qwen3-VL multi-turn, image-returning rollout for pinned SLIME FSDP.

This follows SLIME's custom-generate contract at commit
0104a9e922ecd3cea768f7c2520e7d1a122afdc9. Model-generated tokens are
trainable; environment/tool observations, including returned image tokens,
are present in the context but masked from the policy loss.
"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import torch

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import build_processor_kwargs, encode_image_for_rollout_engine
from slime.utils.types import Sample

from .protocol import (
    DEFAULT_TOOL_NAMES,
    append_response_segment,
    extract_latest_tool_call,
    tool_observation_message,
    validate_sample_alignment,
)
from .tool_client import execute_visual_tool


DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _allowed_tool_names() -> frozenset[str]:
    raw = os.environ.get("VISUAL_AGENT_ALLOWED_TOOL_NAMES", "")
    if not raw.strip():
        return DEFAULT_TOOL_NAMES
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def _merge_multimodal_train_inputs(chunks: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    values_by_key: dict[str, list[Any]] = {}
    for chunk in chunks:
        for key, value in (chunk or {}).items():
            if value is not None:
                values_by_key.setdefault(key, []).append(value)
    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if values and all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat(values, dim=0)
    return merged or None


def _initial_inputs(sample: Sample, state: GenerateState) -> tuple[list[int], list[str], dict[str, Any] | None]:
    if state.processor is None:
        raise RuntimeError("Qwen3-VL rollout requires an AutoProcessor")
    if not sample.multimodal_inputs or not sample.multimodal_inputs.get("images"):
        raise RuntimeError("Qwen3-VL rollout sample has no input images")
    processor_output = state.processor(
        text=sample.prompt,
        **build_processor_kwargs(sample.multimodal_inputs),
    )
    prompt_ids = [int(token) for token in processor_output["input_ids"][0]]
    multimodal_train_inputs = {
        key: value
        for key, value in processor_output.items()
        if key not in {"input_ids", "attention_mask"}
    } or None
    image_data = [encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_train_inputs


def _encode_tool_observation(
    state: GenerateState,
    message: dict[str, Any],
    metadata: dict[str, Any],
    args: Any,
) -> tuple[list[int], list[str], dict[str, Any] | None, dict[str, Any] | None]:
    tools = metadata.get("tools") if metadata else None
    template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    dummy_prompt = state.tokenizer.apply_chat_template(
        DUMMY_MESSAGES,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    formatted = state.tokenizer.apply_chat_template(
        DUMMY_MESSAGES + [message],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    trim_length = len(state.tokenizer.encode(dummy_prompt, add_special_tokens=False))

    from qwen_vl_utils import process_vision_info

    images, videos = process_vision_info([message])
    multimodal_inputs = {"images": images, "videos": videos}
    processor_output = state.processor(
        text=formatted,
        **build_processor_kwargs(multimodal_inputs),
    )
    observation_ids = [int(token) for token in processor_output["input_ids"][0]][trim_length:]
    if state.tokenizer.bos_token_id is not None and observation_ids[:1] == [state.tokenizer.bos_token_id]:
        observation_ids = observation_ids[1:]
    image_data = [encode_image_for_rollout_engine(image) for image in images or []]
    train_inputs = {
        key: value
        for key, value in processor_output.items()
        if key not in {"input_ids", "attention_mask"}
    } or None
    return observation_ids, image_data, multimodal_inputs, train_inputs


def _update_raw_multimodal_inputs(sample: Sample, observation_inputs: dict[str, Any] | None) -> None:
    if not observation_inputs:
        return
    if sample.multimodal_inputs is None:
        sample.multimodal_inputs = observation_inputs
        return
    for key, values in observation_inputs.items():
        if not values:
            continue
        existing = sample.multimodal_inputs.setdefault(key, [])
        if isinstance(existing, list) and isinstance(values, list):
            existing.extend(values)


def _tool_error_output(name: str, error: Exception) -> str:
    return json.dumps(
        {
            "status": "error",
            "tool": name,
            "error_type": type(error).__name__,
            "message": (str(error).strip() or type(error).__name__)[:1000],
            "recoverable": True,
        },
        ensure_ascii=False,
    )


def _remaining_budget(args: Any, sampling_params: dict[str, Any], sample: Sample, response_tokens: list[int]) -> int:
    response_budget = int(sampling_params.get("max_new_tokens") or args.rollout_max_response_len)
    remaining = response_budget - len(response_tokens)
    context_limit = getattr(args, "rollout_max_context_len", None)
    if context_limit is not None:
        remaining = min(remaining, int(context_limit) - len(sample.tokens))
    return remaining


async def _generate_turn(
    args: Any,
    state: GenerateState,
    sample: Sample,
    image_data: list[str],
    sampling_params: dict[str, Any],
) -> tuple[str, list[int], list[float], str]:
    payload: dict[str, Any] = {
        "input_ids": sample.tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data
    headers = None
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(url, payload, headers=headers)
    meta = output["meta_info"]
    token_logprobs = meta.get("output_token_logprobs") or []
    tokens = [int(item[1]) for item in token_logprobs]
    log_probs = [float(item[0]) for item in token_logprobs]
    finish_type = str(meta.get("finish_reason", {}).get("type", "stop"))
    return str(output.get("text", "")), tokens, log_probs, finish_type


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """SLIME custom generation entry point for image-returning visual tools."""
    if getattr(args, "partial_rollout", False):
        raise RuntimeError("partial rollout is not supported by the Visual-Agent PoC")

    state = GenerateState(args)
    prompt_ids, image_data, initial_train_inputs = _initial_inputs(sample, state)
    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    elif sample.tokens[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("existing sample tokens do not start with the encoded prompt")
    response_tokens = list(sample.tokens[len(prompt_ids) :])
    sample.loss_mask = list(sample.loss_mask or [])
    sample.rollout_log_probs = list(sample.rollout_log_probs or [])
    sample.response_length = len(response_tokens)
    multimodal_train_inputs = [initial_train_inputs]
    max_turns = int(getattr(args, "max_turns", 0) or 0)
    if max_turns < 1:
        raise RuntimeError("max_turns must be positive in the custom SLIME config")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    traces = metadata.setdefault("visual_tool_traces", [])
    sample.metadata = metadata
    instance_id = f"slime-{sample.index}-{uuid4().hex}"
    allowed_tools = _allowed_tool_names()

    for turn in range(max_turns):
        budget = _remaining_budget(args, sampling_params, sample, response_tokens)
        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        turn_sampling = dict(sampling_params)
        turn_sampling["max_new_tokens"] = budget
        response_text, new_tokens, new_log_probs, finish_type = await _generate_turn(
            args,
            state,
            sample,
            image_data,
            turn_sampling,
        )
        append_response_segment(
            sample,
            response_tokens,
            new_tokens,
            log_probs=new_log_probs,
            trainable=True,
        )
        if finish_type in {"length", "abort"}:
            sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.ABORTED
            break

        tool_call = extract_latest_tool_call(response_text, allowed_tools)
        if tool_call is None:
            sample.status = Sample.Status.COMPLETED
            break
        try:
            tool_result = await execute_visual_tool(
                name=tool_call.name,
                arguments=tool_call.arguments,
                images=image_data,
                instance_id=instance_id,
            )
            output_text = tool_result.output
            returned_images = tool_result.images
            trace = {
                "turn": turn,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "success",
                **tool_result.metrics,
            }
        except Exception as exc:
            output_text = _tool_error_output(tool_call.name, exc)
            returned_images = []
            trace = {
                "turn": turn,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        traces.append(trace)

        observation = tool_observation_message(output_text, returned_images)
        observation_ids, returned_image_data, observation_inputs, observation_train_inputs = (
            _encode_tool_observation(state, observation, metadata, args)
        )
        append_response_segment(
            sample,
            response_tokens,
            observation_ids,
            log_probs=None,
            trainable=False,
        )
        image_data.extend(returned_image_data)
        _update_raw_multimodal_inputs(sample, observation_inputs)
        multimodal_train_inputs.append(observation_train_inputs)
        if turn + 1 == max_turns:
            sample.status = Sample.Status.COMPLETED

    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs)
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    validate_sample_alignment(sample, len(prompt_ids))
    return sample
