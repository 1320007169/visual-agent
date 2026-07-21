# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import heapq
import importlib
import io
import itertools
import json
import logging
import time
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _tool_error_observation(
    tool_name: str,
    error: Exception,
    *,
    xml_mode: bool,
    tool_call_id: str | None,
) -> Dict[str, Any]:
    error_text = str(error).strip() or type(error).__name__
    payload = json.dumps(
        {
            "status": "error",
            "tool": tool_name,
            "error_type": type(error).__name__,
            "message": error_text[:1000],
            "recoverable": True,
        },
        ensure_ascii=False,
    )
    if xml_mode:
        return {"role": "user", "content": f"<tool_response>\n{payload}\n</tool_response>"}
    return {"role": "tool", "content": payload, "tool_call_id": tool_call_id}


def _image_to_data_url(image: Any) -> str:
    if isinstance(image, str) and image.startswith(("data:image/", "http://", "https://")):
        return image
    if isinstance(image, bytes):
        data, mime = image, "image/jpeg"
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data, mime = buffer.getvalue(), "image/png"
    else:
        raise TypeError(f"Unsupported rollout image type: {type(image).__name__}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _attach_images_to_messages(messages: List[Dict[str, Any]], images: List[Any]) -> None:
    """Replace Qwen chat-template image placeholders for OpenAI HTTP serving."""
    image_iter = iter(images)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        normalized = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                try:
                    image = next(image_iter)
                except StopIteration as exc:
                    raise ValueError("Prompt contains more image placeholders than sample images") from exc
                normalized.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
            else:
                normalized.append(part)
        message["content"] = normalized

def _collect_message_images(messages: List[Dict[str, Any]]) -> List[Any]:
    """Collect OpenAI image parts from a slice of a conversation."""
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url is not None:
                images.append(image_url)
    return images


def _encode_response_with_images(tokenizer, processor, response_text: str, images: List[Any]):
    """Tokenize a response and build matching features for returned images."""
    if not images:
        encoded = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"][0], encoded["attention_mask"][0], {}
    if processor is None:
        raise RuntimeError("a multimodal processor is required for tool-returned images")

    from verl.utils.dataset.vision_utils import process_image

    model_inputs = processor(
        text=[response_text],
        images=[process_image(image) for image in images],
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]
    multi_modal_inputs = dict(model_inputs)
    multi_modal_inputs.pop("second_per_grid_ts", None)
    return input_ids, attention_mask, multi_modal_inputs


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        native_tool_calls = completions.choices[0].message.tool_calls or []
        xml_tool_calls = []
        if not native_tool_calls and message.get("content"):
            for raw_call in TOOL_CALL_RE.findall(message["content"]):
                try:
                    call = json.loads(raw_call)
                    if isinstance(call.get("name"), str) and isinstance(call.get("arguments", {}), dict):
                        xml_tool_calls.append(call)
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed XML tool call: %s", raw_call[:500])

        if finish_reason != "tool_calls" and not xml_tool_calls:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = native_tool_calls or xml_tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call, info, xml_mode=not bool(native_tool_calls)))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call, info: Dict[str, Any], *, xml_mode: bool = False) -> Dict[str, Any]:
        """Call tool and return tool response."""
        if xml_mode:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            tool_call_id = None
        else:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                return _tool_error_observation(
                    tool_name, exc, xml_mode=False, tool_call_id=tool_call_id
                )
        if not isinstance(tool_args, dict):
            return _tool_error_observation(
                tool_name,
                ValueError("Tool arguments must be a JSON object"),
                xml_mode=xml_mode,
                tool_call_id=tool_call_id,
            )

        trace = info.get("__trace__")
        tool_started = time.perf_counter()
        tool_trace = {
            "tool": tool_name,
            "arguments": tool_args,
            "model_turn": len(trace["model_calls"]) if trace is not None else None,
            "started_at_unix": time.time(),
        }
        tool = self.tools.get(tool_name)
        instance_id = None
        tool_response = ""
        tool_metrics: Dict[str, Any] = {}
        tool_error: Exception | None = None
        try:
            if tool is None:
                raise KeyError(f"Model requested unknown tool: {tool_name}")
            instance_id = await tool.create(images=info.get("images", []))
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as exc:
            tool_error = exc
            logger.exception("Error when executing tool %s: %s", tool_name, exc)
        finally:
            if instance_id is not None and tool is not None:
                try:
                    await tool.release(instance_id)
                except Exception as exc:
                    tool_trace["release_error"] = f"{type(exc).__name__}: {exc}"
                    logger.exception("Error when releasing tool %s: %s", tool_name, exc)
            tool_trace["latency_ms"] = round((time.perf_counter() - tool_started) * 1000, 3)
            if trace is not None:
                trace["tool_calls"].append(tool_trace)

        if tool_error is not None:
            tool_trace["status"] = "error"
            tool_trace["error"] = f"{type(tool_error).__name__}: {tool_error}"
            tool_trace["returned_image_count"] = 0
            return _tool_error_observation(
                tool_name,
                tool_error,
                xml_mode=xml_mode,
                tool_call_id=tool_call_id,
            )

        returned_images = tool_metrics.get("returned_images", [])
        tool_trace["status"] = "success"
        tool_trace["service_latency_ms"] = tool_metrics.get("latency_ms")
        tool_trace["returned_image_count"] = len(returned_images)
        if returned_images:
            info.setdefault("images", []).extend(returned_images)

        if xml_mode:
            text = f"<tool_response>\n{tool_response}\n</tool_response>"
            if returned_images:
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        *[{"type": "image_url", "image_url": {"url": image}} for image in returned_images],
                    ],
                }
            return {"role": "user", "content": text}
        if returned_images:
            # Native/Hermes tool calls must return crop images in the tool
            # message itself. Merely extending ``info["images"]`` keeps the
            # images in scheduler state, but the next OpenAI chat request only
            # sees ``messages`` and therefore cannot inspect those crops.
            return {
                "role": "tool",
                "content": [
                    {"type": "text", "text": tool_response},
                    *[{"type": "image_url", "image_url": {"url": image}} for image in returned_images],
                ],
                "tool_call_id": tool_call_id,
            }
        return {"role": "tool", "content": tool_response, "tool_call_id": tool_call_id}

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompt_texts = [
            self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]
        assert len(batch_conversations) == len(prompt_texts) * n

        sequences = [
            self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False)
            for conversation in batch_conversations
        ]
        response_texts = [sequence[len(prompt_texts[i // n]) :] for i, sequence in enumerate(sequences)]
        response_input_ids = []
        response_attention_masks = []
        rollout_multi_modal_inputs = []
        raw_prompts = batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)
        for index, response_text in enumerate(response_texts):
            response_messages = batch_conversations[index][len(raw_prompts[index]) :]
            returned_images = _collect_message_images(response_messages)
            input_ids, attention_mask, mm_inputs = _encode_response_with_images(
                self.tokenizer,
                self.processor,
                response_text,
                returned_images,
            )
            response_input_ids.append(input_ids)
            response_attention_masks.append(attention_mask)
            rollout_multi_modal_inputs.append(mm_inputs)

        responses = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                response_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                response_attention_masks,
                batch_first=True,
                padding_value=0,
            ),
        }

        prompt_input_ids = batch.batch["input_ids"]
        prompt_attention_mask = batch.batch["attention_mask"]
        prompt_position_ids = batch.batch["position_ids"]
        if n > 1:
            prompt_input_ids = prompt_input_ids.repeat_interleave(n, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(n, dim=0)
            prompt_position_ids = prompt_position_ids.repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompt_input_ids, responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, responses["attention_mask"]], dim=1)
        # Multi-turn GRPO and the actor update expect a full-sequence
        # loss_mask. Prompt tokens and tool-call/tool-response tokens must not
        # contribute to the policy loss; response_mask already excludes the
        # latter, so prepend zeros for the prompt portion.
        prompt_loss_mask = torch.zeros_like(prompt_attention_mask)
        loss_mask = torch.cat([prompt_loss_mask, response_mask], dim=1)
        response_length = responses["input_ids"].size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(len(input_ids), -1)
        if prompt_position_ids.dim() == 3:
            delta_position_id = delta_position_id.unsqueeze(1).expand(-1, 3, -1)
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt_input_ids,  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "loss_mask": loss_mask,  # [bsz, prompt_length + response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "__num_turns__": num_turns,
                "rollout_multi_modal_inputs": np.array(rollout_multi_modal_inputs, dtype=object),
            },
        )

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        def response_role(response):
            content = response.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if response.get("role") == "tool" or "<tool_response>" in str(content):
                return "tool"
            return response.get("role")

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response_role(response) for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            for j in range(len(roles)):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        self.trace_rollouts = bool(config.trainer.get("rollout_data_dir", None))
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)
        max_concurrent_requests = int(self.config.multi_turn.get("max_concurrent_requests", 28))
        if max_concurrent_requests <= 0:
            raise ValueError(f"max_concurrent_requests must be positive, got {max_concurrent_requests}")
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        print(f"Chat scheduler max concurrent multi-turn requests: {max_concurrent_requests}", flush=True)

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any]):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info))

        # “fire-and-forget” background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        model_started = time.perf_counter()
        model_trace = {
            "turn": len(messages),
            "started_at_unix": time.time(),
            "server": address,
        }
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e
        finally:
            model_trace["latency_ms"] = round((time.perf_counter() - model_started) * 1000, 3)
            model_trace["status"] = "error" if exception is not None else "success"
            if exception is not None:
                model_trace["error"] = f"{type(exception).__name__}: {exception}"
            elif completions is not None:
                model_trace["finish_reason"] = str(completions.choices[0].finish_reason)
                if completions.usage is not None:
                    model_trace["prompt_tokens"] = completions.usage.prompt_tokens
                    model_trace["completion_tokens"] = completions.usage.completion_tokens
            if info.get("__trace__") is not None:
                info["__trace__"]["model_calls"].append(model_trace)

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()
            source_index = batch_index // n
            images = []
            if "origin_multi_modal_data" in batch.non_tensor_batch:
                mm_data = batch.non_tensor_batch["origin_multi_modal_data"][source_index]
                if isinstance(mm_data, dict):
                    images = list(mm_data.get("image") or [])
            _attach_images_to_messages(batch_conversations[batch_index], images)

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                        images=images,
                    )
                )
            )

        trajectory_traces = await asyncio.gather(*tasks)
        output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
        if self.trace_rollouts:
            output_batch.non_tensor_batch["rollout_trace"] = np.array(trajectory_traces, dtype=object)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _submit_chat_completions_semaphore(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        sampling_params: Dict[str, Any],
        images: List[Any] | None = None,
    ):
        # Hold one slot for the full trajectory, including all recursive tool
        # turns. This bounds vLLM and visual-tool load while allowing a large
        # global rollout batch to be processed in waves.
        queued_at = time.perf_counter()
        async with self.request_semaphore:
            trajectory_started = time.perf_counter()
            done = asyncio.Event()
            trace = None
            if self.trace_rollouts:
                trace = {
                    "started_at_unix": time.time(),
                    "queue_latency_ms": round((trajectory_started - queued_at) * 1000, 3),
                    "model_calls": [],
                    "tool_calls": [],
                }

            info = {
                "__done__": done,
                "__depth__": 0,  # indicate how many ongoing completion requests
                "__sampling_params__": sampling_params,
                "__trace__": trace,
                "images": images or [],
            }

            self.submit_chat_completions(messages=messages, request_id=request_id, info=info)

            # Wait until all completion requests are done
            await done.wait()
            if trace is None:
                return None

            active_latency_ms = (time.perf_counter() - trajectory_started) * 1000
            model_latency_ms = sum(item["latency_ms"] for item in trace["model_calls"])
            tool_latency_ms = sum(item["latency_ms"] for item in trace["tool_calls"])
            scheduler_latency_ms = max(0.0, active_latency_ms - model_latency_ms - tool_latency_ms)
            trace.update(
                {
                    "active_latency_ms": round(active_latency_ms, 3),
                    "total_latency_ms": round(active_latency_ms + trace["queue_latency_ms"], 3),
                    "model_latency_ms": round(model_latency_ms, 3),
                    "tool_latency_ms": round(tool_latency_ms, 3),
                    "scheduler_latency_ms": round(scheduler_latency_ms, 3),
                    "model_time_ratio": round(model_latency_ms / active_latency_ms, 6) if active_latency_ms else 0.0,
                    "tool_time_ratio": round(tool_latency_ms / active_latency_ms, 6) if active_latency_ms else 0.0,
                    "scheduler_time_ratio": round(scheduler_latency_ms / active_latency_ms, 6) if active_latency_ms else 0.0,
                    "model_call_count": len(trace["model_calls"]),
                    "tool_call_count": len(trace["tool_calls"]),
                }
            )
            return trace
