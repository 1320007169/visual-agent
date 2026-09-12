# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Perceive2Reason Rollout Worker

Extends SGLangRollout to support perceive2reason training mode where:
1. For samples with correct perception but wrong reasoning, we fix the tool-calling prefix
2. Only the final reasoning/answer suffix participates in rollout sampling
3. Tool calls are prohibited during reasoning generation via logit bias

This allows focused optimization on reasoning while preserving learned perception skills.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Perceive2ReasonRollout(SGLangRollout):
    """
    Rollout worker for perceive2reason training.

    Key differences from standard SGLangRollout:
    - Detects samples with 'perception_prefix' field in non_tensor_batch
    - Injects fixed tool-calling history before rollout starts
    - Prohibits <tool_call> tokens via logit bias during final answer generation
    - Falls back to standard rollout for samples without prefix
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cache tool_call token IDs for logit bias
        self._tool_call_token_ids = self._get_tool_call_token_ids()
        self._p2r_mode_enabled = os.getenv("PERCEIVE2REASON_MODE", "false").lower() == "true"

        if self._p2r_mode_enabled:
            logger.info(f"Perceive2ReasonRollout initialized with tool_call_token_ids={self._tool_call_token_ids}")
        else:
            logger.info("Perceive2ReasonRollout initialized but P2R mode is disabled (set PERCEIVE2REASON_MODE=true to enable)")

    def _get_tool_call_token_ids(self) -> list[int]:
        """Get all token IDs that could start a tool call."""
        tool_call_strings = ["<tool_call>", "<tool", "tool_call", "tool"]
        token_ids = []
        for s in tool_call_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if isinstance(ids, list):
                token_ids.extend(ids)
            else:
                token_ids.append(int(ids))
        # Remove duplicates
        return list(set(token_ids))

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs
    ) -> AsyncRolloutRequest:
        """
        Override to detect and handle perceive2reason samples.

        If req has 'perception_prefix' in meta_info:
          1. Inject fixed prefix messages into req.messages
          2. Set req state to skip tool calling phase
          3. Generate only the final answer with tool prohibition
        Else:
          Fall back to standard async rollout
        """
        perception_prefix = req.meta_info.get("perception_prefix", None)

        if not self._p2r_mode_enabled or perception_prefix is None:
            # Standard rollout for normal samples or when P2R is disabled
            return await super()._async_rollout_a_request(req, do_sample, is_validate, **kwargs)

        # Perceive2reason mode
        logger.debug(f"Processing perceive2reason sample {req.request_id} with prefix of {len(perception_prefix)} messages")

        # Inject fixed perception prefix into request messages
        self._inject_perception_prefix(req, perception_prefix)

        # Mark that we should prohibit tool calls in the next generation
        req.meta_info["prohibit_tool_calls"] = True

        # Set state to RUNNING (skip PENDING tool creation, go straight to generation)
        req.state = AsyncRolloutRequestStateEnum.RUNNING

        # Call parent's rollout logic, which will now generate with the fixed prefix
        # The _handle_engine_call override will add logit bias to block tool calls
        return await super()._async_rollout_a_request(req, do_sample, is_validate, **kwargs)

    def _inject_perception_prefix(
        self,
        req: AsyncRolloutRequest,
        perception_prefix: list[dict]
    ) -> None:
        """
        Inject fixed perception prefix into request messages.

        Modifies req.messages in-place to add tool call history before generation.

        Original messages:
          [system, user]

        After injection:
          [system, user, assistant(tool_call), tool(response), assistant(tool_call), tool(response), ...]
        """
        # perception_prefix format: [{"role": "assistant", "content": "<tool_call>..."}, {"role": "tool", "content": "<tool_response>..."}, ...]
        for prefix_msg in perception_prefix:
            msg = Message(
                role=prefix_msg["role"],
                content=prefix_msg["content"]
            )
            req.messages.append(msg)

        logger.debug(f"Injected {len(perception_prefix)} prefix messages into request {req.request_id}")

    async def _handle_engine_call(
        self,
        _req: AsyncRolloutRequest,
        do_sample: bool,
        is_validate: bool,
        override_n: bool = True,
        **kwargs
    ) -> dict:
        """
        Override to add logit bias for tool call prohibition in P2R mode.

        When req.meta_info["prohibit_tool_calls"] is True:
          - Add logit_bias to sampling params to block <tool_call> tokens
          - This forces the model to generate final answer directly
        """
        # Check if we should prohibit tool calls
        prohibit_tool_calls = _req.meta_info.get("prohibit_tool_calls", False)

        if prohibit_tool_calls:
            # Build logit bias to block tool call tokens
            logit_bias = {token_id: -100.0 for token_id in self._tool_call_token_ids}

            # Add logit_bias to kwargs, which will be passed to update_sampling_params
            # The parent's _handle_engine_call uses: with self.update_sampling_params(**kwargs)
            # So we need to ensure logit_bias is in kwargs AND in self.sampling_params
            kwargs["logit_bias"] = logit_bias

            # Also pre-set it in self.sampling_params to ensure it's there
            # because update_sampling_params only updates existing keys
            if "logit_bias" not in self.sampling_params:
                self.sampling_params["logit_bias"] = {}

            logger.debug(f"Adding logit_bias to prohibit tool calls for request {_req.request_id}: {len(logit_bias)} tokens blocked")

        # Call parent's engine call with potentially modified kwargs
        return await super()._handle_engine_call(_req, do_sample, is_validate, override_n, **kwargs)

    def _preprocess_prompt_to_async_rollout_requests(
        self,
        prompts,
        n: int = 1,
    ) -> list[AsyncRolloutRequest]:
        """
        Override to extract perception_prefix from non_tensor_batch and attach to requests.

        Expected non_tensor_batch format:
          {
              "perception_prefix": [
                  [{"role": "assistant", "content": "..."}, {"role": "tool", "content": "..."}],  # sample 0
                  None,  # sample 1 (no prefix, standard rollout)
                  [{"role": "assistant", "content": "..."}, ...],  # sample 2
                  ...
              ]
          }
        """
        req_list = super()._preprocess_prompt_to_async_rollout_requests(prompts, n)

        # Extract perception_prefix if present
        non_tensor_batch = prompts.non_tensor_batch
        perception_prefixes = non_tensor_batch.get("perception_prefix", None)

        if perception_prefixes is not None:
            # Match each request with its perception prefix
            for req in req_list:
                # req.batch_data_id is the index in the batch
                prefix = perception_prefixes[req.batch_data_id]
                if prefix is not None:
                    req.meta_info["perception_prefix"] = prefix
                    logger.debug(f"Attached perception_prefix to request {req.request_id} (batch_id={req.batch_data_id})")

        return req_list
