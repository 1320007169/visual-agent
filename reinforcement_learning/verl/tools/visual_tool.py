# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
import base64
import io
import json
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

import aiohttp

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


# One rollout may call different visual tools in sequence. All tool objects
# therefore share the image list for the same rollout/request instance ID.
_ONLINE_VISUAL_TOOL_INSTANCES: dict[str, dict[str, Any]] = {}
_ONLINE_VISUAL_TOOL_PENDING: dict[tuple[str, ...], dict[str, int]] = {}


class OfflineVisualTool(BaseTool):
    """Schema-only visual tool for offline SFT and parser compatibility.

    This class intentionally does not connect to SAM3 or GroundingDINO. Offline
    SFT samples already contain tool observations, so training only needs the
    tool names and schemas to be valid.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"calls": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["calls"].append(parameters)
        response = {
            "error": "offline_visual_tool_has_no_executor",
            "tool": self.name,
            "message": "Visual tool execution is disabled for offline SFT.",
        }
        return json.dumps(response), 0.0, {"offline": True, "tool": self.name}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


def _image_to_data_url(image: Any) -> str:
    """Serialize a PIL image, bytes, path, or existing URL for the HTTP API."""
    if isinstance(image, str):
        if image.startswith(("data:image/", "http://", "https://")):
            return image
        with open(os.path.expanduser(image), "rb") as image_file:
            data = image_file.read()
        suffix = os.path.splitext(image)[1].lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
    elif isinstance(image, bytes):
        data = image
        mime = "image/jpeg"
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        mime = "image/png"
    else:
        raise TypeError(f"Unsupported visual-tool image type: {type(image).__name__}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class OnlineVisualTool(BaseTool):
    """Execute a visual tool through the shared online HTTP tool service.

    The rollout scheduler binds the original sample images in ``create``. Each
    call then sends the tool name, arguments, and those images to ``POST
    /execute``. SAM3 and GroundingDINO can therefore be deployed independently
    from the RL workers.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        configured_urls = config.get("base_urls") or config.get("base_url", "http://127.0.0.1:9000")
        if isinstance(configured_urls, str):
            configured_urls = configured_urls.split(",")
        self.base_urls = tuple(str(url).strip().rstrip("/") for url in configured_urls if str(url).strip())
        if not self.base_urls:
            raise ValueError("OnlineVisualTool requires at least one base URL")
        self._pending = _ONLINE_VISUAL_TOOL_PENDING.setdefault(
            self.base_urls, {base_url: 0 for base_url in self.base_urls}
        )
        self.api_key = config.get("api_key") or None
        self.timeout = float(config.get("timeout", 300.0))
        self.max_retries = int(config.get("max_retries", 2))
        self._instance_dict = _ONLINE_VISUAL_TOOL_INSTANCES

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = instance_id or str(uuid4())
        images = kwargs.get("images") or []
        state = self._instance_dict.setdefault(instance_id, {"images": [], "calls": []})
        if images and not state["images"]:
            state["images"].extend(_image_to_data_url(image) for image in images)
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        if instance_id not in self._instance_dict:
            raise KeyError(f"Unknown visual-tool instance: {instance_id}")

        state = self._instance_dict[instance_id]
        payload = {
            "instance_id": instance_id,
            "name": self.name,
            "arguments": parameters,
            "images": state["images"],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        error: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        selected_base_url = self.base_urls[0]
        failed_base_url: str | None = None
        for attempt in range(self.max_retries + 1):
            candidates = tuple(url for url in self.base_urls if url != failed_base_url) or self.base_urls
            selected_base_url = min(candidates, key=lambda url: self._pending[url])
            self._pending[selected_base_url] += 1
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{selected_base_url}/execute", json=payload, headers=headers
                    ) as response:
                        response_text = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"visual tool HTTP {response.status}: {response_text[:1000]}")
                        result = json.loads(response_text)
                if result.get("status") in {"error", "failed"}:
                    raise RuntimeError(str(result.get("error") or result))
                break
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                error = exc
                failed_base_url = selected_base_url
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Online visual tool {self.name} failed: {exc}") from exc
                await asyncio.sleep(min(2**attempt, 5))
            finally:
                self._pending[selected_base_url] -= 1
        else:  # pragma: no cover - loop always raises or breaks
            raise RuntimeError(f"Online visual tool {self.name} failed: {error}")

        returned_images = result.get("images") or []
        state["images"].extend(returned_images)
        state["calls"].append(parameters)
        output = result.get("result", result.get("output", result))
        response_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        metrics = dict(result.get("metrics") or {})
        metrics.update(
            {
                "online": True,
                "tool": self.name,
                "endpoint": selected_base_url,
                "returned_images": returned_images,
            }
        )
        return response_text, float(result.get("reward", 0.0)), metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
