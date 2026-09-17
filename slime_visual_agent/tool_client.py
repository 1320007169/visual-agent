"""Async client for the existing Visual-Agent HTTP tool service."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp


_PENDING_BY_ENDPOINT: dict[str, int] = {}


def _least_busy_endpoint(urls: tuple[str, ...], failed_url: str | None = None) -> str:
    candidates = tuple(url for url in urls if url != failed_url) or urls
    return min(candidates, key=lambda url: _PENDING_BY_ENDPOINT.get(url, 0))


@dataclass(frozen=True)
class ToolResult:
    output: str
    images: list[str]
    metrics: dict[str, Any]


def _base_urls() -> tuple[str, ...]:
    raw = os.environ.get("VISUAL_TOOL_API_BASES") or os.environ.get("VISUAL_TOOL_API_BASE", "")
    urls = tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    if not urls:
        raise RuntimeError("set VISUAL_TOOL_API_BASES to one or more running visual-tool endpoints")
    return urls


async def execute_visual_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    images: list[str],
    instance_id: str,
) -> ToolResult:
    """Execute a tool with bounded retries; returned images stay as data URLs."""
    urls = _base_urls()
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("VISUAL_TOOL_TIMEOUT", "300")))
    retries = max(0, int(os.environ.get("VISUAL_TOOL_MAX_RETRIES", "2")))
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("VISUAL_TOOL_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "instance_id": instance_id,
        "name": name,
        "arguments": arguments,
        "images": images,
    }

    last_error: Exception | None = None
    failed_url: str | None = None
    for attempt in range(retries + 1):
        base_url = _least_busy_endpoint(urls, failed_url)
        _PENDING_BY_ENDPOINT[base_url] = _PENDING_BY_ENDPOINT.get(base_url, 0) + 1
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{base_url}/execute", json=payload, headers=headers) as response:
                    body = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(f"visual tool HTTP {response.status}: {body[:1000]}")
                    result = json.loads(body)
            if result.get("status") in {"error", "failed"}:
                raise RuntimeError(str(result.get("error") or result))
            returned_images = result.get("images") or []
            if not isinstance(returned_images, list) or not all(isinstance(item, str) for item in returned_images):
                raise RuntimeError("visual tool returned a malformed images list")
            output = result.get("result", result.get("output", result))
            output_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            metrics = dict(result.get("metrics") or {})
            metrics.update({"tool": name, "endpoint": base_url, "returned_image_count": len(returned_images)})
            return ToolResult(output=output_text, images=returned_images, metrics=metrics)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            failed_url = base_url
            if attempt < retries:
                await asyncio.sleep(min(2**attempt, 5))
        finally:
            _PENDING_BY_ENDPOINT[base_url] -= 1
    raise RuntimeError(f"visual tool {name} failed after {retries + 1} attempts: {last_error}")
