#!/usr/bin/env python3
"""Serve a local Qwen3-VL checkpoint through a minimal OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 4096


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        items: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                items.append({"type": "image", "image": url})
            else:
                items.append(item)
        normalized.append({**message, "content": items})
    return normalized


def create_app(model_path: str, served_model_name: str) -> FastAPI:
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()
    model.config.use_cache = True
    lock = asyncio.Lock()
    app = FastAPI(title="Visual Agent Transformers Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": served_model_name}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": served_model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        async with lock:
            try:
                content, prompt_tokens, completion_tokens = await asyncio.to_thread(
                    generate, model, processor, normalize_messages(request.messages),
                    request.max_tokens, request.temperature,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def generate(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generation_args: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_args["temperature"] = temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_args)
    prompt_tokens = inputs.input_ids.shape[1]
    completion = generated[:, prompt_tokens:]
    text = processor.batch_decode(completion, skip_special_tokens=True)[0]
    return text, prompt_tokens, completion.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="visual-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(args.model, args.served_model_name),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
