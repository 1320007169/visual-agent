"""VLMEvalKit API adapter for the local visual-agent tool loop."""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

from .base import BaseAPI


VISUAL_AGENT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = VISUAL_AGENT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from visual_agent_inference import (  # noqa: E402
    HTTPVisualToolExecutor,
    OpenAICompatibleModelClient,
    VisualAgent,
    image_to_data_url,
)


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _split_bases(value: str | list[str]) -> list[str]:
    values = value if isinstance(value, list) else value.split(",")
    return [item.strip().rstrip("/") for item in values if item.strip()]


def _model_api_base(value: str) -> str:
    return value if value.endswith("/v1") else f"{value}/v1"


def _redact_data_urls(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return f"<image data URL omitted: {len(value)} chars>"
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_data_urls(item) for key, item in value.items()}
    return value


class VisualAgentAPI(BaseAPI):
    """Run VLMEvalKit samples through vLLM plus the visual-tool HTTP server."""

    is_api = True

    def __init__(
        self,
        model: str | None = None,
        api_base: str | list[str] = "http://127.0.0.1:8000/v1",
        tool_api_base: str | list[str] = "http://127.0.0.1:9000",
        api_key: str = "EMPTY",
        tool_api_key: str | None = None,
        retry: int = 2,
        wait: int = 1,
        timeout: float = 600,
        max_turns: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0,
        use_tools: bool = True,
        use_native_tools: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_bases = [_model_api_base(item) for item in _split_bases(api_base)]
        self.tool_api_bases = _split_bases(tool_api_base)
        if not self.api_bases:
            raise ValueError("api_base must contain at least one model endpoint")
        if not self.tool_api_bases:
            raise ValueError("tool_api_base must contain at least one visual-tool endpoint")
        self.api_key = api_key
        self.tool_api_key = tool_api_key
        self.timeout = timeout
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_tools = use_tools
        self.use_native_tools = use_native_tools
        self._endpoint_index = 0
        self._endpoint_lock = threading.Lock()
        super().__init__(retry=retry, wait=wait, verbose=verbose, **kwargs)

    def set_inference_mode(self, mode: str) -> None:
        expected_mode = "agent" if self.use_tools else "non-think"
        if mode != expected_mode:
            self.logger.warning(
                "VisualAgentAPI is configured for %s inference; ignoring inference mode %r",
                expected_mode,
                mode,
            )

    def _next_endpoints(self) -> tuple[str, str]:
        with self._endpoint_lock:
            index = self._endpoint_index
            self._endpoint_index += 1
        return (
            self.api_bases[index % len(self.api_bases)],
            self.tool_api_bases[index % len(self.tool_api_bases)],
        )

    @staticmethod
    def _unpack_inputs(inputs: list[dict[str, Any]]) -> tuple[list[str], str]:
        images = [str(item["value"]) for item in inputs if item.get("type") == "image"]
        text = "\n".join(
            str(item["value"]) for item in inputs if item.get("type") == "text"
        ).strip()
        if not images:
            raise ValueError("VisualAgentAPI requires at least one image")
        if not text:
            raise ValueError("VisualAgentAPI requires a non-empty benchmark prompt")
        return images, text

    def generate_inner(self, inputs: list[dict[str, Any]], **kwargs: Any):
        images, question = self._unpack_inputs(inputs)
        model_base, tool_base = self._next_endpoints()
        model_client = OpenAICompatibleModelClient(
            model_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
        )
        tool_executor = HTTPVisualToolExecutor(
            tool_base,
            api_key=self.tool_api_key,
            timeout=self.timeout,
        )
        if not self.use_tools:
            content = [
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
                for path in images
            ]
            content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": content}]
            assistant = model_client.chat(
                messages,
                tools=None,
                temperature=float(kwargs.pop("temperature", self.temperature)),
                max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            )
            answer = assistant.get("content") or assistant.get("reasoning_content") or ""
            answer = str(answer).strip()
            messages.append({"role": "assistant", "content": answer})
            trace = {
                "response": answer,
                "turns": 1,
                "tool_calls": [],
                "messages": messages,
            }
            return 0, {
                "response": answer,
                "raw_response": json.dumps(_redact_data_urls(trace), ensure_ascii=False),
                "image_path_list": images,
            }, trace

        agent = VisualAgent(
            model_client,
            tool_executor=tool_executor,
            max_turns=int(kwargs.pop("max_turns", self.max_turns)),
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            temperature=float(kwargs.pop("temperature", self.temperature)),
            use_native_tools=self.use_native_tools,
        )
        result = agent.run(images, question)
        match = ANSWER_RE.search(result.response)
        answer = (match.group(1) if match else result.response).strip()
        trace = _redact_data_urls(result.to_dict())
        response = {
            "response": answer,
            "raw_response": json.dumps(trace, ensure_ascii=False),
            "image_path_list": images,
        }
        return 0, response, trace


__all__ = ["VisualAgentAPI"]
