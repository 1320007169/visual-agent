#!/usr/bin/env python3
"""Run the fine-tuned visual agent against an OpenAI-compatible model server.

The model server and visual-tool server are deliberately separate.  A vLLM
server provides chat completions, while an optional tool server executes SAM3
or GroundingDINO calls through the small HTTP contract implemented below.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from uuid import uuid4
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

try:
    from .visual_tools import get_visual_tool_schemas
except ImportError:
    from visual_tools import get_visual_tool_schemas


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class InferenceError(RuntimeError):
    """Raised when model or tool inference cannot continue safely."""


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolExecutionResult:
    output: Any
    images: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    response: str
    turns: int
    tool_calls: list[dict[str, Any]]
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]: ...


class ToolExecutor(Protocol):
    def execute(
        self,
        invocation: ToolInvocation,
        images: list[str],
    ) -> ToolExecutionResult: ...


def image_to_data_url(path: str | Path) -> str:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise InferenceError(f"Image does not exist: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return "" if content is None else str(content)


def parse_tool_invocation(message: dict[str, Any]) -> ToolInvocation | None:
    native_calls = message.get("tool_calls") or []
    if native_calls:
        function = native_calls[-1].get("function", {})
        name = function.get("name")
        raw_arguments = function.get("arguments", {})
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        return _validate_tool_invocation(name, arguments)

    matches = TOOL_CALL_RE.findall(_message_text(message))
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Model returned invalid <tool_call> JSON: {exc}") from exc
    return _validate_tool_invocation(payload.get("name"), payload.get("arguments", {}))


def _validate_tool_invocation(name: Any, arguments: Any) -> ToolInvocation:
    if not isinstance(name, str) or not name:
        raise InferenceError("Tool call is missing a non-empty name")
    if not isinstance(arguments, dict):
        raise InferenceError(f"Tool arguments for {name!r} must be a JSON object")
    known_names = {
        schema["function"]["name"] for schema in get_visual_tool_schemas()
    }
    if name not in known_names:
        raise InferenceError(f"Model requested unsupported tool {name!r}")
    return ToolInvocation(name=name, arguments=arguments)


def _xml_tool_call(invocation: ToolInvocation) -> str:
    payload = {"name": invocation.name, "arguments": invocation.arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def build_system_prompt() -> str:
    schemas = json.dumps(get_visual_tool_schemas(), ensure_ascii=False, indent=2)
    return (
        "You are a visual agent. Answer the user's question from the supplied images. "
        "When closer inspection, grounding, segmentation, or counting is needed, call one "
        "visual tool and wait for its result. Emit a tool call exactly as "
        "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>. "
        "After receiving <tool_response>, continue reasoning and give the final answer inside "
        "<answer>...</answer>. Available tools:\n"
        f"{schemas}"
    )


class OpenAICompatibleModelClient:
    """Minimal chat-completions client for vLLM/SGLang/OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        model: str | None = None,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        response = self.session.get(
            f"{self.base_url}/models",
            headers=self.headers,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "list models")
        models = response.json().get("data", [])
        if not models or not models[0].get("id"):
            raise InferenceError("Model server returned no models from /models")
        self.model = str(models[0]["id"])
        return self.model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "create chat completion")
        choices = response.json().get("choices", [])
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise InferenceError("Model server returned no assistant message")
        return choices[0]["message"]

    @staticmethod
    def _raise_for_status(response: requests.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = response.text[:1000] if response.text else str(exc)
            raise InferenceError(f"Failed to {operation}: {detail}") from exc


class HTTPVisualToolExecutor:
    """Call a separately deployed SAM3/GroundingDINO tool service.

    Request contract::

        POST {base_url}/execute
        {"name": str, "arguments": object, "images": [data_url, ...]}

    The response must contain ``result`` (any JSON value) and may contain an
    ``images`` array of data URLs or raw base64 strings.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()
        self.instance_id = str(uuid4())

    def execute(
        self,
        invocation: ToolInvocation,
        images: list[str],
    ) -> ToolExecutionResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.post(
            f"{self.base_url}/execute",
            headers=headers,
            json={
                "instance_id": self.instance_id,
                "name": invocation.name,
                "arguments": invocation.arguments,
                "images": images,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = response.text[:1000] if response.text else str(exc)
            raise InferenceError(f"Visual tool request failed: {detail}") from exc

        payload = response.json()
        if payload.get("status") in {"error", "failed"}:
            raise InferenceError(f"Visual tool failed: {payload.get('error') or payload}")
        returned_images = [self._normalize_image(item) for item in payload.get("images", [])]
        output = payload.get("result", payload.get("output", payload))
        return ToolExecutionResult(output=output, images=returned_images)

    @staticmethod
    def _normalize_image(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("data_url") or value.get("url") or value.get("base64")
        if not isinstance(value, str) or not value:
            raise InferenceError("Tool service returned an invalid image")
        if value.startswith("data:image/"):
            return value
        return f"data:image/jpeg;base64,{value}"


class VisualAgent:
    def __init__(
        self,
        model_client: ModelClient,
        *,
        tool_executor: ToolExecutor | None = None,
        max_turns: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        use_native_tools: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_native_tools = use_native_tools
        self.system_prompt = system_prompt or build_system_prompt()

    def run(self, image_paths: list[str | Path], question: str) -> InferenceResult:
        if not image_paths:
            raise InferenceError("At least one image is required")
        if not question.strip():
            raise InferenceError("Question must not be empty")

        images = [image_to_data_url(path) for path in image_paths]
        placeholders = "\n".join("<image>" for _ in images)
        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
        user_content.append({"type": "text", "text": f"{placeholders}\n{question.strip()}"})
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        trace: list[dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            assistant = self.model_client.chat(
                messages,
                tools=get_visual_tool_schemas() if self.use_native_tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            invocation = parse_tool_invocation(assistant)
            response_text = _message_text(assistant)
            if invocation and not response_text:
                response_text = _xml_tool_call(invocation)
            messages.append({"role": "assistant", "content": response_text})

            if invocation is None:
                return InferenceResult(
                    response=response_text,
                    turns=turn,
                    tool_calls=trace,
                    messages=messages,
                )
            if self.tool_executor is None:
                raise InferenceError(
                    f"Model requested {invocation.name!r}, but no visual tool service is configured. "
                    "Set --tool-api-base or VISUAL_TOOL_API_BASE."
                )

            tool_result = self.tool_executor.execute(invocation, images)
            trace.append({
                "name": invocation.name,
                "arguments": invocation.arguments,
                "result": tool_result.output,
                "returned_images": len(tool_result.images),
            })
            serialized = (
                tool_result.output
                if isinstance(tool_result.output, str)
                else json.dumps(tool_result.output, ensure_ascii=False)
            )
            tool_text = f"<tool_response>\n{serialized}\n</tool_response>"
            if tool_result.images:
                images.extend(tool_result.images)
                content: str | list[dict[str, Any]] = [
                    {"type": "text", "text": tool_text},
                    *[
                        {"type": "image_url", "image_url": {"url": image}}
                        for image in tool_result.images
                    ],
                ]
            else:
                content = tool_text
            messages.append({"role": "user", "content": content})

        raise InferenceError(f"Agent exceeded the maximum of {self.max_turns} turns")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, help="Input image path; repeat for multiple images")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--api-base",
        default=os.getenv("VISUAL_AGENT_API_BASE", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("VISUAL_AGENT_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("VISUAL_AGENT_MODEL"))
    parser.add_argument("--tool-api-base", default=os.getenv("VISUAL_TOOL_API_BASE"))
    parser.add_argument("--tool-api-key", default=os.getenv("VISUAL_TOOL_API_KEY"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--native-tools", action="store_true", help="Send schemas through the OpenAI tools field")
    parser.add_argument("--json", action="store_true", help="Print the complete trace as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_client = OpenAICompatibleModelClient(
        args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )
    tool_executor = None
    if args.tool_api_base:
        tool_executor = HTTPVisualToolExecutor(
            args.tool_api_base,
            api_key=args.tool_api_key,
            timeout=args.timeout,
        )
    agent = VisualAgent(
        model_client,
        tool_executor=tool_executor,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_native_tools=args.native_tools,
    )
    try:
        result = agent.run(args.image, args.question)
    except (InferenceError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.json else result.response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
