#!/usr/bin/env python3
"""Run the fine-tuned visual agent against an OpenAI-compatible model server.

The model server and visual-tool server are deliberately separate.  A vLLM
server provides chat completions, while an optional tool server executes SAM3
or GroundingDINO calls through the small HTTP contract implemented below.
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import json
import mimetypes
import os
import random
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
TOOL_CALL_ATTR_RE = re.compile(
    r'<tool_call\b(?=[^>]*\bfunction="([^"]+)")(?=[^>]*\barguments="([^"]*)")[^>]*>',
    re.DOTALL,
)


class InferenceError(RuntimeError):
    """Raised when model or tool inference cannot continue safely."""


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    # Set when a native call's arguments cannot be used; answered with an error observation.
    argument_error: str | None = None


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


def parse_tool_invocations(message: dict[str, Any]) -> list[ToolInvocation]:
    """Parse calls with the RL rollout's rules (chat_scheduler.ToolCompletionCallback).

    Native calls are all kept; bad arguments become an error observation. Text calls
    that are not a JSON object with a string name and object arguments are skipped.
    """
    native_calls = message.get("tool_calls") or []
    if native_calls:
        invocations = []
        for call in native_calls:
            function = call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except (TypeError, json.JSONDecodeError) as exc:
                invocations.append(ToolInvocation(function["name"], {}, call["id"], f"{type(exc).__name__}: {exc}"))
                continue
            if not isinstance(arguments, dict):
                invocations.append(ToolInvocation(function["name"], {}, call["id"], "Tool arguments must be a JSON object"))
                continue
            invocations.append(ToolInvocation(function["name"], arguments, call["id"]))
        return invocations

    action_text = re.sub(r"<think>.*?</think>", "", _message_text(message), flags=re.DOTALL)
    if "<think>" in action_text:
        return []
    raw_calls = TOOL_CALL_RE.findall(action_text)
    if raw_calls:
        payloads = []
        for raw_call in raw_calls:
            try:
                payloads.append(json.loads(raw_call))
            except json.JSONDecodeError:
                continue
        return [
            ToolInvocation(payload["name"], payload.get("arguments", {}))
            for payload in payloads
            if isinstance(payload, dict) and isinstance(payload.get("name"), str)
            and isinstance(payload.get("arguments", {}), dict)
        ]

    # Qwen attribute-style calls are an evaluation-only fallback; the RL rollout never parses them.
    invocations = []
    for name, raw_arguments in TOOL_CALL_ATTR_RE.findall(action_text):
        try:
            arguments = json.loads(html.unescape(raw_arguments))
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            invocations.append(ToolInvocation(html.unescape(name), arguments))
    return invocations


def load_training_tool_schemas(path: str | Path) -> list[dict[str, Any]]:
    """Dump tool-config schemas exactly as the RL rollout does, without importing the verl package."""
    import importlib.util

    import yaml

    schemas_file = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/schemas.py"
    spec = importlib.util.spec_from_file_location("verl_tool_schemas", schemas_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["tools"]
    return [
        module.OpenAIFunctionToolSchema.model_validate(tool["tool_schema"]).model_dump(
            exclude_unset=True, exclude_none=True
        )
        for tool in tools
    ]


@functools.cache
def _tool_faults():
    """Load the RL fault functions by path, without importing the verl package."""
    import importlib.util

    faults_file = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/tools/tool_faults.py"
    spec = importlib.util.spec_from_file_location("verl_tool_faults", faults_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _xml_tool_call(invocation: ToolInvocation) -> str:
    payload = {"name": invocation.name, "arguments": invocation.arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def build_system_prompt(allowed_tool_names: set[str] | None = None) -> str:
    schemas = json.dumps(
        get_visual_tool_schemas(allowed_tool_names), ensure_ascii=False, indent=2
    )
    return (
        "You are a visual agent. Answer the user's question from the supplied images. "
        "When closer inspection, grounding, segmentation, or counting is needed, call one "
        "visual tool and wait for its result. Emit a tool call exactly as "
        "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>. "
        "If a tool reports an error or cannot localize the target, adjust the query, "
        "try another available tool, or answer from the supplied image instead of "
        "repeating the same failed call. "
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
        allowed_tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ocr_raw_backslash = os.getenv("VISUAL_AGENT_OCR_RAW_BACKSLASH", "0") == "1"
        # Stress test: fault the first faultable result of these tools in each sample.
        self.fault_tools = {
            name.strip() for name in os.getenv("VISUAL_AGENT_FAULT_TOOLS", "").split(",") if name.strip()
        }
        self.fault_seed = os.getenv("VISUAL_AGENT_FAULT_SEED", "0")
        self.fault_type = os.getenv("VISUAL_AGENT_FAULT_TYPE", "training")
        self._fault_rng = random.Random(self.fault_seed)
        self._fault_pending = bool(self.fault_tools)
        self._faulted_call = None
        self.use_native_tools = use_native_tools
        self.allowed_tool_names = (
            set(allowed_tool_names) if allowed_tool_names is not None else None
        )
        if self.allowed_tool_names is not None:
            get_visual_tool_schemas(self.allowed_tool_names)
        self.tool_schemas = tool_schemas or get_visual_tool_schemas(self.allowed_tool_names)
        self.system_prompt = system_prompt or build_system_prompt(self.allowed_tool_names)

    def run(
        self, image_paths: list[str | Path], question: str,
        *, trace_sink: dict[str, Any] | None = None,
    ) -> InferenceResult:
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
        if trace_sink is not None:
            trace_sink.update(messages=messages, tool_calls=trace)
        # Seeding by question pairs a faulted run with its clean run on the same sample.
        self._fault_rng = random.Random(f"{self.fault_seed}:{question.strip()}")
        self._fault_pending = bool(self.fault_tools)
        self._faulted_call = None

        for turn in range(1, self.max_turns + 1):
            if trace_sink is not None:
                trace_sink["turn"] = turn
            assistant = self.model_client.chat(
                messages,
                tools=self.tool_schemas if self.use_native_tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response_text = _message_text(assistant)
            messages.append({"role": "assistant", "content": response_text})
            if trace_sink is not None:
                trace_sink["last_assistant"] = assistant
            invocations = parse_tool_invocations(assistant)
            native_calls = assistant.get("tool_calls")
            if native_calls:
                # Keep structured calls so the chat template renders history as the RL rollout does.
                messages[-1]["tool_calls"] = native_calls
            elif invocations and not response_text:
                response_text = "".join(_xml_tool_call(invocation) for invocation in invocations)
                messages[-1]["content"] = response_text

            if not invocations:
                return InferenceResult(
                    response=response_text,
                    turns=turn,
                    tool_calls=trace,
                    messages=messages,
                )
            # Like the RL rollout, answer every call, in order, before the next model turn.
            for invocation in invocations:
                messages.append(self._execute_tool(invocation, images, trace))

        return InferenceResult(
            response=f"Agent exceeded the maximum of {self.max_turns} turns",
            turns=self.max_turns,
            tool_calls=trace,
            messages=messages,
        )

    def _execute_tool(
        self, invocation: ToolInvocation, images: list[str], trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        error = invocation.argument_error
        if (
            error is None
            and self.allowed_tool_names is not None
            and invocation.name not in self.allowed_tool_names
        ):
            allowed = sorted(self.allowed_tool_names)
            error = f"Tool {invocation.name!r} is not allowed; available tools: {allowed}"
        if error is not None:
            trace.append({
                "name": invocation.name,
                "arguments": invocation.arguments,
                "error": error,
                "returned_images": 0,
            })
            return _observation(json.dumps({"status": "error", "error": error}, ensure_ascii=False), invocation)
        if self.tool_executor is None:
            raise InferenceError(
                f"Model requested {invocation.name!r}, but no visual tool service is configured. "
                "Set --tool-api-base or VISUAL_TOOL_API_BASE."
            )

        replay_arguments = invocation.arguments
        if invocation.name == "ocr_read":
            replay_arguments = {"mode": "text", "bbox_2d": [0, 0, 1000, 1000], **replay_arguments}
        if self._faulted_call is not None:
            name, arguments, fault, serialized = self._faulted_call
            if invocation.name == name and replay_arguments == arguments:
                trace.append({"name": name, "arguments": invocation.arguments, "fault": fault,
                              "replayed_fault": True, "returned_images": 0})
                return _observation(serialized, invocation)

        try:
            tool_result = self.tool_executor.execute(invocation, images)
        except InferenceError as exc:
            error_result = {"status": "error", "error": str(exc)}
            trace.append({
                "name": invocation.name,
                "arguments": invocation.arguments,
                "error": str(exc),
                "returned_images": 0,
            })
            return _observation(json.dumps(error_result, ensure_ascii=False), invocation)

        trace.append({
            "name": invocation.name,
            "arguments": invocation.arguments,
            "result": tool_result.output,
            "returned_images": len(tool_result.images),
        })
        output = tool_result.output
        if isinstance(output, dict) and output.get("status") not in {"error", "failed"}:
            if invocation.name == "grounding_detect":
                output = {key: output[key] for key in ("boxes", "confidence", "labels") if key in output}
            elif invocation.name == "crop_zoom":
                output = {"target_image": output["crop_zoom"]["target_image"]}
            elif invocation.name == "depth_measure":
                if "depths_m" in output:
                    output = {"regions": [
                        {"bbox_2d": box, "depth_m": depth}
                        for box, depth in zip(output["bboxes_2d"], output["depths_m"], strict=True)
                    ]}
                else:
                    output = {"regions": [{key: output[key] for key in ("bbox_2d", "depth_m")}]}
            elif invocation.name == "object_count":
                output = {"count": output["count"]}
            elif invocation.name == "ocr_read" and output.get("source") == "paddleocr_vl":
                output = {key: output[key] for key in ("text", "truncated") if key in output}
            elif invocation.name == "chart_parse":
                output = {key: output[key] for key in ("text", "truncated") if key in output}
            elif invocation.name in {"text_detect", "text_recognize"}:
                keys = ("bbox_2d", "text") if invocation.name == "text_recognize" else ("bbox_2d",)
                output = {"regions": [{key: region[key] for key in keys} for region in output["regions"]]}
            if self._fault_pending and invocation.name in self.fault_tools:
                faulty = _tool_faults().inject_fault(
                    invocation.name, output, self._fault_rng, variant=self.fault_type,
                )
                if faulty is not None:
                    self._fault_pending = False
                    trace[-1]["fault"] = {"original": output, "injected": faulty}
                    output = faulty
        serialized = (
            output
            if isinstance(output, str)
            else json.dumps(output, ensure_ascii=False)
        )
        if (
            self.ocr_raw_backslash and invocation.name == "ocr_read"
            and isinstance(output, dict) and "text" in output
            and output.get("status") not in {"error", "failed"}
        ):
            # Undo only doubled backslashes in the text field; keep other JSON escapes.
            encoded_text = json.dumps(output["text"], ensure_ascii=False)
            serialized = serialized.replace(
                '"text": ' + encoded_text,
                '"text": ' + encoded_text.replace("\\\\", "\\"),
                1,
            )
        if "fault" in trace[-1]:
            self._faulted_call = (invocation.name, dict(replay_arguments), trace[-1]["fault"], serialized)
        message = _observation(serialized, invocation)
        if tool_result.images:
            images.extend(tool_result.images)
            message["content"] = [
                {"type": "text", "text": message["content"]},
                *[
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in tool_result.images
                ],
            ]
        return message


def _observation(text: str, invocation: ToolInvocation) -> dict[str, Any]:
    if invocation.call_id is not None:
        # Native call: the chat template adds <tool_response> tags to tool-role messages.
        return {"role": "tool", "tool_call_id": invocation.call_id, "content": text}
    return {"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"}


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
