"""Bridge the Visual-Agent server to isolated VTS tool services.

The model-visible Visual-Agent protocol stays unchanged. This module only
translates the server-side transport from in-memory PIL images to VTS's
shared-filesystem ToolResult contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


class VtsBridgeError(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root = root.resolve()
    return resolved == root or root in resolved.parents


def _encode_image(path: Path) -> str:
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
    except OSError as exc:
        raise VtsBridgeError(f"could not read VTS output image {path}: {exc}") from exc
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@dataclass(frozen=True)
class VtsBridgeResult:
    result: dict[str, Any]
    images: list[str]


@dataclass(frozen=True)
class VtsRemoteBridge:
    endpoint: str
    tool_name: str
    shared_root: Path
    timeout: float = 300.0
    token: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        endpoint_env: str,
        tool_name: str,
    ) -> "VtsRemoteBridge | None":
        endpoint = os.environ.get(endpoint_env, "").strip()
        if not endpoint:
            return None
        configured_root = os.environ.get("VTS_TOOL_BRIDGE_ROOT", "").strip()
        if not configured_root:
            output_root = os.environ.get("VTS_OUTPUT_ROOT", "").strip()
            configured_root = (
                str(Path(output_root) / "visual_agent_bridge")
                if output_root
                else "/tmp/visual_agent_vts_bridge"
            )
        root = Path(configured_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        token_env = os.environ.get("VTS_TOOL_SERVICE_TOKEN_ENV", "VTS_TOOL_SERVICE_TOKEN")
        endpoints = [item.strip().rstrip("/") for item in endpoint.split(",") if item.strip()]
        if not endpoints:
            raise ValueError(f"{endpoint_env} must contain at least one endpoint")
        return cls(
            endpoint=",".join(endpoints),
            tool_name=tool_name,
            shared_root=root,
            timeout=float(os.environ.get("VTS_TOOL_TIMEOUT", "300")),
            token=os.environ.get(token_env) or None,
        )

    def execute(
        self,
        *,
        images: list[Image.Image],
        arguments: dict[str, Any],
        instance_id: str,
    ) -> VtsBridgeResult:
        if not images:
            raise VtsBridgeError("VTS tool call requires at least one image")
        safe_instance = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in instance_id)[:80]
        with tempfile.TemporaryDirectory(prefix=f"{safe_instance or 'request'}-", dir=self.shared_root) as temporary:
            request_root = Path(temporary)
            source_dir = request_root / "images"
            artifact_dir = request_root / "artifacts"
            source_dir.mkdir()
            artifact_dir.mkdir()
            image_records = []
            for index, image in enumerate(images):
                path = source_dir / f"image_{index}.png"
                image.convert("RGB").save(path, format="PNG")
                image_records.append(
                    {
                        "image_id": index,
                        "path": str(path),
                        "parent_image_id": None,
                        "source": "visual_agent",
                        "width": image.width,
                        "height": image.height,
                    }
                )
            payload = {
                "tool": self.tool_name,
                "args": arguments,
                "context": {
                    "uid": instance_id,
                    "images": image_records,
                    "artifact_dir": str(artifact_dir),
                    "branch_id": instance_id,
                },
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            endpoints = tuple(item.strip().rstrip("/") for item in self.endpoint.split(",") if item.strip())
            if not endpoints:
                raise VtsBridgeError("VTS service has no endpoints")
            start = int.from_bytes(hashlib.sha256(instance_id.encode()).digest()[:8], "big") % len(endpoints)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            errors = []
            for offset in range(len(endpoints)):
                endpoint = endpoints[(start + offset) % len(endpoints)]
                request = urllib.request.Request(f"{endpoint}/execute", data=data, headers=headers, method="POST")
                try:
                    with opener.open(request, timeout=self.timeout) as response:
                        body = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            else:
                raise VtsBridgeError(f"VTS services failed: {'; '.join(errors)}")
            if not isinstance(body, dict):
                raise VtsBridgeError("VTS service response must be a JSON object")

            returned_images = []
            for item in body.get("images") or []:
                if not isinstance(item, dict) or not item.get("path"):
                    raise VtsBridgeError("VTS service returned an invalid image record")
                path = Path(str(item["path"]))
                if not _within(path, request_root):
                    raise VtsBridgeError(f"VTS output image is outside the request directory: {path}")
                returned_images.append(_encode_image(path))
            return VtsBridgeResult(result=body, images=returned_images)
