#!/usr/bin/env python3
"""Serve SAM3 and GroundingDINO behind the visual-agent /execute contract."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import io
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sft_tool_call_logic import execute_tool_call
from vts_tool_bridge import VtsBridgeError, VtsRemoteBridge


SUPPORTED_TOOLS = {
    "crop_zoom",
    "sam3_segment_multi",
    "grounding_detect",
    "sam3_crop_zoom",
    "sam3_crop_zoom_multi",
    "ocr_read",
    "depth_measure",
    "ground_depth",
    "object_count",
}


class ToolServerError(RuntimeError):
    pass


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("data_url") or value.get("base64") or value.get("url")
    if not isinstance(value, str):
        raise ToolServerError("Each image must be a data URL or base64 string")
    if value.startswith("data:image/"):
        try:
            value = value.split(",", 1)[1]
        except IndexError as exc:
            raise ToolServerError("Malformed image data URL") from exc
    try:
        raw = base64.b64decode(value, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ToolServerError(f"Could not decode request image: {exc}") from exc


def encode_image(image: Image.Image, *, image_format: str = "JPEG", quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, quality=quality)
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _normalize_boxes(boxes: Any, width: int, height: int) -> list[list[float]]:
    normalized = []
    for box in _to_list(boxes):
        coords = [float(x) for x in box]
        if len(coords) != 4:
            continue
        if max(abs(x) for x in coords) <= 1.5:
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
        normalized.append([round(x, 4) for x in coords])
    return normalized


def _mask_areas(masks: Any) -> list[int]:
    if masks is None:
        return []
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu()
    areas = []
    for mask in masks:
        if hasattr(mask, "sum"):
            area = mask.sum().item()
        else:
            area = sum(bool(pixel) for row in mask for pixel in row)
        areas.append(int(area))
    return areas


def _load_object(path: str):
    module_name, object_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def _resolve_sam3_image_checkpoint(model_path: str | None) -> str | None:
    if not model_path:
        return None
    checkpoint = Path(model_path).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "sam3.pt"
    if "multiplex" in checkpoint.name:
        raise ToolServerError(
            "SAM3.1 multiplex checkpoints are for video tracking and cannot back the current image tools; "
            "set SAM3_MODEL_PATH to the SAM3 image checkpoint sam3.pt"
        )
    if not checkpoint.is_file():
        raise ToolServerError(f"SAM3 image checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


class Sam3Backend:
    """Official SAM3 image processor adapter with optional custom factory."""

    def __init__(self, model_path: str | None, device: str, threshold: float):
        import torch

        self.torch = torch
        self.device = device
        self.threshold = threshold
        checkpoint_path = _resolve_sam3_image_checkpoint(model_path)
        factory_path = os.getenv("SAM3_FACTORY")
        if factory_path:
            built = _load_object(factory_path)(model_path=checkpoint_path, device=device)
            if isinstance(built, tuple):
                self.model, self.processor = built
            else:
                self.model, self.processor = built, None
        else:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            signature = inspect.signature(build_sam3_image_model)
            kwargs: dict[str, Any] = {}
            if "device" in signature.parameters:
                kwargs["device"] = device
            if checkpoint_path:
                for name in ("checkpoint_path", "ckpt_path", "model_path"):
                    if name in signature.parameters:
                        kwargs[name] = checkpoint_path
                        break
                if "load_from_HF" in signature.parameters:
                    kwargs["load_from_HF"] = False
            elif "load_from_HF" in signature.parameters:
                kwargs["load_from_HF"] = True
            self.model = build_sam3_image_model(**kwargs)
            self.processor = Sam3Processor(self.model, confidence_threshold=threshold)

        if hasattr(self.model, "to"):
            self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if self.processor is None:
            raise ToolServerError("SAM3_FACTORY must return (model, processor)")
        self.lock = threading.Lock()

    def segment(self, image: Image.Image, query: str) -> dict[str, Any]:
        device_type = self.device.split(":", 1)[0]
        with self.lock, self.torch.inference_mode(), self.torch.autocast(
            device_type=device_type,
            dtype=self.torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            state = self.processor.set_image(image)
            try:
                output = self.processor.set_text_prompt(state=state, prompt=query)
            except TypeError:
                output = self.processor.set_text_prompt(state, query)

        boxes = _normalize_boxes(output.get("boxes"), *image.size)
        scores = [round(float(x), 6) for x in _to_list(output.get("scores"))]
        masks = output.get("masks")
        areas = _mask_areas(masks)
        keep = [i for i, score in enumerate(scores) if score >= self.threshold]
        if not scores:
            keep = list(range(len(boxes)))
        return {
            "boxes": [boxes[i] for i in keep if i < len(boxes)],
            "confidence": [scores[i] for i in keep if i < len(scores)],
            "mask_area_px": [areas[i] for i in keep if i < len(areas)],
        }


class GroundingDinoBackend:
    def __init__(self, model_path: str, device: str, box_threshold: float, text_threshold: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device).eval()
        self.lock = threading.Lock()

    def detect(self, image: Image.Image, query: str) -> dict[str, Any]:
        # GroundingDINO's text encoder has a 256-token limit. Long model
        # queries otherwise fail inside the encoder and turn into repeated
        # HTTP 500s during a rollout.
        prompt = " ".join(str(query).split()).strip().rstrip(".") + "."
        if prompt == ".":
            raise ValueError("grounding_detect query must not be empty")
        max_text_tokens = int(os.getenv("GROUNDING_DINO_MAX_TEXT_TOKENS", "256"))
        if max_text_tokens < 1:
            raise ValueError("GROUNDING_DINO_MAX_TEXT_TOKENS must be positive")
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_text_tokens,
            ).to(self.device)
            outputs = self.model(**inputs)
            postprocess = self.processor.post_process_grounded_object_detection
            threshold_name = (
                "box_threshold"
                if "box_threshold" in inspect.signature(postprocess).parameters
                else "threshold"
            )
            result = postprocess(
                outputs,
                inputs.input_ids,
                **{
                    threshold_name: self.box_threshold,
                    "text_threshold": self.text_threshold,
                    "target_sizes": [image.size[::-1]],
                },
            )[0]
        boxes = _normalize_boxes(result.get("boxes"), *image.size)
        scores = [round(float(x), 6) for x in _to_list(result.get("scores"))]
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels")
        if labels is None:
            labels = [query] * len(boxes)
        labels = [str(x) for x in _to_list(labels)]
        return {"boxes": boxes, "confidence": scores, "labels": labels}


class BackendPool:
    """Dispatch blocking inference calls across independent model replicas."""

    def __init__(self, backends: list[Any]):
        if not backends:
            raise ValueError("BackendPool requires at least one backend")
        self.backends = backends
        self.acquire_timeout = float(os.getenv("VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT", "60"))
        if self.acquire_timeout <= 0:
            raise ValueError("VISUAL_TOOL_BACKEND_ACQUIRE_TIMEOUT must be positive")
        self._available: queue.Queue[Any] = queue.Queue()
        for backend in backends:
            self._available.put(backend)

    @property
    def replica_count(self) -> int:
        return len(self.backends)

    def _call(self, method: str, *args, **kwargs):
        try:
            backend = self._available.get(timeout=self.acquire_timeout)
        except queue.Empty as exc:
            raise ToolServerError(
                f"Timed out waiting {self.acquire_timeout:.1f}s for a {method} backend replica"
            ) from exc
        try:
            return getattr(backend, method)(*args, **kwargs)
        finally:
            self._available.put(backend)

    def segment(self, image: Image.Image, query: str) -> dict[str, Any]:
        return self._call("segment", image, query)

    def detect(self, image: Image.Image, query: str) -> dict[str, Any]:
        return self._call("detect", image, query)


def _replica_count(env_name: str) -> int:
    value = int(os.getenv(env_name, "1"))
    if value < 1:
        raise ToolServerError(f"{env_name} must be at least 1, got {value}")
    return value


def _target_image(arguments: dict[str, Any], images: list[Image.Image]) -> int:
    try:
        index = int(arguments.get("target_image", 0))
    except (TypeError, ValueError) as exc:
        raise ToolServerError("target_image must be an integer") from exc
    if not 0 <= index < len(images):
        raise ToolServerError(f"target_image {index} is outside the image list")
    return index


def _relative_box_to_pixels(box: Any, size: tuple[int, int]) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ToolServerError("bbox_2d must contain four relative coordinates")
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ToolServerError("bbox_2d coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ToolServerError("bbox_2d coordinates must be finite")
    x1, y1, x2, y2 = (min(1000.0, max(0.0, value)) for value in (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        raise ToolServerError("bbox_2d must describe a positive-area region")
    width, height = size
    return [x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0]


def _pixel_box_to_relative(box: Any, size: tuple[int, int]) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ToolServerError("tool backend returned an invalid bbox")
    width, height = size
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        round(min(1000.0, max(0.0, x1 * 1000.0 / width)), 4),
        round(min(1000.0, max(0.0, y1 * 1000.0 / height)), 4),
        round(min(1000.0, max(0.0, x2 * 1000.0 / width)), 4),
        round(min(1000.0, max(0.0, y2 * 1000.0 / height)), 4),
    ]


def _ocr_bbox_to_relative(box: Any, size: tuple[int, int]) -> list[float] | None:
    if not isinstance(box, (list, tuple)):
        return None
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        return _pixel_box_to_relative(box, size)
    points = []
    for point in box:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
    if not points:
        return None
    xs, ys = zip(*points)
    return _pixel_box_to_relative([min(xs), min(ys), max(xs), max(ys)], size)


def _remote_error(body: dict[str, Any], tool: str) -> dict[str, Any] | None:
    if body.get("status") == "success" or body.get("success") is True:
        return None
    return {
        "status": "error",
        "tool": tool,
        "code": str(body.get("error_code") or "tool_failed"),
        "message": str(body.get("error_message") or body.get("text") or "tool execution failed"),
        "recoverable": True,
    }


@dataclass
class ToolService:
    sam3: Any | None
    grounding_dino: Any | None
    ocr_bridge: VtsRemoteBridge | None = None
    depth_bridge: VtsRemoteBridge | None = None
    count_bridge: VtsRemoteBridge | None = None
    crop_size: int = 336
    minimum_crop_size: int = 96

    def _execute_ocr(
        self,
        arguments: dict[str, Any],
        images: list[Image.Image],
        instance_id: str,
    ) -> tuple[dict, list[str]]:
        if self.ocr_bridge is None:
            raise ToolServerError("ocr_read service is not configured")
        target = _target_image(arguments, images)
        threshold = 0.0
        max_results = 50
        response = self.ocr_bridge.execute(
            images=images,
            arguments={"image_id": target, "minimum_confidence": threshold},
            instance_id=instance_id,
        )
        if error := _remote_error(response.result, "ocr_read"):
            return error, []
        structured = dict(response.result.get("structured") or {})
        regions = []
        for item in (structured.get("results") or [])[:max_results]:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            region = {"text": str(item["text"])}
            bbox = _ocr_bbox_to_relative(item.get("bbox"), images[target].size)
            if bbox is not None:
                region["bbox_2d"] = bbox
            if item.get("confidence") is not None:
                region["confidence"] = round(float(item["confidence"]), 4)
            regions.append(region)
        returned_images = response.images
        result = {
            "text": str(structured.get("text") or "\n".join(item["text"] for item in regions)),
            "regions": regions,
            "count": len(regions),
            "target_image": target,
            "coordinate_space": "relative_0_1000",
            "source": "ocr",
        }
        if returned_images:
            result["annotated_image"] = len(images)
        return result, returned_images

    def _execute_count(
        self,
        arguments: dict[str, Any],
        images: list[Image.Image],
        instance_id: str,
    ) -> tuple[dict, list[str]]:
        if self.count_bridge is None:
            raise ToolServerError("object_count service is not configured")
        target = _target_image(arguments, images)
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolServerError("object_count.query must be non-empty")
        response = self.count_bridge.execute(
            images=images,
            arguments={"image_id": target, "query": query},
            instance_id=instance_id,
        )
        if error := _remote_error(response.result, "object_count"):
            return error, []
        structured = dict(response.result.get("structured") or {})
        points = []
        for point in structured.get("points") or []:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                width, height = images[target].size
                points.append([
                    round(min(1000.0, max(0.0, float(point[0]) * 1000.0 / width)), 4),
                    round(min(1000.0, max(0.0, float(point[1]) * 1000.0 / height)), 4),
                ])
        boxes = [
            _pixel_box_to_relative(box, images[target].size)
            for box in structured.get("boxes") or []
        ]
        returned_images = response.images
        result = {
            "query": query,
            "count": int(structured.get("count", len(points) or len(boxes))),
            "points_2d": points,
            "boxes": boxes,
            "target_image": target,
            "coordinate_space": "relative_0_1000",
            "source": "object_count",
        }
        if returned_images:
            result["annotated_image"] = len(images)
        return result, returned_images

    def _execute_depth(
        self,
        name: str,
        arguments: dict[str, Any],
        images: list[Image.Image],
        instance_id: str,
    ) -> tuple[dict, list[str]]:
        if self.depth_bridge is None:
            raise ToolServerError(f"{name} service is not configured")
        target = _target_image(arguments, images)
        query = None
        if name == "depth_measure":
            relative_box = arguments.get("bbox_2d")
            pixel_box = _relative_box_to_pixels(relative_box, images[target].size)
            relative_box = _pixel_box_to_relative(pixel_box, images[target].size)
        elif name == "ground_depth":
            query = str(arguments.get("query") or "").strip()
            if not query:
                raise ToolServerError("ground_depth.query must be non-empty")
            if self.grounding_dino is None:
                return {
                    "status": "error",
                    "tool": name,
                    "message": "GroundingDINO is unavailable for query-based depth measurement.",
                    "recoverable": True,
                }, []
            detected = self.grounding_dino.detect(images[target], query)
            boxes = detected.get("boxes") or []
            confidence = detected.get("confidence") or []
            if not boxes:
                return {
                    "status": "error",
                    "tool": name,
                    "message": f"No object was found for {query!r}; retry with a more concrete query.",
                    "recoverable": True,
                }, []
            selected = max(
                range(len(boxes)),
                key=lambda index: float(confidence[index]) if index < len(confidence) else 0.0,
            )
            pixel_box = [float(value) for value in boxes[selected]]
            relative_box = _pixel_box_to_relative(pixel_box, images[target].size)
        else:  # pragma: no cover - only called from the checked dispatch below
            raise ToolServerError(f"Unsupported depth tool: {name}")

        response = self.depth_bridge.execute(
            images=images,
            arguments={"image_id": target, "bboxes": [pixel_box]},
            instance_id=instance_id,
        )
        if error := _remote_error(response.result, name):
            return error, []
        structured = dict(response.result.get("structured") or {})
        statistics = dict(structured.get("statistics") or {})
        metric = bool((response.result.get("provenance") or {}).get("metric_depth") or statistics.get("metric_depth"))
        regions = list(structured.get("region_depths") or [])
        if not metric:
            return {
                "status": "error",
                "tool": name,
                "message": "The configured backend does not provide metric depth.",
                "recoverable": False,
            }, []
        if not regions or regions[0].get("median_depth_m") is None:
            return {
                "status": "error",
                "tool": name,
                "message": "No valid depth pixels were found inside the target region.",
                "recoverable": True,
            }, []
        returned_images = response.images
        result = {
            "depth_m": round(float(regions[0]["median_depth_m"]), 4),
            "bbox_2d": relative_box,
            "target_image": target,
            "coordinate_space": "relative_0_1000",
        }
        if query is not None:
            result["query"] = query
        if returned_images:
            result["depth_image"] = len(images)
        return result, returned_images

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        images: list[Image.Image],
        *,
        instance_id: str = "request",
    ) -> tuple[dict, list[str]]:
        if name not in SUPPORTED_TOOLS:
            raise ToolServerError(f"Unsupported tool: {name}")
        if name == "ocr_read":
            try:
                return self._execute_ocr(arguments, images, instance_id)
            except VtsBridgeError as exc:
                raise ToolServerError(str(exc)) from exc
        if name == "object_count":
            try:
                return self._execute_count(arguments, images, instance_id)
            except VtsBridgeError as exc:
                raise ToolServerError(str(exc)) from exc
        if name in {"depth_measure", "ground_depth"}:
            try:
                return self._execute_depth(name, arguments, images, instance_id)
            except VtsBridgeError as exc:
                raise ToolServerError(str(exc)) from exc
        if name == "grounding_detect":
            max_words = int(os.getenv("GROUNDING_DINO_MAX_QUERY_WORDS", "0"))
            query = arguments.get("query")
            word_count = len(query.split()) if isinstance(query, str) else 0
            if max_words > 0 and word_count > max_words:
                return {
                    "status": "error",
                    "error": (
                        f"grounding_detect.query has {word_count} words; use one concrete "
                        f"object noun phrase of at most {max_words} words, then retry"
                    ),
                    "word_count": word_count,
                    "max_words": max_words,
                }, []
        output_images: list[str] = []

        def collect_crop(crop: Image.Image, filename: str, target_image: int) -> str:
            output_images.append(encode_image(crop))
            return f"tool://images/{target_image}/{filename}"

        try:
            result = execute_tool_call(
                {"name": name, "arguments": arguments},
                images=images,
                sam3_backend=self.sam3,
                grounding_backend=self.grounding_dino,
                uid=instance_id,
                crop_writer=collect_crop,
                min_crop_side=self.minimum_crop_size,
                output_side=self.crop_size,
            )
        except (TypeError, ValueError) as exc:
            raise ToolServerError(str(exc)) from exc
        return result, output_images


def load_service(backend: str) -> ToolService:
    sam3 = None
    grounding_dino = None
    if backend in {"all", "sam3"}:
        sam3 = BackendPool(
            [
                Sam3Backend(
                    os.getenv("SAM3_MODEL_PATH") or None,
                    os.getenv("SAM3_DEVICE", "cuda:0"),
                    float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")),
                )
                for _ in range(_replica_count("SAM3_REPLICAS"))
            ]
        )
    if backend in {"all", "groundingdino"}:
        grounding_dino = BackendPool(
            [
                GroundingDinoBackend(
                    os.getenv("GROUNDING_DINO_MODEL_PATH", "IDEA-Research/grounding-dino-base"),
                    os.getenv("GROUNDING_DINO_DEVICE", "cuda:1" if backend == "all" else "cuda:0"),
                    float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.35")),
                    float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.25")),
                )
                for _ in range(_replica_count("GROUNDING_DINO_REPLICAS"))
            ]
        )
    return ToolService(
        sam3=sam3,
        grounding_dino=grounding_dino,
        ocr_bridge=VtsRemoteBridge.from_env(endpoint_env="VTS_OCR_ENDPOINT", tool_name="ocr_read"),
        depth_bridge=VtsRemoteBridge.from_env(endpoint_env="VTS_DEPTH_ENDPOINT", tool_name="depth_estimate"),
        count_bridge=VtsRemoteBridge.from_env(
            endpoint_env="VTS_COUNT_ENDPOINT", tool_name="countgd_plusplus_count"
        ),
    )


def create_app(service: ToolService):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Visual Agent Tool Server")

    @app.middleware("http")
    async def authenticate(request, call_next):
        api_key = os.getenv("VISUAL_TOOL_API_KEY")
        if api_key and request.headers.get("Authorization") != f"Bearer {api_key}":
            return JSONResponse(status_code=401, content={"detail": "Invalid visual-tool API key"})
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "sam3_loaded": service.sam3 is not None,
            "grounding_dino_loaded": service.grounding_dino is not None,
            "sam3_replicas": getattr(service.sam3, "replica_count", 0),
            "grounding_dino_replicas": getattr(service.grounding_dino, "replica_count", 0),
            "ocr_service": service.ocr_bridge is not None,
            "depth_service": service.depth_bridge is not None,
            "count_service": service.count_bridge is not None,
        }

    @app.post("/execute")
    async def execute(payload: dict[str, Any]):
        started = time.perf_counter()
        try:
            images = [decode_image(value) for value in payload.get("images", [])]
            result, output_images = await asyncio.to_thread(
                service.execute,
                payload.get("name"),
                payload.get("arguments") or {},
                images,
                instance_id=str(payload.get("instance_id") or "request"),
            )
            return {
                "status": "success",
                "result": result,
                "images": output_images,
                "metrics": {"latency_ms": round((time.perf_counter() - started) * 1000, 3)},
            }
        except ToolServerError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Visual tool execution failed: {exc}") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", "sam3", "groundingdino"), default=os.getenv("VISUAL_TOOL_BACKEND", "all"))
    parser.add_argument("--host", default=os.getenv("VISUAL_TOOL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISUAL_TOOL_PORT", "9000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(load_service(args.backend)), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
