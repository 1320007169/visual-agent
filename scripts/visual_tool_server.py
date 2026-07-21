#!/usr/bin/env python3
"""Serve SAM3 and GroundingDINO behind the visual-agent /execute contract."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import io
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


SUPPORTED_TOOLS = {
    "crop_zoom",
    "sam3_segment_multi",
    "grounding_detect",
    "sam3_crop_zoom",
    "sam3_crop_zoom_multi",
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
        prompt = query.strip().rstrip(".") + "."
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
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
        self._available: queue.Queue[Any] = queue.Queue()
        for backend in backends:
            self._available.put(backend)

    @property
    def replica_count(self) -> int:
        return len(self.backends)

    def _call(self, method: str, *args, **kwargs):
        backend = self._available.get()
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


@dataclass
class ToolService:
    sam3: Any | None
    grounding_dino: Any | None
    crop_size: int = 336
    minimum_crop_size: int = 96

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
    return ToolService(sam3=sam3, grounding_dino=grounding_dino)


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
