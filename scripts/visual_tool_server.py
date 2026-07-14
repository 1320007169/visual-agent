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
import threading
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image


SUPPORTED_TOOLS = {
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


class Sam3Backend:
    """Official SAM3 image processor adapter with optional custom factory."""

    def __init__(self, model_path: str | None, device: str, threshold: float):
        import torch

        self.torch = torch
        self.device = device
        self.threshold = threshold
        factory_path = os.getenv("SAM3_FACTORY")
        if factory_path:
            built = _load_object(factory_path)(model_path=model_path, device=device)
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
            if model_path:
                for name in ("checkpoint_path", "ckpt_path", "model_path"):
                    if name in signature.parameters:
                        kwargs[name] = model_path
                        break
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
        with self.lock, self.torch.inference_mode():
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
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
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


@dataclass
class ToolService:
    sam3: Any | None
    grounding_dino: Any | None
    crop_size: int = 336
    minimum_crop_size: int = 96

    def execute(self, name: str, arguments: dict[str, Any], images: list[Image.Image]) -> tuple[dict, list[str]]:
        if name not in SUPPORTED_TOOLS:
            raise ToolServerError(f"Unsupported tool: {name}")
        target_image = arguments.get("target_image", 0)
        if not isinstance(target_image, int) or target_image < 0 or target_image >= len(images):
            raise ToolServerError(f"target_image {target_image!r} is outside images[0:{len(images)}]")
        image = images[target_image]

        if name == "grounding_detect":
            if self.grounding_dino is None:
                raise ToolServerError("GroundingDINO backend is not loaded")
            query = self._required_query(arguments.get("query"))
            detected = self.grounding_dino.detect(image, query)
            result = {"query": query, **detected, "count": len(detected["boxes"]), "source": "groundingdino"}
            return result, []

        if self.sam3 is None:
            raise ToolServerError("SAM3 backend is not loaded")
        if name == "sam3_segment_multi":
            queries = self._required_queries(arguments.get("queries"))
            results = []
            for item in queries:
                segmented = self.sam3.segment(image, item["query"])
                results.append({**item, **segmented, "count": len(segmented["boxes"])})
            return {"queries": results}, []

        if name == "sam3_crop_zoom":
            query = self._required_query(arguments.get("query"))
            result, crop = self._crop_one(image, query, target_image, arguments.get("slack_ratio", 0.35), len(images))
            return result, [encode_image(crop)]

        queries = self._required_queries(arguments.get("queries"))
        results, crops = [], []
        for offset, item in enumerate(queries):
            result, crop = self._crop_one(
                image, item["query"], target_image, arguments.get("slack_ratio", 0.35), len(images) + offset
            )
            result["role"] = item.get("role", "target")
            results.append(result)
            crops.append(encode_image(crop))
        return {"queries": results, "source": "sam3_crop_zoom_multi"}, crops

    @staticmethod
    def _required_query(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ToolServerError("query must be a non-empty string")
        return value.strip()

    def _required_queries(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list) or not value:
            raise ToolServerError("queries must be a non-empty list")
        output = []
        for item in value:
            if not isinstance(item, dict):
                raise ToolServerError("Each queries item must be an object")
            output.append({"role": str(item.get("role", "target")), "query": self._required_query(item.get("query"))})
        return output

    def _crop_one(
        self, image: Image.Image, query: str, source_index: int, slack_ratio: Any, output_index: int
    ) -> tuple[dict[str, Any], Image.Image]:
        segmented = self.sam3.segment(image, query)
        if not segmented["boxes"]:
            raise ToolServerError(f"SAM3 found no region for query {query!r}")
        scores = segmented["confidence"] or [0.0] * len(segmented["boxes"])
        selected = max(range(len(segmented["boxes"])), key=lambda i: scores[i])
        box = segmented["boxes"][selected]
        crop_box = self._expanded_square(box, image.size, float(slack_ratio))
        crop = image.crop(tuple(crop_box)).resize((self.crop_size, self.crop_size), Image.Resampling.LANCZOS)
        target = {
            "role": "target",
            "query": query,
            **segmented,
            "count": len(segmented["boxes"]),
            "selected_box": box,
            "selected_confidence": scores[selected],
        }
        if segmented["mask_area_px"]:
            target["selected_mask_area_px"] = segmented["mask_area_px"][selected]
        image_output = {
            "target_image": output_index,
            "width": self.crop_size,
            "height": self.crop_size,
            "source": "crop_zoom",
        }
        result = {
            "query": query,
            "target_image": source_index,
            "target": target,
            "crop_zoom": {
                "target_image": output_index,
                "requested_bbox_2d": box,
                "bbox_2d": crop_box,
                "slack_ratio": float(slack_ratio),
                "image_outputs": [image_output],
                "text_summary": "Localized the target and cropped the region for closer visual inspection.",
            },
            "image_outputs": [image_output],
            "source": "sam3_crop_zoom",
        }
        return result, crop

    def _expanded_square(self, box: list[float], image_size: tuple[int, int], slack_ratio: float) -> list[float]:
        width, height = image_size
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * max(slack_ratio, 0.0))
        side = min(max(side, self.minimum_crop_size), width, height)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = min(max(cx - side / 2.0, 0.0), width - side)
        top = min(max(cy - side / 2.0, 0.0), height - side)
        return [round(left, 4), round(top, 4), round(left + side, 4), round(top + side, 4)]


def load_service(backend: str) -> ToolService:
    sam3 = None
    grounding_dino = None
    if backend in {"all", "sam3"}:
        sam3 = Sam3Backend(
            os.getenv("SAM3_MODEL_PATH") or None,
            os.getenv("SAM3_DEVICE", "cuda:0"),
            float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")),
        )
    if backend in {"all", "groundingdino"}:
        grounding_dino = GroundingDinoBackend(
            os.getenv("GROUNDING_DINO_MODEL_PATH", "IDEA-Research/grounding-dino-base"),
            os.getenv("GROUNDING_DINO_DEVICE", "cuda:1" if backend == "all" else "cuda:0"),
            float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.35")),
            float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.25")),
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
