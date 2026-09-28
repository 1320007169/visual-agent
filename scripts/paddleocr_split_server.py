#!/usr/bin/env python3
"""Serve PP-OCRv5 text detection and region recognition independently."""

import argparse
import math
from pathlib import Path
import threading

import cv2
from fastapi import FastAPI, HTTPException
from paddlex import create_model
import uvicorn


def create_app(model_root: Path) -> FastAPI:
    detector = create_model(
        model_name="PP-OCRv5_server_det",
        model_dir=str(model_root / "PP-OCRv5_server_det"),
        device="gpu:0",
    )
    recognizer = create_model(
        model_name="PP-OCRv5_server_rec",
        model_dir=str(model_root / "PP-OCRv5_server_rec"),
        device="gpu:0",
    )
    lock = threading.Lock()
    app = FastAPI(title="PP-OCRv5 Split Tools")

    @app.get("/health")
    def health():
        return {"status": "ok", "tools": ["text_detect", "text_recognize"]}

    @app.post("/execute")
    def execute(payload: dict):
        tool = payload["tool"]
        arguments = payload["args"]
        image_path = payload["context"]["images"][arguments["image_id"]]["path"]
        with lock:
            if tool == "text_detect":
                prediction = next(iter(detector.predict(image_path)))
                regions = []
                for polygon, score in zip(prediction["dt_polys"], prediction["dt_scores"], strict=True):
                    xs, ys = zip(*polygon)
                    regions.append({
                        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                        "confidence": float(score),
                    })
            elif tool == "text_recognize":
                image = cv2.imread(image_path)
                if image is None:
                    raise HTTPException(status_code=422, detail=f"Cannot read image: {image_path}")
                crops = []
                for box in arguments["bboxes"]:
                    x1, y1 = (math.floor(value) for value in box[:2])
                    x2, y2 = (math.ceil(value) for value in box[2:])
                    crops.append(image[y1:y2, x1:x2])
                predictions = recognizer.predict(input=crops, batch_size=16)
                regions = [
                    {"bbox": box, "text": prediction["rec_text"], "confidence": float(prediction["rec_score"])}
                    for box, prediction in zip(arguments["bboxes"], predictions, strict=True)
                ]
            else:
                raise HTTPException(status_code=422, detail=f"Unsupported OCR tool: {tool}")
        return {"status": "success", "structured": {"results": regions}, "images": []}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model_root), host="127.0.0.1", port=args.port)
