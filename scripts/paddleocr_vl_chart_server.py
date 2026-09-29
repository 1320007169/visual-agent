#!/usr/bin/env python3
"""Serve chart element recognition from the cached PaddleOCR-VL weights."""

import argparse
from pathlib import Path
import threading

from PIL import Image


class ChartRecognizer:
    def __init__(self, model_root: Path, device: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_root), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_root), trust_remote_code=True, local_files_only=True,
        )

    def read(self, image: Image.Image) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Chart Recognition:"},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True,
            )
        # Decode only generated tokens, excluding the fixed task prompt.
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        truncated = generated.shape[1] >= self.max_new_tokens and int(generated[0, -1]) not in eos_ids
        return {"text": text, "truncated": truncated}


def create_app(recognizer, allowed_root: Path):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="PaddleOCR-VL Chart Parser")
    lock = threading.Lock()
    allowed_root = allowed_root.resolve()

    @app.get("/health")
    def health():
        return {"status": "ok", "tools": ["chart_parse"], "backend": "paddleocr_vl"}

    @app.post("/execute")
    def execute(payload: dict):
        if payload.get("tool") != "chart_parse":
            raise HTTPException(status_code=422, detail="Only chart_parse is supported")
        try:
            arguments = payload["args"]
            index = arguments["image_id"]
            if type(index) is not int or index < 0 or set(arguments) != {"image_id"}:
                raise ValueError("Expected a nonnegative image_id only")
            path = Path(payload["context"]["images"][index]["path"]).resolve()
            path.relative_to(allowed_root)
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail="Invalid chart image or arguments") from exc
        with lock:
            result = recognizer.read(image)
        if not result["text"].strip():
            return {"status": "failed", "error_code": "empty_chart",
                    "error_message": "Chart parser returned no content", "images": []}
        return {"status": "success", "structured": result, "images": []}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=9007)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not (args.model_root / "model.safetensors").is_file():
        parser.error("Local PaddleOCR-VL model.safetensors is missing")
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be between 1 and 4096")
    import uvicorn

    recognizer = ChartRecognizer(args.model_root, args.device, args.max_new_tokens)
    uvicorn.run(create_app(recognizer, args.allowed_root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
