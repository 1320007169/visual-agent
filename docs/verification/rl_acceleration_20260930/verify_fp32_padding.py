#!/usr/bin/env python3
"""Direct HF FP32/SDPA padding control; deliberately bypass actor BF16 autocast.

This changes precision AND attention backend relative to the actor experiment.
It tests padding sensitivity, not PPO updates or packed FlashAttention semantics.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("verification", ROOT / "scripts/verify_qwen3vl_actor_forward.py")
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


def direct(model, batch):
    mm = {key: torch.cat([item[key] for item in batch["multi_modal_inputs"] if key in item], 0)
          for key in {key for item in batch["multi_modal_inputs"] for key in item}}
    width = batch["responses"].shape[-1]
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"].transpose(0, 1), **mm, use_cache=False).logits
        log_probs = logits[:, -width - 1:-1].float().log_softmax(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        scores = log_probs.gather(-1, batch["responses"][..., None]).squeeze(-1)
    mask = batch["attention_mask"][:, -width:].bool()
    return scores[mask].cpu(), entropy[mask].cpu(), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32, attn_implementation="sdpa",
    ).to("cuda").eval()
    for seed, case in enumerate(verification.CASES, 1):
        sample = verification.build_sample(processor, case, seed)
        padded = verification.collate(model, processor.tokenizer.pad_token_id, [sample], "cuda")
        compact = verification.collate(model, processor.tokenizer.pad_token_id, [sample], "cuda", 0, 0)
        result = verification.difference(direct(model, padded), direct(model, compact))
        print(json.dumps({"check": "fp32_hf_padding/" + case, **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
