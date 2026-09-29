#!/usr/bin/env python3
"""GPU checks of the production actor forward on a pretrained Qwen3-VL checkpoint.

crop: compare padding off/on, with repeat and manually compacted input controls.
rmpad: compare padded/packed micro batches and change neighbours at fixed shape.
Gradients use assistant-only NLL, not PPO/KL/entropy loss or an optimizer update.
Numerical gate failures alone do not identify an implementation error: BF16
outputs can change with tensor shape even when both runs have crop disabled.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reinforcement_learning"))

TOOL_CALL = '<tool_call>\n{"name": "crop_zoom", "arguments": {"bbox_2d": [100, 100, 600, 600], "target_image": 0}}\n</tool_call>'
CASES = ("text", "image", "crop", "long_tool_text")


def build_sample(processor, case, seed):
    """Return prompt length, full ids, multimodal inputs and production response loss mask."""
    from verl.workers.rollout.chat_scheduler import ToolCompletionCallback

    rng = np.random.RandomState(seed)
    image = Image.fromarray(rng.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    question = {"type": "text", "text": f"Question {seed}: which option matches the picture?"}
    prompt = [{"role": "user", "content": [question] if case == "text" else [{"type": "image"}, question]}]
    prompt_images = [] if case == "text" else [image]
    answer = {"role": "assistant", "content": "<answer>A</answer>"}
    if case == "crop":
        observation = [{"type": "text", "text": '<tool_response>\n{"target_image": 1}\n</tool_response>'}, {"type": "image"}]
        turns = [{"role": "assistant", "content": TOOL_CALL}, {"role": "user", "content": observation}, answer]
        images = prompt_images + [image.crop((64, 64, 288, 288)).resize((320, 320))]
    elif case == "long_tool_text":
        observation = "<tool_response>\n" + " ".join(f"word{seed}_{index}" for index in range(300)) + "\n</tool_response>"
        turns = [{"role": "assistant", "content": TOOL_CALL}, {"role": "user", "content": observation}, answer]
        images = prompt_images
    else:
        turns = [answer]
        images = prompt_images
    prompt_text = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    full_text = processor.apply_chat_template(prompt + turns, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template output does not start with the generation prompt")
    prompt_ids = processor(text=[prompt_text], images=prompt_images or None, return_tensors="pt")["input_ids"][0]
    encoded = processor(text=[full_text], images=images or None, return_tensors="pt")
    multi_modal = {key: encoded[key] for key in ("pixel_values", "image_grid_thw") if key in encoded}
    response_ids = encoded["input_ids"][:, prompt_ids.numel():]
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tokenizer = processor.tokenizer
    loss_mask = callback._mask_out_tools_calling_tokens(
        [prompt], [prompt + turns], response_ids, torch.ones_like(response_ids),
    )[0]
    return prompt_ids.numel(), encoded["input_ids"][0], multi_modal, loss_mask


def collate(model, pad_token_id, samples, device, prompt_pad=5, response_pad=7):
    """Left-pad prompts and right-pad responses to shared widths, as the rollout batch does."""
    prompt_width = max(sample[0] for sample in samples) + prompt_pad
    response_width = max(ids.numel() - prompt_length for prompt_length, ids, _, _ in samples) + response_pad
    rows, masks, positions, loss_masks = [], [], [], []
    for prompt_length, ids, multi_modal, response_loss_mask in samples:
        left = prompt_width - prompt_length
        right = response_width - (ids.numel() - prompt_length)
        row = torch.cat([torch.full((left,), pad_token_id), ids, torch.full((right,), pad_token_id)])
        mask = torch.cat([torch.zeros(left), torch.ones(ids.numel()), torch.zeros(right)]).long()
        grid = multi_modal.get("image_grid_thw")
        kwargs = {"image_grid_thw": grid.to(device) if grid is not None else None, "attention_mask": mask[None].to(device)}
        # Newer Transformers releases also require per-token modality ids.
        if "mm_token_type_ids" in inspect.signature(model.model.get_rope_index).parameters:
            kwargs["mm_token_type_ids"] = (row[None] == model.config.image_token_id).int().to(device)
        position, _ = model.model.get_rope_index(row[None].to(device), **kwargs)
        rows.append(row)
        masks.append(mask)
        positions.append(position[:, 0].cpu())
        loss_masks.append(torch.cat([torch.zeros(prompt_width), response_loss_mask, torch.zeros(right)]).long())
    input_ids = torch.stack(rows).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.stack(masks).to(device),
        "loss_mask": torch.stack(loss_masks).to(device),
        "position_ids": torch.stack(positions).to(device),  # (bsz, 3, seqlen)
        "responses": input_ids[:, prompt_width:],
        "multi_modal_inputs": np.array(
            [{key: value.to(device) for key, value in multi_modal.items()} for _, _, multi_modal, _ in samples],
            dtype=object,
        ),
    }


def make_actor(model, use_remove_padding, crop_common_padding):
    from omegaconf import OmegaConf

    import verl.utils.torch_functional as verl_F
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    # Skip __init__, which needs torch.distributed; set only what the forward reads.
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.config = OmegaConf.create({"crop_common_padding": crop_common_padding, "entropy_checkpointing": False})
    actor.use_remove_padding = use_remove_padding
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.ulysses_sequence_parallel_size = 1
    actor.device_name = "cuda"
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    return actor


def masked_nll(log_probs, loss_mask):
    """Keep a backward graph with zero gradients when no loss tokens remain."""
    return -(log_probs * loss_mask).sum() / loss_mask.sum().clamp_min(1)


def run(actor, batch, with_grad):
    """Compare all attended tokens; differentiate NLL only on production loss tokens."""
    model = actor.actor_module
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(with_grad):
        entropy, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=True)
    mask = batch["attention_mask"][:, -batch["responses"].shape[-1]:].bool()
    gradients = None
    if with_grad:
        loss_mask = batch["loss_mask"][:, -batch["responses"].shape[-1]:]
        masked_nll(log_probs, loss_mask).backward()
        gradients = [parameter.grad.detach().cpu() if parameter.grad is not None else None for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)
    return log_probs.detach().float()[mask].cpu(), entropy.detach().float()[mask].cpu(), gradients


def change_neighbour(batch, old_token, new_token, row=1):
    """Change one row's content, preserving other rows, shapes, grids, masks and positions."""
    batch["input_ids"][row].masked_fill_(batch["input_ids"][row].eq(old_token), new_token)
    # responses is a view in collate(), but also support independent response storage.
    batch["responses"][row].masked_fill_(batch["responses"][row].eq(old_token), new_token)
    if "pixel_values" in batch["multi_modal_inputs"][row]:
        # Assign a new tensor: collating the same sample twice can alias image storage.
        batch["multi_modal_inputs"][row]["pixel_values"] = -batch["multi_modal_inputs"][row]["pixel_values"]


def difference(first, second):
    result = {
        "log_prob_max_abs": float((first[0] - second[0]).abs().max()),
        "log_prob_mean_abs": float((first[0] - second[0]).abs().mean()),
        "entropy_max_abs": float((first[1] - second[1]).abs().max()),
        "entropy_mean_abs": float((first[1] - second[1]).abs().mean()),
    }
    if first[2] is not None:
        numerator = 0.0
        for a, b in zip(first[2], second[2]):
            if (a is None) != (b is None):
                numerator = float("inf")  # a parameter received a gradient in only one run
            elif a is not None:
                numerator += float((a.float() - b.float()).norm() ** 2)
        denominator = sum(float(a.float().norm() ** 2) for a in first[2] if a is not None)
        result["gradient_relative_difference"] = (numerator / max(denominator, 1e-30)) ** 0.5
    return result


# Every reported difference and the argument holding its limit.
LIMITS = {
    "log_prob_max_abs": "max_abs",
    "log_prob_mean_abs": "mean_abs",
    "entropy_max_abs": "entropy_max_abs",
    "entropy_mean_abs": "entropy_mean_abs",
    "gradient_relative_difference": "grad_rel",
}


def check(name, result, args):
    unchecked = set(result) - set(LIMITS)
    if unchecked:
        raise ValueError(f"differences without a pass limit: {sorted(unchecked)}")
    # `not value <= limit` also fails NaN differences.
    failed = sorted(key for key, value in result.items() if not value <= getattr(args, LIMITS[key]))
    print(json.dumps({"check": name, "passed": not failed, "failed": failed, **result}, sort_keys=True), flush=True)
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Pretrained Qwen3-VL checkpoint with its processor")
    parser.add_argument("--mode", choices=("crop", "rmpad"), required=True)
    parser.add_argument("--attn", default="flash_attention_2", help="Attention implementation; rmpad requires flash_attention_2")
    parser.add_argument("--no-grad", action="store_true", help="Compare forward outputs only")
    parser.add_argument("--max-abs", type=float, default=0.05, help="Maximum valid-token log-prob difference")
    parser.add_argument("--mean-abs", type=float, default=0.005, help="Mean valid-token log-prob difference")
    parser.add_argument("--entropy-max-abs", type=float, default=0.05, help="Maximum valid-token entropy difference")
    parser.add_argument("--entropy-mean-abs", type=float, default=0.005, help="Mean valid-token entropy difference")
    parser.add_argument("--grad-rel", type=float, default=0.02, help="Relative L2 gradient difference")
    args = parser.parse_args()
    if args.mode == "rmpad" and args.attn != "flash_attention_2":
        parser.error("rmpad requires --attn flash_attention_2")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    ).to(device).eval()
    apply_monkey_patch(model=model, use_remove_padding=args.mode == "rmpad", ulysses_sp_size=1)
    pad = processor.tokenizer.pad_token_id
    with_grad = not args.no_grad
    samples = {case: build_sample(processor, case, seed) for seed, case in enumerate(CASES, start=1)}
    results = []

    if args.mode == "crop":
        baseline, cropped = make_actor(model, False, False), make_actor(model, False, True)
        for case in CASES:
            batch = collate(model, pad, [samples[case]], device)
            compact = collate(model, pad, [samples[case]], device, prompt_pad=0, response_pad=0)
            reference = run(baseline, batch, with_grad)
            results.append(check(f"repeat/{case}", difference(reference, run(baseline, batch, with_grad)), args))
            manual = run(baseline, compact, with_grad)
            # A diagnostic control, excluded from the implementation control gates.
            check(f"padding_only/{case}", difference(reference, manual), args)
            candidate = run(cropped, batch, with_grad)
            results.append(check(f"crop_manual_compact/{case}", difference(manual, candidate), args))
            results.append(check(f"crop/{case}", difference(reference, candidate), args))
    else:
        padded, packed = make_actor(model, False, False), make_actor(model, True, False)
        for case in CASES:
            batch = collate(model, pad, [samples[case]], device)
            compact = collate(model, pad, [samples[case]], device, prompt_pad=0, response_pad=0)
            reference = run(padded, batch, with_grad)
            results.append(check(f"repeat/{case}", difference(reference, run(padded, batch, with_grad)), args))
            manual = run(padded, compact, with_grad)
            check(f"padding_only/{case}", difference(reference, manual), args)
            candidate = run(packed, batch, with_grad)
            results.append(check(f"rmpad_manual_compact/{case}", difference(manual, candidate), args))
            results.append(check(f"rmpad/{case}", difference(reference, candidate), args))

        # Test both directions: missing causal boundaries can leak A into B but not B into A.
        sample_a = samples["crop"]
        old_token, = processor.tokenizer.encode("A", add_special_tokens=False)
        new_token, = processor.tokenizer.encode("B", add_special_tokens=False)
        for name, actor in (("padded", padded), ("packed", packed)):
            for neighbour in ("long_tool_text", "text", "image", "crop"):
                for changed_row in (0, 1):
                    batch = collate(model, pad, [sample_a, samples[neighbour]], device)
                    reference = run(actor, batch, False)
                    change_neighbour(batch, old_token, new_token, row=changed_row)
                    changed = run(actor, batch, False)
                    counts = batch["attention_mask"][:, -batch["responses"].shape[-1]:].sum(-1)
                    unchanged_row = 1 - changed_row
                    start = int(counts[:unchanged_row].sum())
                    end = start + int(counts[unchanged_row])
                    first = (reference[0][start:end], reference[1][start:end], None)
                    second = (changed[0][start:end], changed[1][start:end], None)
                    results.append(check(f"isolation_fixed_shape/{name}/{neighbour}/changed_{changed_row}",
                                         difference(first, second), args))

    print(json.dumps({"mode": args.mode, "all_passed": all(results),
                      "gradient_objective": "assistant_only_nll" if with_grad else None,
                      "optimizer_update_tested": False,
                      "failure_interpretation": "Numerical gate failure alone does not establish an implementation defect."}), flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
