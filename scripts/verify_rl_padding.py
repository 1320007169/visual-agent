#!/usr/bin/env python3
"""Numerical comparison of the production actor forward, optionally on Qwen3-VL."""

import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_actor_forward():
    tree = ast.parse((ROOT / "reinforcement_learning/verl/workers/actor/dp_actor.py").read_text(encoding="utf-8"))
    crop = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "crop_common_padding")
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    forward = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch")
    forward.decorator_list = []
    cls = ast.ClassDef(name="ActorForward", bases=[], keywords=[], body=[forward], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[crop, cls], type_ignores=[]))
    namespace = {
        "torch": torch, "Tuple": Tuple,
        "logprobs_from_logits": lambda logits, labels: logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        "verl_F": SimpleNamespace(entropy_from_logits=lambda logits: -(logits.float().softmax(-1) * logits.float().log_softmax(-1)).sum(-1)),
    }
    exec(compile(module, str(ROOT / "reinforcement_learning/verl/workers/actor/dp_actor.py"), "exec"), namespace)
    return namespace["ActorForward"], namespace["crop_common_padding"]


def load_loss_functions():
    namespace = {"torch": torch}
    tree = ast.parse((ROOT / "reinforcement_learning/verl/utils/torch_functional.py").read_text(encoding="utf-8"))
    mean = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "masked_mean")
    exec(compile(ast.Module(body=[mean], type_ignores=[]), "masked_mean", "exec"), namespace)
    namespace["verl_F"] = SimpleNamespace(masked_mean=namespace["masked_mean"])
    tree = ast.parse((ROOT / "reinforcement_learning/verl/trainer/ppo/core_algos.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"agg_loss", "compute_policy_loss", "kl_penalty"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "core_algos", "exec"), namespace)
    return namespace


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(512, 32)
        self.projection = torch.nn.Linear(32, 512)

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = hidden.cumsum(1) / attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)
        return SimpleNamespace(logits=self.projection(hidden))


def qwen_model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={"vocab_size": 512, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "patch_size": 2, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "out_hidden_size": 32,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": [0]},
        image_token_id=500, video_token_id=499, vision_start_token_id=501, vision_end_token_id=502,
    )
    config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config)


def make_batch(model, crops=0, original_image=False, long_text=False, empty=False):
    prompt = [11, 12, 13]
    image = [501, 500, 500, 500, 500, 502]
    if original_image:
        prompt += image
    response = [14, 15] + image * crops + ([16] * 40 if long_text else []) + [17]
    image_count = int(original_image) + crops
    mm = {"pixel_values": torch.randn(image_count * 16, 24),
          "image_grid_thw": torch.tensor([[1, 4, 4]] * image_count)} if image_count else {}
    if empty:
        response = []
    ids = torch.tensor([[0] * 7 + prompt + response + [0] * 13])
    mask = torch.tensor([[0] * 7 + [1] * (len(prompt) + len(response)) + [0] * 13])
    if hasattr(model, "model"):
        positions, _ = model.model.get_rope_index(ids, image_grid_thw=mm.get("image_grid_thw"), attention_mask=mask)
        positions = positions.transpose(0, 1)
    else:
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
            "responses": ids[:, len(prompt) + 7:], "multi_modal_inputs": [mm]}


def compare(model, batch, atol=0.002, rtol=0.02):
    cls, crop = load_actor_forward()
    functions = load_loss_functions()
    outputs, gradients, updated = [], [], []
    response_mask = batch["attention_mask"][:, -batch["responses"].shape[-1]:].float()
    for enabled in (False, True):
        module = copy.deepcopy(model).to(batch["input_ids"].device).train()
        actor = cls()
        actor.config = {"crop_common_padding": enabled}
        actor.actor_module = module
        actor.device_name = batch["input_ids"].device.type
        actor.use_remove_padding = False
        actor.use_fused_kernels = False
        entropy, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=True)
        if not outputs:
            old_log_probs = log_probs.detach() + 0.02
        advantages = torch.linspace(-1, 1, log_probs.shape[-1], device=log_probs.device).unsqueeze(0)
        pg_loss, *_ = functions["compute_policy_loss"](
            old_log_probs, log_probs, advantages, response_mask,
            cliprange=0.2, clip_ratio_c=3.0, loss_agg_mode="token-mean",
        )
        entropy_loss = functions["agg_loss"](entropy, response_mask, "token-mean")
        kl = functions["kl_penalty"](log_probs, old_log_probs - 0.03, "low_var_kl")
        kl_loss = functions["agg_loss"](kl, response_mask, "token-mean")
        loss = pg_loss - 0.01 * entropy_loss + 0.001 * kl_loss
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        loss.backward()
        grads = [parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter) for parameter in module.parameters()]
        if response_mask.any():
            optimizer.step()
        outputs.append((log_probs.detach() * response_mask, entropy.detach() * response_mask, loss.detach()))
        gradients.append(grads)
        updated.append([parameter.detach().clone() for parameter in module.parameters()])
    for first, second in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
    for first, second in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
    for first, second in zip(updated[0], updated[1]):
        torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
    cropped, _ = crop(batch)
    return {"original_width": batch["input_ids"].shape[-1], "cropped_width": cropped["input_ids"].shape[-1],
            "log_prob_max_difference": float((outputs[0][0] - outputs[1][0]).abs().max()),
            "entropy_max_difference": float((outputs[0][1] - outputs[1][1]).abs().max()),
            "loss_difference": float((outputs[0][2] - outputs[1][2]).abs()),
            "gradient_max_difference": max(float((a - b).abs().max()) for a, b in zip(gradients[0], gradients[1])),
            "update_max_difference": max(float((a - b).abs().max()) for a, b in zip(updated[0], updated[1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3-vl", action="store_true", help="Use a small randomly initialized real Qwen3-VL")
    parser.add_argument("--device", default="cpu", help="Comparison device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Bound the tiny verification model's allocator on shared GPUs.
        torch.cuda.set_per_process_memory_fraction(0.01, device)
    model = qwen_model() if args.qwen3_vl else TinyCausalModel()
    for name, settings in [("text", {}), ("original_image", {"original_image": True}),
                           ("multiple_crops", {"original_image": True, "crops": 2}),
                           ("long_tool_text", {"long_text": True}), ("empty_loss", {"empty": True})]:
        batch = make_batch(model, **settings)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        batch["multi_modal_inputs"] = [{key: value.to(device) for key, value in inputs.items()}
                                       for inputs in batch["multi_modal_inputs"]]
        result = compare(model, batch)
        print(json.dumps({"case": name, "model": type(model).__name__, "device": str(device), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
