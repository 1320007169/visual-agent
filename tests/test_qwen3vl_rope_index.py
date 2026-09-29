"""The trainer rebuilds Qwen3-VL mRoPE positions with verl's Qwen2-VL get_rope_index.

This test checks those positions against the Transformers Qwen3-VL implementation on real
processor outputs: original image, returned crops, tool text and left padding.
"""

import ast
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "reinforcement_learning/verl"
REQUIRED = ("torch", "transformers", "PIL")


def hf_rope_index(model, input_ids, image_grid_thw, attention_mask, image_token_id):
    """Call the model's get_rope_index across Transformers versions."""
    kwargs = {"image_grid_thw": image_grid_thw, "attention_mask": attention_mask}
    if "mm_token_type_ids" in inspect.signature(model.model.get_rope_index).parameters:
        kwargs["mm_token_type_ids"] = (input_ids == image_token_id).int()
    positions, _ = model.model.get_rope_index(input_ids, **kwargs)
    return positions


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in REQUIRED), "requires torch, transformers and Pillow")
class Qwen3VLRopeIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        model = Path(os.environ.get("QWEN3_VL_PROCESSOR_PATH", ROOT.parent / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        if not (model / "config.json").is_file() or not (model / "tokenizer.json").is_file():
            raise unittest.SkipTest("Set QWEN3_VL_PROCESSOR_PATH to a Qwen3-VL directory with config and processor files")
        cls.torch = torch
        cls.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        # Positions depend only on config and token ids, so no weights are materialized.
        with torch.device("meta"):
            cls.model = Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(str(model), local_files_only=True))
        tree = ast.parse((VERL / "models/transformers/qwen2_vl.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_rope_index")
        namespace = {"torch": torch, "Optional": Optional}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "qwen2_vl", "exec"), namespace)
        cls.verl_rope_index = staticmethod(namespace["get_rope_index"])

    def encode(self, messages, images, left_padding):
        torch = self.torch
        text = self.processor.apply_chat_template(messages, tokenize=False)
        encoded = self.processor(text=[text], images=images or None, return_tensors="pt")
        ids = encoded["input_ids"][0]
        pad = self.processor.tokenizer.pad_token_id
        ids = torch.cat([torch.full((left_padding,), pad), ids, torch.full((3,), pad)])
        mask = torch.cat([torch.zeros(left_padding), torch.ones(ids.numel() - left_padding - 3), torch.zeros(3)]).long()
        return ids, mask, encoded.get("image_grid_thw")

    def assert_same_positions(self, messages, images, left_padding=0):
        ids, mask, grid = self.encode(messages, images, left_padding)
        verl_positions = self.verl_rope_index(self.processor, ids, image_grid_thw=grid, attention_mask=mask)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        hf_positions = hf_rope_index(self.model, ids[None], grid, mask[None], image_token_id)[:, 0]
        valid = mask.bool()
        self.assertTrue(self.torch.equal(verl_positions[:, valid].cpu(), hf_positions[:, valid].cpu()))

    def images(self, *sizes):
        from PIL import Image

        return [Image.new("RGB", size, color) for size, color in zip(sizes, ("red", "green", "blue", "white"))]

    def test_text_only(self):
        self.assert_same_positions([{"role": "user", "content": "q"}, {"role": "assistant", "content": "<answer>A</answer>"}], [], 4)

    def test_original_image_with_crops_and_tool_text(self):
        images = self.images((448, 448), (320, 192), (160, 416))
        call = '<tool_call>\n{"name": "crop_zoom", "arguments": {"bbox_2d": [1, 2, 3, 4], "target_image": 0}}\n</tool_call>'
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
            {"role": "assistant", "content": call},
            {"role": "user", "content": [{"type": "text", "text": "<tool_response>\n{\"target_image\": 1}\n</tool_response>"}, {"type": "image"}]},
            {"role": "assistant", "content": call},
            {"role": "user", "content": [{"type": "text", "text": "<tool_response>\n" + "word " * 50 + "\n</tool_response>"}, {"type": "image"}]},
            {"role": "assistant", "content": "<answer>A</answer>"},
        ]
        self.assert_same_positions(messages, images, 7)


if __name__ == "__main__":
    unittest.main()
