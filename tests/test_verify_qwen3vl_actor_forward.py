import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_check():
    """Load check() without importing torch, which the GPU script needs at module level."""
    tree = ast.parse((ROOT / "scripts/verify_qwen3vl_actor_forward.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name == "check")
             or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "LIMITS" for target in node.targets))]
    namespace = {"json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "verify_qwen3vl_actor_forward", "exec"), namespace)
    return namespace["check"]


ARGS = SimpleNamespace(max_abs=0.05, mean_abs=0.005, entropy_max_abs=0.05, entropy_mean_abs=0.005, grad_rel=0.02)
PASSING = {"log_prob_max_abs": 0.0, "log_prob_mean_abs": 0.0, "entropy_max_abs": 0.0,
           "entropy_mean_abs": 0.0, "gradient_relative_difference": 0.0}


class VerificationCheckTest(unittest.TestCase):
    def setUp(self):
        self.check = load_check()

    def test_all_differences_within_limits_pass(self):
        self.assertTrue(self.check("case", dict(PASSING), ARGS))

    def test_each_reported_difference_can_fail(self):
        for key in PASSING:
            with self.subTest(key=key):
                self.assertFalse(self.check("case", {**PASSING, key: 100.0}, ARGS))

    def test_nan_fails(self):
        self.assertFalse(self.check("case", {**PASSING, "entropy_max_abs": float("nan")}, ARGS))

    def test_forward_only_results_do_not_need_gradients(self):
        result = {key: value for key, value in PASSING.items() if key != "gradient_relative_difference"}
        self.assertTrue(self.check("case", result, ARGS))

    def test_unlisted_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check("case", {**PASSING, "new_difference": 0.0}, ARGS)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class VerificationInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        spec = importlib.util.spec_from_file_location("verify_actor", ROOT / "scripts/verify_qwen3vl_actor_forward.py")
        cls.verification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verification)

    def test_nll_ignores_observation_and_padding_and_handles_empty_loss(self):
        torch = self.torch
        log_probs = torch.tensor([[-2., -100., -4., -200.]], requires_grad=True)
        loss = self.verification.masked_nll(log_probs, torch.tensor([[1, 0, 1, 0]]))
        self.assertEqual(loss.item(), 3.)
        loss.backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([[-.5, 0., -.5, 0.]]))
        empty = log_probs.detach().clone().requires_grad_()
        loss = self.verification.masked_nll(empty, torch.zeros_like(empty))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(empty.grad, torch.zeros_like(empty))

    def test_neighbour_mutation_preserves_a_even_with_shared_image_storage(self):
        import copy

        torch = self.torch
        pixels = torch.ones(2, 3)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "responses": torch.tensor([[2, 3], [2, 3]]),
            "attention_mask": torch.ones(2, 3),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            "position_ids": torch.arange(3).expand(2, 3, 3),
            "multi_modal_inputs": [{"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 2, 2]])}
                                   for _ in range(2)],
        }
        before = copy.deepcopy(batch)
        self.verification.change_neighbour(batch, 2, 4)
        for key in ("input_ids", "responses", "attention_mask", "loss_mask", "position_ids"):
            self.assertEqual(batch[key].shape, before[key].shape)
            torch.testing.assert_close(batch[key][0], before[key][0])
        for key in ("attention_mask", "loss_mask", "position_ids"):
            torch.testing.assert_close(batch[key], before[key])
        for row in (0, 1):
            torch.testing.assert_close(batch["multi_modal_inputs"][row]["image_grid_thw"],
                                       before["multi_modal_inputs"][row]["image_grid_thw"])
        torch.testing.assert_close(batch["multi_modal_inputs"][0]["pixel_values"], pixels)
        torch.testing.assert_close(batch["multi_modal_inputs"][1]["pixel_values"], -pixels)
        self.assertEqual(batch["responses"][1, 0].item(), 4)
        self.assertEqual(batch["input_ids"][1, 1].item(), 4)
        batch = copy.deepcopy(before)
        self.verification.change_neighbour(batch, 2, 4, row=0)
        self.assertEqual(batch["responses"][0, 0].item(), 4)
        for key in ("input_ids", "responses", "attention_mask", "loss_mask", "position_ids"):
            torch.testing.assert_close(batch[key][1], before[key][1])
        torch.testing.assert_close(batch["multi_modal_inputs"][1]["pixel_values"], pixels)
        torch.testing.assert_close(batch["multi_modal_inputs"][0]["pixel_values"], -pixels)

    def test_real_processor_observations_are_excluded_from_gradient_mask(self):
        path = os.environ.get("QWEN3_VL_PROCESSOR_PATH")
        if not path:
            self.skipTest("Set QWEN3_VL_PROCESSOR_PATH to a real Qwen3-VL processor directory")
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        torch = self.torch
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(path, local_files_only=True))
        for seed, case in enumerate(("text", "image", "crop", "long_tool_text"), 1):
            with self.subTest(case=case):
                sample = self.verification.build_sample(processor, case, seed)
                length, ids, _, loss_mask = sample
                response = ids[length:]
                eos = response.eq(processor.tokenizer.eos_token_id).nonzero().flatten()
                self.assertEqual(response.shape, loss_mask.shape)
                if case in ("crop", "long_tool_text"):
                    self.assertEqual(len(eos), 3)
                    self.assertTrue(loss_mask[:eos[0] + 1].bool().all())
                    self.assertEqual(loss_mask[eos[0] + 1:eos[1] + 1].sum().item(), 0)
                    self.assertTrue(loss_mask[eos[1] + 1:].bool().all())
                else:
                    self.assertTrue(loss_mask.bool().all())
                batch = self.verification.collate(model, processor.tokenizer.pad_token_id, [sample], "cpu")
                compact = self.verification.collate(model, processor.tokenizer.pad_token_id, [sample], "cpu", 0, 0)
                self.assertEqual(batch["loss_mask"][:, :length + 5].sum().item(), 0)
                self.assertEqual(batch["loss_mask"][:, -7:].sum().item(), 0)
                torch.testing.assert_close(batch["loss_mask"][:, 5:-7], compact["loss_mask"])
                torch.testing.assert_close(batch["position_ids"][:, :, 5:-7], compact["position_ids"])


if __name__ == "__main__":
    unittest.main()
