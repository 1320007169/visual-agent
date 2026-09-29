import ast
import json
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


if __name__ == "__main__":
    unittest.main()
