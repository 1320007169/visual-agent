import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_qwen3vl_actor_forward.py"


class ActorForwardVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name == "check")
                 or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "LIMITS" for target in node.targets))]
        namespace = {"json": json}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
        cls.check = staticmethod(namespace["check"])

    def verify(self, **overrides):
        result = {"log_prob_max_abs": 0.0, "log_prob_mean_abs": 0.0, "entropy_max_abs": 0.0, "entropy_mean_abs": 0.0}
        result.update(overrides)
        output = io.StringIO()
        with redirect_stdout(output):
            passed = self.check("test", result, SimpleNamespace(
                max_abs=0.05, mean_abs=0.005, entropy_max_abs=0.05, entropy_mean_abs=0.005, grad_rel=0.02,
            ))
        report = json.loads(output.getvalue())
        self.assertEqual(report["passed"], passed)
        self.assertEqual(not report["failed"], passed)
        return passed

    def test_accepts_matching_outputs_with_or_without_gradients(self):
        self.assertTrue(self.verify())
        self.assertTrue(self.verify(gradient_relative_difference=0.0))

    def test_rejects_entropy_mismatch_even_when_log_probs_and_gradients_match(self):
        self.assertFalse(self.verify(entropy_max_abs=100.0, gradient_relative_difference=0.0))

    def test_rejects_nonfinite_entropy(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertFalse(self.verify(entropy_max_abs=value))

    def test_rejects_log_probability_or_gradient_mismatch(self):
        for name in ("log_prob_max_abs", "log_prob_mean_abs", "gradient_relative_difference"):
            with self.subTest(metric=name):
                self.assertFalse(self.verify(**{name: 1.0}))


if __name__ == "__main__":
    unittest.main()
