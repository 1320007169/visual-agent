import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_eval_multitool_vlocr_64gpu_16gpu_latest_8gpu_modelarts.sh"
RUNS = (
    "qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c",
    "qwen3base_multitool_vlocr_ocr_chart_n16_2node_from_step60_20261002T170651989009_040bfc67",
)


class RetryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="vlocr-retry-test-"))
        self.repo = self.base / "repo"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        self.previous = self.base / "previous"
        self.previous.mkdir()
        rows = ["checkpoint\tstep\tmodel_path"]
        self.models = []
        for label, run, latest, previous in zip(("64gpu", "16gpu"), RUNS, (70, 100), (80, 110)):
            output = self.repo / "saves/visual_agent_zwz_rl/qwen3" / run
            for step in (latest, previous):
                model = output / f"global_step_{step}/actor/huggingface"
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}")
                (model / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"weight": "model.safetensors"}}))
                (model / "model.safetensors").write_text(f"weights {step}")
            (output / "latest_checkpointed_iteration.txt").write_text(str(latest))
            self.models.append(model)
            rows.append(f"{label}\t{previous}\t{model}")
        (self.previous / "checkpoints.tsv").write_text("\n".join(rows) + "\n")
        (scripts / "run_visual_agent_eval_multitool_vlocr_8gpu.sh").write_text(
            '#!/usr/bin/env bash\n'
            'printf "EVAL|%s|%s|%s|%s\\n" "$VLOCR_STEP" "$VLOCR_MODEL_PATH" '
            '"${EVAL_DATASETS:-default}" "${VLOCR_REUSE_GROUP_ROOT:-}"\n'
            'if [[ "$VLOCR_STEP" == 80 ]]; then exit "${TEST_FIRST_EXIT:-0}"; fi\n')
        self.env = {
            "PATH": os.environ["PATH"], "BASE": str(self.base), "REPO_ROOT": str(self.repo),
            "WORK_ROOT": str(self.base / "results"), "RUN_ID": "retry_test",
        }

    def run_launcher(self, **changes):
        return subprocess.run(["bash", str(LAUNCHER)], env={**self.env, **changes},
                              capture_output=True, text=True, timeout=30)

    def retry(self, **changes):
        return self.run_launcher(**{
            "VLOCR_RETRY_FAILED_ONLY": "1", "VLOCR_REUSE_GROUP_ROOT": str(self.previous), **changes,
        })

    def test_default_still_selects_latest_and_all_benchmarks(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line.startswith("EVAL|")]
        self.assertEqual([row[1] for row in rows], ["70", "100"])
        self.assertEqual([row[3] for row in rows], ["default", "default"])

    def test_retry_pins_previous_weights_and_preserves_source(self):
        manifest = (self.previous / "checkpoints.tsv").read_bytes()
        for run in RUNS:
            output = self.repo / "saves/visual_agent_zwz_rl/qwen3" / run
            (output / "latest_checkpointed_iteration.txt").write_text("999")
        result = self.retry()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line.startswith("EVAL|")]
        self.assertEqual([row[1] for row in rows], ["80"])
        self.assertEqual([row[2] for row in rows], [str(self.models[0])])
        self.assertEqual([row[3] for row in rows], [
            "HRBench4K HRBench8K MME-RealWorld-Lite MME-RealWorld-CN",
        ])
        self.assertEqual([row[4] for row in rows], [str(self.previous)])
        self.assertEqual((self.previous / "checkpoints.tsv").read_bytes(), manifest)
        self.assertEqual((Path(self.env["WORK_ROOT"]) / "checkpoints.tsv").read_bytes(),
                         b"\n".join(manifest.splitlines()[:2]) + b"\n")

    def test_retry_requires_source_before_creating_output(self):
        result = self.run_launcher(VLOCR_RETRY_FAILED_ONLY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VLOCR_REUSE_GROUP_ROOT", result.stdout + result.stderr)
        self.assertFalse(Path(self.env["WORK_ROOT"]).exists())

    def test_invalid_mode_is_rejected(self):
        result = self.run_launcher(VLOCR_RETRY_FAILED_ONLY="invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(self.env["WORK_ROOT"]).exists())

    def test_retry_keeps_dataset_override_and_configuration_only_mode(self):
        result = self.retry(EVAL_DATASETS="HRBench8K", CHECKPOINT_EVAL_CONFIG_ONLY="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("64gpu_step80", result.stdout)
        self.assertNotIn("16gpu_step110", result.stdout)
        self.assertNotIn("EVAL|", result.stdout)
        self.assertFalse(Path(self.env["WORK_ROOT"]).exists())
        result = self.retry(EVAL_DATASETS="HRBench8K")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("|HRBench8K|"), 1)

    def test_missing_pinned_shard_stops_before_evaluation(self):
        (self.models[0] / "model.safetensors").write_bytes(b"")
        result = self.retry()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EVAL|", result.stdout)
        self.assertFalse(Path(self.env["WORK_ROOT"]).exists())

    def test_existing_results_are_preserved(self):
        output = Path(self.env["WORK_ROOT"])
        output.mkdir()
        marker = output / "old_result.txt"
        marker.write_text("previous results")
        result = self.retry()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EVAL|", result.stdout)
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "previous results")

    def test_missing_16gpu_weights_do_not_block_64gpu_retry(self):
        (self.models[1] / "model.safetensors").write_bytes(b"")
        result = self.retry()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("EVAL|"), 1)

    def test_retry_records_64gpu_failure(self):
        result = self.retry(TEST_FIRST_EXIT="17")
        self.assertEqual(result.returncode, 1)
        status = (Path(self.env["WORK_ROOT"]) / "status.tsv").read_text().splitlines()
        self.assertEqual([line.split("\t")[:2] for line in status[1:]], [
            ["64gpu_step80", "17"],
        ])


if __name__ == "__main__":
    unittest.main()
