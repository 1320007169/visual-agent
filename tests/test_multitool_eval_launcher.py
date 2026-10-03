import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_visual_agent_eval_multitool_ocr_step20_8gpu.sh"
SINGLE_LAUNCHER = LAUNCHER.with_name("run_visual_agent_eval_multitool_ocr_24gpu_step40_8gpu.sh")


class MultitoolEvalLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        pipeline = root / "pipeline"
        pipeline.mkdir()
        (pipeline / ".env").write_text(f'VTS_OUTPUT_ROOT="{root}/bridge"\n')
        scripts = root / "repo/scripts"
        scripts.mkdir(parents=True)
        (scripts / LAUNCHER.name).symlink_to(LAUNCHER)
        (scripts / "run_visual_agent_eval_qwen3.sh").write_text(
            '#!/bin/bash\n'
            'mkdir -p "$WORK_ROOT"\n'
            'printf "CHECKPOINT|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n" '
            '"$RL_DINO_LATEST_STEP" "$RL_DINO_LATEST_MODEL_PATH" "$WORK_ROOT" '
            '"$EVAL_DATASETS" "$VISUAL_AGENT_ALLOWED_TOOL_NAMES" '
            '"$VISUAL_AGENT_MAX_TURNS" "$VISUAL_AGENT_MAX_TOKENS" '
            '"$VLMEVAL_API_NPROC" "$HF_HOME" "$VLMEVAL_CHARTQA_RULE_ONLY"\n'
            'printf "DIAGNOSTICS|%s|%s|%s\\n" "${VLMEVAL_FAILED_SAMPLE_RETRIES:-0}" '
            '"${VISUAL_AGENT_FAILURE_TRACE_DIR:-}" "${VISUAL_AGENT_FAILURE_REPORT:-0}"\n'
            'exit "${TEST_PREFLIGHT_EXIT_CODE:-0}"\n'
        )
        self.root = root
        self.env = dict(os.environ, BASE=str(root), REPO_ROOT=str(scripts.parent),
                        PIPELINE_ROOT=str(pipeline), RUN_ID="test_run", WORK_ROOT=str(root / "results"),
                        LOG_DIR=str(root / "logs"), HF_HOME=str(root / "hf_cache"), EVAL_PREFLIGHT_ONLY="1")
        for key in ("EVAL_DATASETS", "STEP20_MODEL_PATH", "STEP24_MODEL_PATH"):
            self.env.pop(key, None)

    def run_launcher(self, **changes):
        return subprocess.run(["bash", str(LAUNCHER)], env=dict(self.env, **changes),
                              text=True, capture_output=True, timeout=30)

    def test_sequential_checkpoints_share_protocol_and_have_separate_results(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("CHECKPOINT|")]
        self.assertEqual([row[0] for row in rows], ["20", "40"])
        self.assertIn("8node", rows[0][1])
        self.assertIn("3node", rows[1][1])
        self.assertTrue(rows[0][2].endswith("/64gpu_step20"))
        self.assertTrue(rows[1][2].endswith("/24gpu_step40"))
        self.assertEqual(rows[0][3:], rows[1][3:])
        self.assertEqual(rows[0][3].split(), ["VStarBench", "HRBench8K", "OCRBench",
                         "MME-RealWorld-Lite", "HRBench4K", "MME-RealWorld-CN", "CV-Bench-2D", "CV-Bench-3D", "ChartQA_TEST", "FSC147_TEST"])
        self.assertEqual(set(rows[0][4].split(",")), {"crop_zoom", "grounding_detect",
                         "depth_measure", "object_count", "text_detect", "text_recognize"})
        self.assertEqual(rows[0][5:8], ["8", "512", "14"])
        self.assertEqual(rows[0][8], str(self.root / "hf_cache"))
        self.assertEqual(rows[0][9], "1")

    def test_existing_output_is_rejected_before_services(self):
        output = self.root / "results"
        output.mkdir()
        marker = output / "old_result.txt"
        marker.write_text("previous experiment")
        result = self.run_launcher(EVAL_PREFLIGHT_ONLY="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("File exists", result.stdout + result.stderr)
        self.assertEqual(marker.read_text(), "previous experiment")

    def test_preflight_allows_normal_start_with_the_same_run_and_output(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "results"
        self.assertFalse(output.exists())
        rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("CHECKPOINT|")]
        for row in rows:
            preflight_output = Path(row[2])
            self.assertFalse(preflight_output.parent.exists())

        # Stop at data preparation: the fake repository has no preparation
        # script. Reaching services/ proves the fresh-output guard passed,
        # without starting any GPU services.
        normal = self.run_launcher(EVAL_PREFLIGHT_ONLY="0", EVAL_DATASETS="FSC147_TEST")
        self.assertNotEqual(normal.returncode, 0)
        self.assertNotIn("File exists", normal.stdout + normal.stderr)
        self.assertTrue((output / "services").is_dir())

    def test_failed_preflight_cleans_temporary_output_and_preserves_existing_results(self):
        output = self.root / "results"
        output.mkdir()
        marker = output / "old_result.txt"
        marker.write_text("previous experiment")
        result = self.run_launcher(TEST_PREFLIGHT_EXIT_CODE="17")
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("CHECKPOINT|")]
        self.assertEqual(len(rows), 1)
        self.assertFalse(Path(rows[0][2]).parent.exists())
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "previous experiment")

    def test_single_checkpoint_entry_only_selects_24gpu_step40(self):
        result = subprocess.run(["bash", str(SINGLE_LAUNCHER)], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("CHECKPOINT|")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "40")
        self.assertIn("3node", rows[0][1])
        self.assertTrue(rows[0][2].endswith("/24gpu_step40"))
        self.assertEqual(rows[0][5:8], ["8", "512", "14"])
        self.assertFalse((self.root / "results").exists())
        self.assertFalse(Path(rows[0][2]).parent.exists())
        diagnostics = next(line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("DIAGNOSTICS|"))
        self.assertEqual(diagnostics, ["3", str(self.root / "results/24gpu_step40/failures"), "1"])

    def test_invalid_checkpoint_is_rejected(self):
        result = self.run_launcher(EVAL_CHECKPOINTS="unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported checkpoint", result.stdout + result.stderr)


VLOCR_LAUNCHER = LAUNCHER.with_name("run_visual_agent_eval_multitool_vlocr_8gpu.sh")


class VlocrEvalLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        pipeline = root / "pipeline"
        pipeline.mkdir()
        (pipeline / ".env").write_text(f'VTS_OUTPUT_ROOT="{root}/bridge"\n')
        scripts = root / "repo/scripts"
        scripts.mkdir(parents=True)
        (scripts / "run_visual_agent_eval_qwen3.sh").write_text(
            '#!/bin/bash\n'
            'printf "PROTOCOL|%s|%s|%s|%s|%s|%s|%s|%s\\n" '
            '"$RL_DINO_LATEST_STEP" "$RL_DINO_LATEST_MODEL_PATH" "$VISUAL_AGENT_ALLOWED_TOOL_NAMES" '
            '"$VISUAL_AGENT_SYSTEM_PROMPT_FILE" "$VTS_VL_OCR" "$VTS_CHART_ENDPOINT" '
            '"$VISUAL_AGENT_MAX_TURNS" "$VISUAL_AGENT_MAX_TOKENS"\n'
            'printf "NATIVE|%s|%s\\n" "${VISUAL_AGENT_TOOL_CONFIG_PATH:-}" "${VLLM_TOOL_CALL_PARSER:-}"\n'
        )
        self.root = root
        self.env = dict(os.environ, BASE=str(root), REPO_ROOT=str(scripts.parent),
                        PIPELINE_ROOT=str(pipeline), RUN_ID="test_run", WORK_ROOT=str(root / "results"),
                        LOG_DIR=str(root / "logs"), EVAL_PREFLIGHT_ONLY="1",
                        VLOCR_MODEL_PATH="/ckpt/global_step_40/actor/huggingface", VLOCR_STEP="40")

    def test_protocol_matches_vlocr_training(self):
        result = subprocess.run(["bash", str(VLOCR_LAUNCHER)], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = next(line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("PROTOCOL|"))
        self.assertEqual(row[:2], ["40", "/ckpt/global_step_40/actor/huggingface"])
        self.assertEqual(set(row[2].split(",")), {"crop_zoom", "grounding_detect",
                         "depth_measure", "object_count", "ocr_read"})
        self.assertTrue(row[3].endswith("prompts/visual_agent_rl_system_multitool_vlocr.txt"))
        self.assertEqual(row[4:], ["1", "http://127.0.0.1:9007", "8", "512"])
        self.assertFalse((self.root / "results").exists())
        native = next(line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("NATIVE|"))
        self.assertTrue(native[0].endswith("tool_config/visual_tool_multitool_vlocr_config.yaml"))
        self.assertEqual(native[1], "hermes")

    def test_prompt_only_protocol_disables_native_tools(self):
        result = subprocess.run(["bash", str(VLOCR_LAUNCHER)], env=dict(self.env, VLOCR_NATIVE_TOOLS="0"),
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        native = next(line.split("|")[1:] for line in result.stdout.splitlines() if line.startswith("NATIVE|"))
        self.assertEqual(native, ["", ""])

    @unittest.skipUnless(shutil.which("setsid"), "requires util-linux setsid")
    def test_cleanup_stops_service_subprocesses(self):
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        pid_file = self.root / "server.pid"
        # Killing only the service parent would orphan its subprocess.
        wrapper = f'#!/bin/bash\nsleep 300 &\necho $! > "{pid_file}"\nwait\n'
        chart_env = self.root / "chart_env/bin"
        chart_env.mkdir(parents=True)
        (chart_env / "python3").write_text(wrapper)
        count_env = self.root / "count_env/bin"
        count_env.mkdir(parents=True)
        (count_env / "python3").write_text(wrapper.replace("server.pid", "count.pid"))
        # Slow health checks give the fake services time to start before the missing
        # depth service is noticed and cleanup runs.
        (fake_bin / "curl").write_text("#!/bin/bash\nsleep 2\nexit 7\n")
        for path in (chart_env / "python3", count_env / "python3", fake_bin / "curl"):
            path.chmod(0o755)
        # A missing depth environment makes that service exit at once and triggers cleanup.
        env = dict(self.env, EVAL_PREFLIGHT_ONLY="0", EVAL_DATASETS="VStarBench",
                   VLOCR_DEPTH_ENV=str(self.root / "missing_depth_env"),
                   VTS_COUNT_ENV=str(self.root / "count_env"), VTS_CHART_ENV=str(chart_env.parent),
                   PATH=f"{fake_bin}:{os.environ['PATH']}")
        result = subprocess.run(["bash", str(VLOCR_LAUNCHER)], env=env,
                                text=True, capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VTS service exited", result.stdout + result.stderr)
        for name in ("server.pid", "count.pid"):
            child = int((self.root / name).read_text())
            with self.assertRaises(ProcessLookupError, msg=f"{name} child survived cleanup"):
                os.kill(child, 0)

    def test_missing_checkpoint_is_rejected(self):
        env = dict(self.env)
        env.pop("VLOCR_MODEL_PATH")
        result = subprocess.run(["bash", str(VLOCR_LAUNCHER)], env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("VLOCR_MODEL_PATH", result.stderr)


if __name__ == "__main__":
    unittest.main()
