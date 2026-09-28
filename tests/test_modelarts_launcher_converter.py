import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("converter", ROOT / "scripts/convert_training_launcher_to_modelarts.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConverterTest(unittest.TestCase):
    def test_three_node_launcher_and_conversion_keep_divisible_batches(self):
        launcher = "scripts/run_visual_agent_multitool_depth_count_3node_24gpu.sh"
        env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(ROOT),
               "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "three_node_test"}
        original = subprocess.run(["bash", str(ROOT / launcher)], env=env, capture_output=True, text=True, check=True)
        settings = dict(line.split("=", 1) for line in original.stdout.splitlines())
        total_rl_gpus = int(settings["NNODES"]) * len(settings["RL_CUDA_VISIBLE_DEVICES"].split(","))
        self.assertEqual(total_rl_gpus, 21)
        batch = int(settings["TRAIN_BATCH_SIZE"])
        mini = int(settings["PPO_MINI_BATCH_SIZE"])
        samples = int(settings["ROLLOUT_N"])
        self.assertEqual(batch % total_rl_gpus, 0)
        self.assertEqual(mini * samples % total_rl_gpus, 0)
        self.assertEqual(batch % mini, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "three_node_modelarts.sh"
            arguments = [launcher, "-o", str(output)]
            for key in ("NNODES", "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "VAL_BATCH_SIZE",
                        "ROLLOUT_N", "MAX_CONCURRENT_REQUESTS", "RUN_ID"):
                arguments.extend(["--env", f"{key}={settings[key]}"])
            converter.main(arguments)
            result = subprocess.run(["bash", str(output)], env=env, capture_output=True, text=True, check=True)
            converted = dict(line.split("=", 1) for line in result.stdout.splitlines())
            for key in ("NNODES", "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "VAL_BATCH_SIZE",
                        "ROLLOUT_N", "MAX_CONCURRENT_REQUESTS", "TOTAL_TRAINING_STEPS"):
                self.assertEqual(converted[key], settings[key])
            self.assertEqual(converted["LAUNCHER"], str(ROOT / launcher))

    def test_multitool_config_and_safe_environment_quoting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "entry.sh"
            converter.main([
                "scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh", "-o", str(output),
                "--env", "ROLLOUT_N=16", "--env", "NNODES=2", "--env", "RUN_ID=converted_n16",
                "--env", "RESUME_MODE=disable", "--env", "CUSTOM_LABEL=$(touch /never_execute)`id`",
            ])
            env = dict(os.environ, MULTITOOL_CONFIG_ONLY="1", TRAIN_RUN_TOKEN="test_job")
            env.pop("VISUAL_AGENT_RUN_PATHS_READY", None)
            result = subprocess.run(["bash", str(output)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ROLLOUT_N=16\n", result.stdout)
            self.assertIn("RUN_ID=converted_n16_test_job\n", result.stdout)
            self.assertIn(f"LAUNCHER={ROOT}/scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh", result.stdout)
            self.assertTrue(os.access(output, os.X_OK))

    def test_basic_wrapper_passes_arguments_without_shell_execution(self):
        script = converter.render_entrypoint(Path("scripts/train.sh"), "basic", [], ["space value", "$(id)"])
        syntax = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertIn('exec bash "$launcher" \'space value\' \'$(id)\' "$@"', script)
        self.assertLess(script.index("ensure_symlink /opt/huawei/quoteModel"), script.index('launcher="$REPO_ROOT"'))

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "entry.sh"
            output.write_text("original")
            with self.assertRaises(SystemExit):
                converter.main(["scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh", "-o", str(output)])
            self.assertEqual(output.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
