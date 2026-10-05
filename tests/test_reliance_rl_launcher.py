import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_multitool_vlocr_reliance_2node_16gpu_modelarts.sh"


class RelianceLauncherTest(unittest.TestCase):
    def test_judge_defaults_match_the_existing_modelarts_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            common = root / "scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"
            common.write_text(
                'printf "%s\\n" "$LLM_AS_A_JUDGE_BASE" "$LLM_AS_A_JUDGE_MODEL" "$JUDGE_KEY_FILE"\n'
            )
            env = {"PATH": os.environ["PATH"], "BASE": "/shared", "REPO_ROOT": directory,
                   "MULTITOOL_CONFIG_ONLY": "1"}
            result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [
                "https://api-cn.hi-code.cc/v1", "deepseek-v4.1-flash", "/shared/secrets/hicode_judge_api_key.txt",
            ])

    def test_fresh_counterfactual_run_ignores_inherited_resume_and_data_files(self):
        env = {
            "PATH": os.environ["PATH"], "BASE": str(ROOT.parent), "REPO_ROOT": str(ROOT),
            "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "reliance_test",
            "RESUME_MODE": "resume_path", "RESUME_FROM_PATH": "/old/global_step_210",
            "TRAIN_FILES": "/old/train.parquet", "VAL_FILES": "/old/val.parquet",
        }
        result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=True)
        config = dict(line.split("=", 1) for line in result.stdout.splitlines())
        data = ROOT / "data/vlocr_reliance_pairs_tallyhalf_fsc3000_20261006"
        self.assertEqual(config["MODEL_PATH"], str(ROOT.parent / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        self.assertEqual(config["TRAIN_FILES"], str(data / "train.parquet"))
        self.assertEqual(config["VAL_FILES"], str(data / "val.parquet"))
        self.assertEqual(config["RESUME_MODE"], "disable")
        self.assertEqual(config["NNODES"], "2")
        self.assertEqual(config["RL_CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6")
        self.assertEqual(config["TOOL_GPU"], "7")
        self.assertEqual(config["ROLLOUT_N"], "16")
        self.assertEqual(int(config["TRAIN_BATCH_SIZE"]) % 14, 0)
        self.assertEqual(int(config["PPO_MINI_BATCH_SIZE"]) * 16 % 14, 0)
        self.assertEqual(config["MAX_CONCURRENT_REQUESTS"], "112")
        self.assertEqual(config["TOTAL_TRAINING_STEPS"], "null")
        self.assertEqual(config["SAVE_FREQ"], "10")
        self.assertTrue(config["TOOL_CONFIG_PATH"].endswith("visual_tool_multitool_vlocr_config.yaml"))
        for key in ("RUN_ID", "RL_OUTPUT_DIR", "RL_LOG_DIR", "ROLLOUT_DATA_DIR"):
            self.assertTrue(config[key].endswith("qwen3base_multitool_vlocr_reliance_n16_2node_reliance_test"))

    def test_missing_prepared_data_stops_before_services_start(self):
        env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(ROOT),
               "MULTITOOL_DATA_DIR": str(ROOT / "data/nonexistent_reliance_test")}
        result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("prepare reliance pairs before launching", result.stderr)


if __name__ == "__main__":
    unittest.main()
