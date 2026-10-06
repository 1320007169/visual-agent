import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_multitool_vlocr_reliance_v2_lite_2node_16gpu_modelarts.sh"
RUN_ID = "qwen3base_multitool_vlocr_reliance_v2lite_n16_2node"


class RelianceV2LiteLauncherTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("prepare_visual_agent_run_paths.sh",
                     "run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"):
            (scripts / name).write_text((ROOT / "scripts" / name).read_text())
        self.env = {
            "PATH": os.environ["PATH"], "BASE": str(self.root), "REPO_ROOT": str(self.root),
            "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "new_job",
        }

    def launch(self, **overrides):
        return subprocess.run(
            ["bash", str(LAUNCHER)], env={**self.env, **overrides},
            capture_output=True, text=True,
        )

    def checkpoint(self, run_id=RUN_ID):
        checkpoint = self.root / "saves/visual_agent_zwz_rl/qwen3" / f"{run_id}_old_job/global_step_15"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"saved data cursor")
        for component in ("model", "optim", "extra_state"):
            for rank in range(14):
                (checkpoint / "actor" / f"{component}_world_size_14_rank_{rank}.pt").write_bytes(b"saved state")
        return checkpoint

    def test_first_launch_uses_base_data_and_has_no_deadline(self):
        result = self.launch(RESUME_MODE="resume_path", TRAIN_FILES="/v1/train.parquet")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(config["RESUME_MODE"], "disable")
        self.assertEqual(config["RESUME_FROM_PATH"], "")
        self.assertEqual(config["TRAINER_STOP_AFTER_SECONDS"], "0")
        self.assertEqual(config["TEST_FREQ"], "5")
        self.assertEqual(config["SAVE_FREQ"], "5")
        self.assertEqual(config["TRAIN_FILES"], str(self.root / "data/vlocr_reliance_pairs_v2_lite_20261006/train.parquet"))
        self.assertEqual(config["MODEL_PATH"], str(self.root / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        self.assertEqual(config["RUN_ID"], f"{RUN_ID}_new_job")

    def test_resume_keeps_source_state_and_uses_one_new_namespace_for_both_nodes(self):
        checkpoint = self.checkpoint()
        before = {str(p): p.read_bytes() for p in checkpoint.rglob("*.pt")}
        results = [self.launch(RESUME_FROM_PATH=str(checkpoint) + ("/" if rank else ""), NODE_RANK=str(rank),
                               TRAINER_STOP_AFTER_SECONDS="36000") for rank in (0, 1)]
        for result in results:
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(results[0].stdout, results[1].stdout)
        config = dict(line.split("=", 1) for line in results[0].stdout.splitlines())
        self.assertEqual(config["RESUME_MODE"], "resume_path")
        self.assertEqual(config["RESUME_FROM_PATH"], str(checkpoint))
        self.assertEqual(config["TRAINER_STOP_AFTER_SECONDS"], "36000")
        self.assertEqual(config["MODEL_PATH"], str(self.root / "DeepEyesV2/models/Qwen3-VL-8B-Instruct"))
        for key in ("RUN_ID", "RL_OUTPUT_DIR", "RL_LOG_DIR", "ROLLOUT_DATA_DIR"):
            self.assertTrue(config[key].endswith(f"{RUN_ID}_new_job"), (key, config[key]))
        self.assertEqual(before, {str(p): p.read_bytes() for p in checkpoint.rglob("*.pt")})

    def test_v1_checkpoint_is_rejected_before_launch(self):
        checkpoint = self.checkpoint("qwen3base_multitool_vlocr_reliance_n16_2node")
        result = self.launch(RESUME_FROM_PATH=str(checkpoint))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must belong to the v2-lite run", result.stderr)

    def test_incomplete_state_is_rejected_before_launch(self):
        checkpoint = self.checkpoint()
        for name in ("data.pt", "actor/optim_world_size_14_rank_13.pt",
                     "actor/extra_state_world_size_14_rank_0.pt"):
            with self.subTest(name=name):
                path = checkpoint / name
                original = path.read_bytes()
                path.write_bytes(b"")
                result = self.launch(RESUME_FROM_PATH=str(checkpoint))
                self.assertEqual(result.returncode, 2)
                self.assertIn("missing resume", result.stderr)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
