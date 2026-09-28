import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "scripts/prepare_visual_agent_run_paths.sh"


class RunPathsTest(unittest.TestCase):
    def resolve(self, root, **overrides):
        env = {
            "PATH": os.environ["PATH"], "BASE": str(root), "REPO_ROOT": str(root),
            "NNODES": "2", "RUN_ID": "experiment", "RESUME_MODE": "disable",
            "TRAIN_RUN_TOKEN": "job_a",
        }
        env.update(overrides)
        return subprocess.run(
            ["bash", "-euc", 'source "$1"; source "$1"; '
             'for key in RUN_ID RL_OUTPUT_DIR RL_LOG_DIR VALIDATION_DATA_DIR ROLLOUT_DATA_DIR; '
             'do printf "%s=%s\\n" "$key" "${!key:-}"; done', "bash", str(HELPER)],
            env=env, capture_output=True, text=True,
        )

    def test_fresh_runs_separate_paths_and_share_namespace_across_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node0 = self.resolve(root, VC_TASK_INDEX="0")
            node1 = self.resolve(root, VC_TASK_INDEX="1")
            next_job = self.resolve(root, TRAIN_RUN_TOKEN="job_b")
            self.assertEqual(node0.returncode, 0, node0.stderr)
            self.assertEqual(node0.stdout, node1.stdout)
            self.assertIn("RUN_ID=experiment_job_a\n", node0.stdout)
            self.assertIn(f"RL_LOG_DIR={root}/logs/visual-agent-zwz-rl/experiment_job_a", node0.stdout)
            self.assertNotEqual(node0.stdout, next_job.stdout)
            self.assertNotIn("job_a_job_a", node0.stdout)

    def test_fresh_run_rejects_existing_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "saves/visual_agent_zwz_rl/qwen3/experiment_job_a"
            (output / "global_step_80").mkdir(parents=True)
            result = self.resolve(root)
            self.assertEqual(result.returncode, 2)
            self.assertIn("already contains files", result.stderr)
            self.assertTrue((output / "global_step_80").is_dir())

    def test_resume_preserves_explicit_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.resolve(
                directory, RESUME_MODE="resume_path", RL_OUTPUT_DIR="/existing/weights",
                RL_LOG_DIR="/existing/logs", ROLLOUT_DATA_DIR="/existing/rollouts",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RUN_ID=experiment\n", result.stdout)
            self.assertIn("RL_OUTPUT_DIR=/existing/weights\n", result.stdout)
            self.assertIn("RL_LOG_DIR=/existing/logs\n", result.stdout)

    def test_manual_distributed_run_requires_shared_token(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.resolve(directory, TRAIN_RUN_TOKEN="")
            self.assertEqual(result.returncode, 2)
            self.assertIn("same TRAIN_RUN_TOKEN", result.stderr)

    def test_platform_job_id_provides_shared_token(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.resolve(directory, TRAIN_RUN_TOKEN="", VC_JOB_ID="platform_job")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RUN_ID=experiment_platform_job\n", result.stdout)

    def test_same_node_cannot_reuse_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.resolve(directory, VC_TASK_INDEX="0")
            second = self.resolve(directory, VC_TASK_INDEX="0")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 2)
            self.assertIn("already launched", second.stderr)

    def test_modelarts_bootstrap_creates_model_link_before_loading_helper(self):
        launcher = HELPER.parent / "run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"
        script = launcher.read_text()
        start = script.index("ensure_symlink() {")
        end = script.index('source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"', start)
        bootstrap = script[start:end] + 'source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_link = root / "model_link"
            dataset_link = root / "dataset_link"
            bootstrap = bootstrap.replace("/home/ma-user/work/model/xiaoyi_tmpstorage", str(model_link))
            bootstrap = bootstrap.replace("/opt/huawei/quoteModel/xiaoyi_tmpstorage", str(HELPER.parents[1]))
            bootstrap = bootstrap.replace("/opt/huawei/explorer-env/dataset", str(dataset_link))
            bootstrap = bootstrap.replace("/home/ma-user/work/dataset", str(root / "dataset_link2"))
            bootstrap = bootstrap.replace("/home/ma-user/work/algorithm/synaflow_wl", str(root / "algorithm_link"))
            result = subprocess.run(
                ["bash", "-euc", bootstrap], capture_output=True, text=True,
                env={"PATH": os.environ["PATH"], "BASE": directory,
                     "REPO_ROOT": str(model_link), "TRAIN_RUN_TOKEN": "bootstrap_test",
                     "MULTITOOL_CONFIG_ONLY": "1", "RESUME_MODE": "disable"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((model_link / "scripts/prepare_visual_agent_run_paths.sh").is_file())


if __name__ == "__main__":
    unittest.main()
