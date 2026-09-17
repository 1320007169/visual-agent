import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest


SPEC = importlib.util.spec_from_file_location(
    "resume", Path(__file__).parents[1] / "scripts/resolve_mixed_rl_resume.py"
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class MixedResumeTest(unittest.TestCase):
    def make_checkpoint(self, root, world=14):
        path = root / "source/global_step_40"
        actor = path / "actor"
        actor.mkdir(parents=True)
        for kind in ("model", "optim", "extra_state"):
            for rank in range(world):
                (actor / f"{kind}_world_size_{world}_rank_{rank}.pt").touch()
        (path / "data.pt").touch()
        (actor / "huggingface").mkdir()
        (actor / "huggingface/config.json").write_text("{}")
        (actor / "huggingface/model.safetensors").touch()
        (path.parent / "mixed_run_config.json").write_text(json.dumps({
            "initial_model_path": str(root / "base_model"),
            "train_files": str(root / "data/zwz_deepeyesv2_3k_nocount_hr4k_v1/train.parquet"),
        }))
        return path

    def test_new_runs_and_batch_divisibility(self):
        with tempfile.TemporaryDirectory() as directory:
            for nodes in (2, 4, 8):
                values, _ = resume.resolve({"BASE": directory, "REPO_ROOT": directory, "NNODES": str(nodes)})
                self.assertEqual(values["RESUME_MODE"], "auto")
                self.assertEqual(112 % (nodes * 7), 0)
                self.assertEqual(28 * 8 % (nodes * 7), 0)

    def test_exact_and_cross_topology_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_checkpoint(root)
            env = {"BASE": directory, "REPO_ROOT": directory, "CONTINUE_FROM_PATH": str(source)}
            values, _ = resume.resolve(env)
            self.assertEqual(values["RESUME_MODE"], "resume_path")
            for nodes in ("4", "8"):
                values, metadata = resume.resolve(dict(env, NNODES=nodes))
                self.assertEqual(values["RESUME_MODE"], "auto")
                self.assertEqual(values["WARM_START_GLOBAL_STEP"], "40")
                self.assertEqual(values["MODEL_PATH"], str(source / "actor/huggingface"))
                output = Path(values["RL_OUTPUT_DIR"])
                output.mkdir(parents=True)
                (output / "mixed_run_config.json").write_text(json.dumps(metadata))
                retried, retry_meta = resume.resolve({
                    "BASE": directory, "REPO_ROOT": directory, "NNODES": nodes,
                    "RL_OUTPUT_DIR": str(output), "RUN_ID": values["RUN_ID"],
                })
                self.assertEqual(retried["MODEL_PATH"], values["MODEL_PATH"])
                self.assertEqual(metadata, retry_meta)
            (source / "actor/optim_world_size_14_rank_5.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                resume.resolve(env)

    def test_metadata_read_retries_during_concurrent_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_run_config.json"
            path.touch()
            expected = {"nnodes": 8, "train_files": "train.parquet"}

            def publish():
                time.sleep(0.05)
                resume.write_json_atomic(path, expected)

            writer = threading.Thread(target=publish)
            writer.start()
            actual = resume.read_json_with_retry(path, attempts=20, delay=0.01)
            writer.join()
            self.assertEqual(actual, expected)
            self.assertEqual(json.loads(path.read_text()), expected)
            self.assertEqual(list(path.parent.glob(".mixed_run_config.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
