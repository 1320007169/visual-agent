import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_visual_agent_multitool_vlocr_reliance_v3_8node_64gpu_modelarts.sh"


class RelianceV3LauncherTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="v3-launcher-test-"))
        provenance = json.loads((ROOT / "configs/tool_reliance_v3_k2_data_20261008.json").read_text())
        for filename in [*provenance["generation_code_sha256"], "scripts/prepare_visual_agent_run_paths.sh",
                         "scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"]:
            target = self.root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / filename).read_bytes())
        for variant, spec in provenance["datasets"].items():
            directory = self.root / spec["directory"]
            directory.mkdir(parents=True)
            for name, key in (("train.parquet", "train_sha256"), ("val.parquet", "val_sha256")):
                content = (variant + name).encode()
                (directory / name).write_bytes(content)
                spec[key] = hashlib.sha256(content).hexdigest()
            manifest = {
                "generation_commit": provenance["generation_commit"],
                "generation_code_sha256": provenance["generation_code_sha256"],
                "dataset_variant": variant, "counterfactual_variants": 2,
                "replacement_scope": "same_source", "train_sources_before": {"ocr": 6},
                "train_sources_after": {"ocr": 6}, "replacement_groups_before": {"ocr|hme": 6},
                "replacement_groups_after": {"ocr|hme": 6}, "prefixed_groups": {} if variant == "V" else {"ocr|hme": 3},
                "ocr_fault_policy": "hme_visible_symbols_answer_excluded", "max_replaced_fraction_per_original_source": 0.5,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "configs").mkdir()
        (self.root / "configs/tool_reliance_v3_k2_data_20261008.json").write_text(json.dumps(provenance))
        self.provenance = provenance
        self.env = {"PATH": os.environ["PATH"], "BASE": str(self.root), "REPO_ROOT": str(self.root),
                    "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "new_job"}

    def launch(self, **overrides):
        return subprocess.run(["bash", str(LAUNCHER)], env={**self.env, **overrides}, capture_output=True, text=True)

    def checkpoint(self, arm="paired", nodes=2):
        run = f"qwen3base_multitool_vlocr_reliance_v3_{arm}_k2_n16_{nodes}node_old_job"
        path = self.root / "saves/visual_agent_zwz_rl/qwen3" / run / "global_step_5"
        (path / "actor").mkdir(parents=True)
        (path / "data.pt").write_bytes(b"data cursor")
        for rank in range(nodes * 7):
            for kind in ("model", "optim", "extra_state"):
                (path / "actor" / f"{kind}_world_size_{nodes * 7}_rank_{rank}.pt").write_bytes(b"state")
        return path

    def test_arms_have_matching_training_settings_and_distinct_outputs(self):
        for nodes, batch in (("2", "126"), ("8", "336")):
            configs = []
            for arm, variant in (("paired", "P"), ("factual", "F"), ("unprefixed", "V")):
                result = self.launch(V3_VARIANT=arm, NNODES=nodes)
                self.assertEqual(result.returncode, 0, result.stderr)
                config = dict(line.split("=", 1) for line in result.stdout.splitlines())
                self.assertEqual(config["V3_DATA_VARIANT"], variant)
                self.assertEqual(config["TRAIN_BATCH_SIZE"], batch)
                self.assertEqual(config["RESUME_MODE"], "disable")
                self.assertEqual(config["RESUME_FROM_PATH"], "")
                self.assertEqual(config["SAVE_FREQ"], "5")
                self.assertEqual(config["TEST_FREQ"], "5")
                self.assertEqual(config["TRAINER_STOP_AFTER_SECONDS"], "0")
                configs.append(config)
            varying = {"V3_DATA_VARIANT", "V3_TRAIN_SHA256", "TRAIN_FILES", "VAL_FILES", "RUN_ID",
                       "RL_OUTPUT_DIR", "RL_LOG_DIR", "ROLLOUT_DATA_DIR"}
            for config in configs[1:]:
                self.assertEqual({k: v for k, v in config.items() if k not in varying},
                                 {k: v for k, v in configs[0].items() if k not in varying})
            for key in ("RUN_ID", "RL_OUTPUT_DIR", "ROLLOUT_DATA_DIR", "RL_LOG_DIR"):
                self.assertEqual(len({config[key] for config in configs}), 3)

    def test_full_resume_preserves_source_and_uses_fresh_output(self):
        path = self.checkpoint()
        before = {str(p): p.read_bytes() for p in path.rglob("*.pt")}
        result = self.launch(NNODES="2", RESUME_FROM_PATH=str(path), TRAINER_STOP_AFTER_SECONDS="28800")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(config["RESUME_MODE"], "resume_path")
        self.assertEqual(config["RESUME_FROM_PATH"], str(path))
        self.assertEqual(config["TRAINER_STOP_AFTER_SECONDS"], "28800")
        self.assertTrue(config["RL_OUTPUT_DIR"].endswith("_new_job"))
        self.assertEqual(before, {str(p): p.read_bytes() for p in path.rglob("*.pt")})

    def test_wrong_arm_topology_and_incomplete_checkpoints_are_rejected(self):
        path = self.checkpoint()
        for overrides in ({"V3_VARIANT": "factual", "NNODES": "2"}, {"NNODES": "8"}):
            result = self.launch(RESUME_FROM_PATH=str(path), **overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("this v3 arm and topology", result.stderr)
        shard = path / "actor/optim_world_size_14_rank_13.pt"
        shard.write_bytes(b"")
        result = self.launch(NNODES="2", RESUME_FROM_PATH=str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing resume shard", result.stderr)

    def test_modified_data_and_unpublished_code_are_rejected(self):
        dataset = self.root / self.provenance["datasets"]["P"]["directory"] / "train.parquet"
        before = dataset.read_bytes()
        dataset.write_bytes(b"different dataset")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dataset content differs", result.stderr)
        dataset.write_bytes(before)
        script = self.root / "scripts/prepare_reliance_pairs.py"
        script.write_text(script.read_text() + "\n# Unpublished change\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("code differs from the pinned version", result.stderr)


if __name__ == "__main__":
    unittest.main()
