import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import visual_agent_rl_resource_benchmark as benchmark
import run_visual_tool_cpu_server as cpu_server
from run_visual_tool_cpu_server import server_cores
from summarize_rl_resource_benchmark import summarize


def arguments(root, scenario="A"):
    return argparse.Namespace(run_id="test", scenario=scenario, warmup=2, steps=5,
                              model_path=str(root / "model"), output_root=root,
                              cpu_threads=16, cpu_servers=2)


class RLResourceBenchmarkTest(unittest.TestCase):
    def test_scenarios_and_batch_divisibility(self):
        for scenario, gpus, servers, replicas in [("A", 14, 2, 3), ("B", 14, 1, 1), ("C", 16, 2, 1)]:
            env, meta, train, tools = benchmark.configuration(arguments(Path("/tmp"), scenario), {}, range(96))
            self.assertEqual(meta["training_gpus"], gpus)
            self.assertEqual(int(env["VISUAL_TOOL_SERVERS_PER_NODE"]), servers)
            self.assertEqual(int(env["GROUNDING_DINO_REPLICAS"]), replicas)
            self.assertEqual(112 % gpus, 0)
            self.assertEqual((28 * 8) % gpus, 0)
            self.assertFalse(set(train) & set(tools))
            self.assertEqual(env["SAVE_FREQ"], "-1")
            self.assertEqual(env["SAVE_BEST_HF_MODEL"], "False")
            if scenario == "C":
                self.assertEqual(len(tools), 32)
                self.assertEqual(len(train), 64)
                self.assertEqual(env["VISUAL_TOOL_DEVICE"], "cpu")

    def test_inherited_production_state_is_removed(self):
        parent = dict(CONTINUE_FROM_PATH="production", RESUME_FROM_PATH="production",
                      WARM_START_GLOBAL_STEP="100", VISUAL_TOOL_API_BASES="http://production",
                      LOG_FILE="production.log", RESUME_MODE="auto", DRY_RUN="1")
        env, _, _, _ = benchmark.configuration(arguments(Path("/tmp")), parent, range(96))
        self.assertEqual(env["RESUME_MODE"], "disable")
        for key in set(parent) - {"RESUME_MODE"}:
            self.assertNotIn(key, env)

    def test_cpu_budget_and_partition(self):
        with self.assertRaises(ValueError):
            benchmark.cpu_partition(range(40), 16, 2)
        self.assertEqual(server_cores("32,33,34,35", 2, 1), [34, 35])
        with self.assertRaises(ValueError):
            server_cores("32,33", 2, 1)

    def test_cpu_server_disables_cuda_and_sets_threads(self):
        from unittest.mock import Mock
        torch = types.SimpleNamespace(set_num_threads=Mock(), set_num_interop_threads=Mock(),
                                      get_num_threads=lambda: 2)
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, {}, clear=False), \
             patch.object(sys, "argv", ["cpu-server", "--cores", "32,33,34,35", "--threads", "2",
                                         "--server-index", "1", "--backend", "groundingdino"]), \
             patch.object(cpu_server.os, "sched_setaffinity") as affinity, \
             patch.object(cpu_server.runpy, "run_path") as run:
            cpu_server.main()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["GROUNDING_DINO_DEVICE"], "cpu")
            self.assertEqual(sys.argv[1:], ["--backend", "groundingdino"])
            affinity.assert_called_once_with(0, [34, 35])
            torch.set_num_threads.assert_called_once_with(2)
            torch.set_num_interop_threads.assert_called_once_with(1)
            run.assert_called_once()

    def test_mixed_wrapper_preserves_all_three_scenarios(self):
        with tempfile.TemporaryDirectory() as temp:
            for scenario in ("A", "B", "C"):
                env, _, _, _ = benchmark.configuration(arguments(Path(temp), scenario), os.environ, range(96))
                env.update(MIXED_CONFIG_ONLY="1", CONFIG_PYTHON=sys.executable)
                result = subprocess.run(["bash", str(SCRIPTS / "run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh")],
                                        env=env, text=True, capture_output=True, check=True)
                values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                for key in ("RL_CUDA_VISIBLE_DEVICES", "VISUAL_TOOL_DEVICE", "VISUAL_TOOL_SERVERS_PER_NODE",
                            "GROUNDING_DINO_REPLICAS", "TOTAL_TRAINING_STEPS", "RESUME_MODE", "SAVE_BEST_HF_MODEL"):
                    self.assertEqual(values[key], env[key], (scenario, key))
                self.assertFalse((Path(temp) / "test" / scenario).exists())

    def test_registration_rejects_retry_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark.register(root, {"scenario": "A"}, {"node": "one"})
            with self.assertRaises(FileExistsError):
                benchmark.register(root, {"scenario": "A"}, {"node": "one"})
            with self.assertRaises(ValueError):
                benchmark.register(root, {"scenario": "B"}, {"node": "one"})
            with patch.object(benchmark.socket, "gethostname", return_value="second-node"):
                benchmark.register(root, {"scenario": "A"}, {"node": "two"})
            with patch.object(benchmark.socket, "gethostname", return_value="third-node"):
                with self.assertRaises(ValueError):
                    benchmark.register(root, {"scenario": "A"}, {"node": "three"})

    def test_summary_excludes_warmup_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            (root / "rollouts").mkdir()
            (root / "benchmark.json").write_text(json.dumps({
                "warmup_steps": 2, "total_steps": 4, "scenario": "A", "run_name": "test",
                "train_batch_size": 1, "rollout_n": 1}))
            (root / "logs/test-node0.log").write_text("\n".join(
                f"step:{step} - training/global_step:{step} - timing_s/step:{seconds} - timing_s/gen:10"
                for step, seconds in [(1, 900), (2, 800), (3, 100), (4, 120)]))
            for step in (3, 4):
                (root / f"rollouts/{step}.jsonl").write_text(json.dumps({"rollout_trace": {
                    "tool_calls": [{"status": "success" if step == 3 else "error", "latency_ms": step * 100}]}}))
            result = summarize(root)
            self.assertEqual(result["metrics"]["timing_s/step"]["median"], 110)
            self.assertEqual(result["tool_errors"], 1)
            self.assertEqual(result["tool_calls"], 2)
            self.assertEqual(result["status"], "incomplete_or_failed")
            self.assertTrue(result["trace_coverage_complete"])
            for node in (0, 1):
                (root / f"exit-node{node}.json").write_text('{"exit_code":0}')
            self.assertEqual(summarize(root)["status"], "complete")

    def test_shell_syntax(self):
        for name in ("run_visual_agent_rl_resource_benchmark_modelarts.sh",
                     "run_visual_agent_zwz_deepeyesv2_n8_2node_16gpu.sh", "run_visual_tool_rl_2node_16gpu.sh"):
            subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


if __name__ == "__main__":
    unittest.main()
