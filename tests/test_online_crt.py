import ast
import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "reinforcement_learning/verl"
SCHEDULER = RL / "workers/rollout/chat_scheduler.py"
CONTROLLER = RL / "trainer/ppo/online_faults.py"
LAUNCHER = ROOT / "scripts/run_visual_agent_multitool_vlocr_online_crt_modelarts.sh"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool_faults = load_module("tool_faults", RL / "tools/tool_faults.py")
online_tool_faults = load_module("online_tool_faults", RL / "tools/online_tool_faults.py")
online_faults = load_module("online_faults", CONTROLLER)


def load_call_tool():
    tree = ast.parse(SCHEDULER.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_call_tool")
    namespace = {"Any": object, "Dict": dict, "json": json, "time": time,
                 "logger": logging.getLogger(__name__), "tool_faults": tool_faults,
                 "online_tool_faults": online_tool_faults}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SCHEDULER), "exec"), namespace)
    return namespace["_call_tool"]


class FakeBatch:
    def __init__(self, **rows):
        self.non_tensor_batch = rows

    def __len__(self):
        return len(next(iter(self.non_tensor_batch.values())))


class FakeCountTool:
    def __init__(self):
        self.calls = []

    async def create(self, images):
        return "count"

    async def execute(self, instance_id, arguments):
        self.calls.append(arguments)
        return json.dumps({"count": 7}), 0.0, {}

    async def release(self, instance_id):
        return None


class OnlineCRTTest(unittest.TestCase):
    def test_first_call_is_faulted_equivalent_call_replays_and_other_region_is_real(self):
        tool = FakeCountTool()
        info = {"images": ["same", "same", "different"], "online_fault": {"tool": "object_count", "level": 0,
                "seed": 3, "count_truth": "7"}, "__trace__": {"model_calls": [], "tool_calls": []}}
        call = load_call_tool()
        callback = SimpleNamespace(tools={"object_count": tool})

        def invoke(arguments):
            return asyncio.run(call(callback, {"name": "object_count", "arguments": arguments}, info,
                                    xml_mode=True))["content"]

        first = invoke({"query": " apples ", "target_image": 0})
        self.assertNotIn('"count": 7', first)
        self.assertEqual(len(tool.calls), 1)
        self.assertIn("injected_fault", info["__trace__"]["tool_calls"][0])
        self.assertEqual(first, invoke({"query": "apples", "target_image": 1}))
        self.assertTrue(info["__trace__"]["tool_calls"][1]["replayed_fault"])
        self.assertEqual(len(tool.calls), 1)
        self.assertIn('"count": 7', invoke({"query": "apples", "target_image": 2}))

    def test_no_spec_keeps_real_response_and_validation_ignores_specs(self):
        tool = FakeCountTool()
        info = {"images": [], "__trace__": {"model_calls": [], "tool_calls": []}}
        callback = SimpleNamespace(tools={"object_count": tool})
        message = asyncio.run(load_call_tool()(callback, {"name": "object_count", "arguments": {
            "query": "apples", "target_image": 0}}, info, xml_mode=True))
        self.assertIn('"count": 7', message["content"])
        self.assertNotIn("reliance_fault", info)

        tree = ast.parse(SCHEDULER.read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "online_faults"
                                  for target in node.targets))
        expression = compile(ast.Expression(body=assignment.value), str(SCHEDULER), "eval")
        batch = SimpleNamespace(non_tensor_batch={"online_fault": ["fault"]}, meta_info={"validate": True})
        with patch.dict(os.environ, {"VISUAL_AGENT_ONLINE_FAULTS_CONFIG": "enabled"}):
            self.assertIsNone(eval(expression, {"batch": batch, "os": os}))
            batch.meta_info["validate"] = False
            self.assertEqual(eval(expression, {"batch": batch, "os": os}), ["fault"])
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(eval(expression, {"batch": batch, "os": os}))

    def test_full_image_alias_is_recorded_before_first_fault(self):
        class CropTool(FakeCountTool):
            async def execute(self, instance_id, arguments):
                return '{"target_image": 1}', 0.0, {
                    "returned_images": ["reencoded"],
                    "raw_result": {"crop_zoom": {"bbox_2d": [0, 0, 1000, 1000]}},
                }

        info = {"images": ["original"], "online_fault": {"tool": "object_count", "level": 0,
                "seed": 3, "count_truth": "7"}, "__trace__": {"model_calls": [], "tool_calls": []}}
        call = load_call_tool()
        callback = SimpleNamespace(tools={"crop_zoom": CropTool(), "object_count": FakeCountTool()})
        asyncio.run(call(callback, {"name": "crop_zoom", "arguments": {
            "target_image": 0, "bbox_2d": [0, 0, 1000, 1000]}}, info, xml_mode=True))
        first = asyncio.run(call(callback, {"name": "object_count", "arguments": {
            "query": "apples", "target_image": 0}}, info, xml_mode=True))
        replay = asyncio.run(call(callback, {"name": "object_count", "arguments": {
            "query": "apples", "target_image": 1}}, info, xml_mode=True))
        self.assertEqual(first["content"], replay["content"])
        self.assertTrue(info["__trace__"]["tool_calls"][-1]["replayed_fault"])
        self.assertEqual(len(callback.tools["object_count"].calls), 1)

    def test_error_observation_does_not_consume_first_successful_call(self):
        class InitiallyFailingTool(FakeCountTool):
            async def execute(self, instance_id, arguments):
                self.calls.append(arguments)
                if len(self.calls) == 1:
                    return '{"status": "error", "message": "temporary failure"}', 0.0, {}
                return '{"count": 7}', 0.0, {}

        tool = InitiallyFailingTool()
        info = {"images": [], "online_fault": {"tool": "object_count", "level": 0,
                "seed": 3, "count_truth": "7"}, "__trace__": {"model_calls": [], "tool_calls": []}}
        call = load_call_tool()
        callback = SimpleNamespace(tools={"object_count": tool})
        arguments = {"name": "object_count", "arguments": {"query": "apples", "target_image": 0}}
        first = asyncio.run(call(callback, arguments, info, xml_mode=True))
        self.assertIn('"status": "error"', first["content"])
        self.assertNotIn("online_fault_attempted", info)
        second = asyncio.run(call(callback, arguments, info, xml_mode=True))
        self.assertNotIn('"count": 7', second["content"])
        self.assertEqual(len(tool.calls), 2)
        self.assertIn("injected_fault", info["__trace__"]["tool_calls"][1])

    def test_different_ocr_region_returns_real_observation(self):
        class OcrTool(FakeCountTool):
            async def execute(self, instance_id, arguments):
                self.calls.append(arguments)
                return '{"text": "120", "truncated": false}', 0.0, {}

        tool = OcrTool()
        info = {"images": ["original"], "online_fault": {"tool": "ocr_read", "level": 0,
                "seed": 2, "answer": "120", "aliases": ["120"]},
                "__trace__": {"model_calls": [], "tool_calls": []}}
        call = load_call_tool()
        callback = SimpleNamespace(tools={"ocr_read": tool})

        def invoke(arguments):
            return asyncio.run(call(callback, {"name": "ocr_read", "arguments": arguments},
                                    info, xml_mode=True))["content"]

        faulty = invoke({"target_image": 0})
        self.assertNotIn('"text": "120"', faulty)
        self.assertEqual(faulty, invoke({"target_image": 0, "mode": "text",
                                         "bbox_2d": [0, 0, 1000, 1000]}))
        self.assertIn('"text": "120"', invoke({"target_image": 0, "bbox_2d": [0, 0, 500, 500]}))
        self.assertEqual(len(tool.calls), 2)

    def test_controller_assign_update_and_restore(self):
        controller = online_faults.OnlineFaultController(str(ROOT / "configs/online_crt_v1.json"))
        prompts = FakeBatch(data_source=["visual-agent-fsc147"] * 100 + ["other"],
                            extra_info=[{"index": index} for index in range(101)],
                            reward_model=[{"ground_truth": "12"}] * 101)
        specs = controller.assign(prompts)
        selected = [json.loads(spec) for spec in specs if spec]
        self.assertTrue(0 < len(selected) < 100)
        self.assertEqual(specs[-1], "")
        self.assertTrue(all(spec["count_truth"] == "12" and spec["bucket_key"] ==
                            "visual-agent-fsc147|object_count" for spec in selected))
        controller.eligible["visual-agent-fsc147"].append("grounding_detect")
        varied = {json.loads(spec)["tool"] for spec in controller.assign(prompts) if spec}
        self.assertEqual(varied, {"object_count", "grounding_detect"})
        controller.eligible["visual-agent-fsc147"] = ["object_count"]
        key_count = "visual-agent-fsc147|object_count"
        key_ocr, _ = controller._bucket("visual-agent-ocr", {"original_source": "hme100k"})
        self.assertEqual(key_ocr, "visual-agent-ocr|ocr_read|hme")

        count_spec = json.dumps({"bucket_key": key_count})
        ocr_spec = json.dumps({"bucket_key": key_ocr})
        specs = [count_spec] * 16 + [ocr_spec] * 16 + [ocr_spec] * 16
        traces = [{"tool_calls": [{"injected_fault": {"level": 0}}]} for _ in range(32)]
        traces.extend({"tool_calls": []} for _ in range(16))
        batch = FakeBatch(uid=["count"] * 16 + ["ocr"] * 16 + ["not-injected"] * 16,
                          online_fault=specs, rollout_trace=traces,
                          acc=[0] * 16 + [0, 1] * 8 + [1] * 16)
        for _ in range(5):
            metrics = controller.update(batch)
        self.assertEqual(controller.buckets[key_count]["level"], 1)
        self.assertEqual(controller.buckets[key_ocr]["level"], 0)
        self.assertGreater(controller.buckets[key_ocr]["fraction"],
                           controller.buckets[key_count]["fraction"])
        self.assertGreaterEqual(controller.buckets[key_count]["fraction"], 0.02)
        self.assertEqual(metrics["online_faults/visual-agent-ocr/ocr_read/hme/injected_groups"], 1)
        self.assertEqual(metrics["online_faults/visual-agent-ocr/ocr_read/hme/assigned_groups"], 2)
        state = json.loads(json.dumps(controller.state_dict()))
        restored = online_faults.OnlineFaultController(str(ROOT / "configs/online_crt_v1.json"))
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.assign(prompts), controller.assign(prompts))
        with self.assertRaisesRegex(ValueError, "16 rollouts"):
            restored.update(FakeBatch(uid=["short"], online_fault=[count_spec],
                                      rollout_trace=traces[:1], acc=[0]))

    def test_launcher_config_for_two_and_eight_nodes(self):
        for nodes, batch_size in (("2", "126"), ("8", "336")):
            result = subprocess.run(["bash", str(LAUNCHER)], env={"PATH": os.environ["PATH"],
                "REPO_ROOT": str(ROOT), "BASE": str(ROOT.parent), "NNODES": nodes,
                "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "test"},
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = dict(line.split("=", 1) for line in result.stdout.splitlines())
            self.assertEqual(config["REPO_ROOT"], str(ROOT))
            self.assertEqual(config["CURRENT_COMMIT"], subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip())
            self.assertEqual(config["TRAIN_BATCH_SIZE"], batch_size)
            self.assertEqual(config["ROLLOUT_N"], "16")
            self.assertEqual(config["VISUAL_AGENT_ONLINE_FAULTS_CONFIG"],
                             str(ROOT / "configs/online_crt_v1.json"))
            self.assertIn("zwz_multitool_relation20_hme_chartqa_tallyhalf_fsc3000_20261004",
                          config["TRAIN_FILES"])
        invalid = subprocess.run(["bash", str(LAUNCHER)], env={"PATH": os.environ["PATH"],
            "REPO_ROOT": str(ROOT), "NNODES": "3", "MULTITOOL_CONFIG_ONLY": "1"},
            capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        checkpoint = Path(tempfile.mkdtemp(prefix="online-crt-resume-")) / "global_step_5"
        checkpoint.mkdir()
        (checkpoint / "data.pt").write_text("cursor")
        (checkpoint / "online_faults.json").write_text("{}")
        resumed = subprocess.run(["bash", str(LAUNCHER)], env={"PATH": os.environ["PATH"],
            "REPO_ROOT": str(ROOT), "BASE": str(ROOT.parent), "NNODES": "2",
            "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "resume-test",
            "RESUME_FROM_PATH": str(checkpoint)}, capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        config = dict(line.split("=", 1) for line in resumed.stdout.splitlines())
        self.assertEqual(config["RESUME_MODE"], "resume_path")
        self.assertEqual(config["RESUME_FROM_PATH"], str(checkpoint))
        self.assertTrue(config["RL_OUTPUT_DIR"].endswith("_resume-test"))

    def test_trainer_checkpoint_writes_and_restores_controller_state(self):
        trainer_path = RL / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(trainer_path.read_text())
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name in {"_save_checkpoint", "_load_checkpoint"}]
        output = Path(tempfile.mkdtemp(prefix="online-crt-checkpoint-"))

        class Config(dict):
            __getattr__ = dict.__getitem__

        class TorchStub:
            @staticmethod
            def save(state, path):
                Path(path).write_text(json.dumps(state))

            @staticmethod
            def load(path, **kwargs):
                return json.loads(Path(path).read_text())

        namespace = {"os": os, "json": json, "torch": TorchStub,
                     "BaseCheckpointManager": SimpleNamespace(local_mkdir=lambda path: Path(path).mkdir(parents=True)),
                     "find_latest_ckpt_path": lambda path: None}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(trainer_path), "exec"), namespace)
        config = Config(default_local_dir=str(output), default_hdfs_dir=None, resume_mode="disable",
                        del_local_ckpt_after_load=False)
        controller = online_faults.OnlineFaultController(str(ROOT / "configs/online_crt_v1.json"))
        controller._bucket("visual-agent-fsc147", {})
        controller.step = 4
        loaded = []
        runner = SimpleNamespace(config=SimpleNamespace(trainer=config), global_steps=5,
            actor_rollout_wg=SimpleNamespace(save_checkpoint=lambda *args, **kwargs: None,
                load_checkpoint=lambda *args, **kwargs: None), use_critic=False,
            train_dataloader=SimpleNamespace(state_dict=lambda: {"cursor": 5},
                load_state_dict=lambda state: loaded.append(state)), online_fault_controller=controller)
        namespace["_save_checkpoint"](runner)
        checkpoint = output / "global_step_5"
        self.assertEqual(json.loads((checkpoint / "online_faults.json").read_text()), controller.state_dict())
        config["resume_mode"] = "resume_path"
        config["resume_from_path"] = str(checkpoint)
        runner.online_fault_controller = online_faults.OnlineFaultController(str(ROOT / "configs/online_crt_v1.json"))
        runner.global_steps = 0
        namespace["_load_checkpoint"](runner)
        self.assertEqual(runner.global_steps, 5)
        self.assertEqual(loaded, [{"cursor": 5}])
        self.assertEqual(runner.online_fault_controller.state_dict(), controller.state_dict())


if __name__ == "__main__":
    unittest.main()
