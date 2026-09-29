import ast
import asyncio
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import AsyncMock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "reinforcement_learning/verl"

# Load the real tools without importing the GPU/Ray-dependent verl package.
package = ModuleType("_multitool_test_tools")
package.__path__ = [str(VERL / "tools")]
with patch.dict(sys.modules, {package.__name__: package}):
    schemas = importlib.import_module(f"{package.__name__}.schemas")
    visual_tool = importlib.import_module(f"{package.__name__}.visual_tool")


def trainer_helpers():
    path = VERL / "trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    names = {"_resolve_training_steps", "_compute_visual_tool_metrics", "_visual_accuracy_macro_mean", "_prune_checkpoint_directories"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"Optional": Optional, "defaultdict": defaultdict, "os": os, "shutil": shutil}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


helpers = trainer_helpers()
tool_config = yaml.safe_load((
    ROOT / "reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_depth_count_config.yaml"
).read_text())


def tool_schema(name):
    row = next(row for row in tool_config["tools"] if row["tool_schema"]["function"]["name"] == name)
    return schemas.OpenAIFunctionToolSchema.model_validate(row["tool_schema"])


class MultitoolTrainingMetricsTest(unittest.TestCase):
    def test_launcher_defaults_and_explicit_overrides(self):
        script = ROOT / "scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"
        env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(ROOT), "BASE": str(ROOT.parent),
               "MULTITOOL_CONFIG_ONLY": "1", "TRAIN_RUN_TOKEN": "test_config"}
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True)
        self.assertIn("TOTAL_TRAINING_STEPS=null\n", result.stdout)
        self.assertIn("VAL_BEFORE_TRAIN=True\n", result.stdout)
        self.assertIn("BEST_METRIC=val-core/visual-agent/acc/macro_mean\n", result.stdout)
        env.update(TOTAL_TRAINING_STEPS="20", VAL_BEFORE_TRAIN="False", BEST_METRIC="custom_metric")
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True)
        self.assertIn("TOTAL_TRAINING_STEPS=20\n", result.stdout)
        self.assertIn("VAL_BEFORE_TRAIN=False\n", result.stdout)
        self.assertIn("BEST_METRIC=custom_metric\n", result.stdout)

    def test_training_steps_match_drop_last_batches(self):
        resolve = helpers["_resolve_training_steps"]
        self.assertEqual(resolve(30808 // 112, 1, None), 275)
        self.assertEqual(resolve(275, 1, 276), 275)
        self.assertEqual(resolve(275, 2, 276), 276)
        self.assertEqual(resolve(275, 1, 40), 40)
        for args in [(0, 1, None), (275, 0, None), (275, 1, 0)]:
            with self.assertRaises(ValueError):
                resolve(*args)

    def test_metrics_track_zero_count_usage_and_errors_by_task(self):
        tally, depth = "visual-agent-tallyqa", "visual-agent-depth-raw"
        metrics = helpers["_compute_visual_tool_metrics"](
            [tally, depth, tally],
            [{"tool_calls": [{"tool": "crop_zoom", "status": "success"}]},
             {"tool_calls": [{"tool": "depth_measure", "status": "error"}]},
             {"tool_calls": []}],
            [1, 0, 0],
        )
        self.assertEqual(metrics[f"train-tools/{tally}/object_count/trajectory_rate"], 0)
        self.assertEqual(metrics[f"train-tools/{tally}/acc_mean"], 0.5)
        self.assertEqual(metrics[f"train-tools/{depth}/error_trajectory_rate"], 1)
        self.assertEqual(metrics[f"train-tools/{depth}/depth_measure/errors"], 1)
        self.assertEqual(metrics[f"train-tools/{depth}/depth_measure/success_rate"], 0)

    def test_macro_accuracy_does_not_weight_by_dataset_size(self):
        macro = helpers["_visual_accuracy_macro_mean"]
        self.assertEqual(macro(["visual-agent-a"] * 8 + ["visual-agent-b"] * 2, [1] * 8 + [0] * 2), 0.5)
        self.assertIsNone(macro(["other"], [1]))
        with self.assertRaises(ValueError):
            macro(["visual-agent-a"], [])
        with self.assertRaises(ValueError):
            helpers["_compute_visual_tool_metrics"](["source"], [], [1])

    def test_schema_keeps_nested_bbox_constraints(self):
        schema = tool_schema("depth_measure").model_dump(exclude_none=True)
        boxes = schema["function"]["parameters"]["properties"]["bboxes_2d"]
        self.assertEqual(boxes["minItems"], 1)
        self.assertEqual(boxes["items"]["minItems"], 4)
        self.assertEqual(boxes["items"]["maxItems"], 4)
        self.assertEqual(boxes["items"]["items"]["maximum"], 1000)

    def test_checkpoint_rotation_preserves_current_step_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "global_step_40/actor/huggingface"
            current.mkdir(parents=True)
            (root / "global_step_80").mkdir()
            removed = helpers["_prune_checkpoint_directories"](directory, 1, protected_step=40)
            self.assertTrue(current.is_dir())
            self.assertEqual(removed, [str(root / "global_step_80")])


class OnlineVisualToolValidationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.statuses = []
        self.result = {"status": "success", "result": {
            "bboxes_2d": [[0, 0, 100, 100], [100, 100, 200, 200]],
            "depths_m": [1.0, 2.0],
        }}
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                test.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status = test.statuses.pop(0) if test.statuses else 200
                body = json.dumps(test.result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        with patch("builtins.print"):
            self.tool = visual_tool.OnlineVisualTool(
                {"base_url": f"http://127.0.0.1:{self.server.server_port}", "max_retries": 2},
                tool_schema("depth_measure"),
            )
        self.instance = await self.tool.create(images=["data:image/png;base64,AA=="])
        self.arguments = {"bboxes_2d": [[0, 0, 100, 100], [100, 100, 200, 200]], "target_image": 0}

    async def asyncTearDown(self):
        await self.tool.release(self.instance)
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        self.thread.join(timeout=5)

    async def test_invalid_boxes_and_image_index_never_reach_http(self):
        bad_arguments = [
            {"bboxes_2d": [[0, 0, 100], [100, 100, 200, 200]], "target_image": 0},
            {"bboxes_2d": [], "target_image": 0},
            {"bboxes_2d": [[0, 0, 1001, 100], [100, 100, 200, 200]], "target_image": 0},
            dict(self.arguments, target_image=1),
        ]
        for arguments in bad_arguments:
            with self.assertRaises(ValueError):
                await self.tool.execute(self.instance, arguments)
        self.assertEqual(self.requests, [])

    async def test_single_depth_box_returns_a_region(self):
        box = [0, 0, 100, 100]
        self.result = {"status": "success", "result": {
            "bbox_2d": box, "depth_m": 1.5,
            "target_image": 0, "coordinate_space": "relative_0_1000",
        }}
        arguments = {"bboxes_2d": [box], "target_image": 0}
        output, _, _ = await self.tool.execute(self.instance, arguments)
        self.assertEqual(json.loads(output), {"regions": [{"bbox_2d": box, "depth_m": 1.5}]})
        self.assertEqual(self.requests[0]["arguments"], arguments)

    async def test_http_422_is_not_retried(self):
        self.statuses = [422]
        with self.assertRaisesRegex(ValueError, "HTTP 422"):
            await self.tool.execute(self.instance, self.arguments)
        self.assertEqual(len(self.requests), 1)

    async def test_http_503_is_retried(self):
        self.statuses = [503, 200]
        with patch.object(visual_tool.asyncio, "sleep", AsyncMock()):
            output, _, _ = await self.tool.execute(self.instance, self.arguments)
        self.assertEqual(json.loads(output), {"regions": [
            {"bbox_2d": [0, 0, 100, 100], "depth_m": 1.0},
            {"bbox_2d": [100, 100, 200, 200], "depth_m": 2.0},
        ]})
        self.assertEqual(len(self.requests), 2)

    async def test_business_error_is_visible_without_retry(self):
        self.result = {"status": "success", "result": {"status": "error", "message": "No valid depth pixels", "recoverable": True}}
        output, _, metrics = await self.tool.execute(self.instance, self.arguments)
        self.assertEqual(json.loads(output)["status"], "error")
        self.assertEqual(metrics["tool_error"], "No valid depth pixels")
        self.assertEqual(len(self.requests), 1)

    async def test_callback_records_business_error_in_trace(self):
        path = VERL / "workers/rollout/chat_scheduler.py"
        tree = ast.parse(path.read_text())
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_call_tool")
        namespace = {"Any": object, "Dict": dict, "json": json, "time": importlib.import_module("time")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        self.result = {"status": "success", "result": {"status": "error", "message": "No valid depth pixels"}}
        info = {"images": ["data:image/png;base64,AA=="], "__trace__": {"model_calls": [], "tool_calls": []}}
        message = await namespace["_call_tool"](
            SimpleNamespace(tools={"depth_measure": self.tool}),
            {"name": "depth_measure", "arguments": self.arguments}, info, xml_mode=True,
        )
        self.assertIn("No valid depth pixels", message["content"])
        self.assertEqual(info["__trace__"]["tool_calls"][0]["status"], "error")
        self.assertEqual(info["__trace__"]["tool_calls"][0]["raw_result"], self.result["result"])

    async def test_callback_preserves_raw_results_and_compact_observations(self):
        path = VERL / "workers/rollout/chat_scheduler.py"
        tree = ast.parse(path.read_text())
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_call_tool")
        namespace = {"Any": object, "Dict": dict, "json": json, "time": importlib.import_module("time")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        boxes = [[0, 0, 100, 100], [100, 100, 200, 200]]
        crop_image = "data:image/jpeg;base64,eA=="
        cases = [
            ("grounding_detect", {"query": "number above entrance", "target_image": 0},
             {"boxes": boxes, "confidence": [0.8, 0.7], "labels": ["entrance", "number"],
              "query": "number above entrance", "source": "groundingdino", "count": 2},
             {"boxes": boxes, "confidence": [0.8, 0.7], "labels": ["entrance", "number"]}, []),
            ("crop_zoom", {"bbox_2d": boxes[0], "target_image": 0},
             {"crop_zoom": {"target_image": 1, "crop_path": "tool://crop.jpg"}, "source": "crop_zoom"},
             {"target_image": 1}, [crop_image]),
            ("depth_measure", self.arguments,
             {"bboxes_2d": boxes, "depths_m": [1.0, 2.0], "target_image": 0},
             {"regions": [{"bbox_2d": box, "depth_m": depth} for box, depth in zip(boxes, [1.0, 2.0])]}, []),
            ("object_count", {"query": "cars", "target_image": 0},
             {"count": 2, "boxes": boxes, "points_2d": [[50, 50], [150, 150]], "source": "object_count"},
             {"count": 2, "boxes": boxes}, []),
        ]
        for name, arguments, original, expected, images in cases:
            with self.subTest(tool=name), patch("builtins.print"):
                tool = visual_tool.OnlineVisualTool(
                    {"base_url": f"http://127.0.0.1:{self.server.server_port}"}, tool_schema(name),
                )
                self.result = {"status": "success", "result": original, "images": images}
                info = {"images": ["data:image/png;base64,AA=="], "__trace__": {"model_calls": [], "tool_calls": []}}
                message = await namespace["_call_tool"](
                    SimpleNamespace(tools={name: tool}), {"name": name, "arguments": arguments}, info, xml_mode=True,
                )
                trace = info["__trace__"]["tool_calls"][0]
                self.assertEqual(trace["status"], "success")
                self.assertEqual(json.loads(trace["model_observation"]), expected)
                self.assertEqual(trace["raw_result"], original)
                self.assertEqual(json.loads(json.dumps(trace))["raw_result"], original)
                self.assertEqual(trace["returned_image_count"], len(images))
                if images:
                    self.assertEqual(message["content"][1]["image_url"]["url"], crop_image)
                else:
                    self.assertEqual(message["content"], f'<tool_response>\n{trace["model_observation"]}\n</tool_response>')


if __name__ == "__main__":
    unittest.main()
