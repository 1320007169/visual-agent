import ast
import asyncio
import json
import importlib.util
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


SCHEDULER = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/workers/rollout/chat_scheduler.py"
SPEC = importlib.util.spec_from_file_location("replay_faults", SCHEDULER.parents[2] / "tools/tool_faults.py")
tool_faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_faults)


def load_call_tool():
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_call_tool")
    namespace = {"Any": object, "Dict": dict, "json": json, "time": time, "logger": logging.getLogger(__name__),
                 "tool_faults": tool_faults}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCHEDULER), "exec"), namespace)
    return namespace["_call_tool"]


class FakeCountTool:
    def __init__(self):
        self.executed = []

    async def create(self, images):
        return "instance"

    async def execute(self, instance_id, arguments):
        self.executed.append(arguments)
        return json.dumps({"count": 7}), 0.0, {}

    async def release(self, instance_id):
        return None


class RelianceFaultReplayTest(unittest.TestCase):
    def test_ocr_replays_after_equivalent_parameters_and_full_image_crops(self):
        tool = FakeCountTool()
        crop_count = 0

        async def crop(instance, arguments):
            nonlocal crop_count
            crop_count += 1
            result = {"crop_zoom": {"target_image": crop_count, "bbox_2d": [0, 0, 1000, 1000]}}
            return json.dumps({"target_image": crop_count}), 0.0, {
                "returned_images": [f"crop-{crop_count}"], "raw_result": result,
            }

        crop_tool = FakeCountTool()
        crop_tool.execute = crop
        fault = {"tool": "ocr_read", "arguments": {"target_image": 0, "mode": "text"},
                 "observation": '{"text": "wrong"}'}
        info = {"images": ["original"], "reliance_fault": fault,
                "__trace__": {"model_calls": [], "tool_calls": []}}
        callback = SimpleNamespace(tools={"ocr_read": tool, "crop_zoom": crop_tool})
        call = load_call_tool()
        for target in (0, 1):
            asyncio.run(call(callback, {"name": "crop_zoom", "arguments": {
                "target_image": target, "bbox_2d": [50, 50, 950, 950]}}, info, xml_mode=True))
        for arguments in ({"target_image": 0}, {"target_image": 2, "bbox_2d": [0, 0, 1000, 1000]}):
            message = asyncio.run(call(callback, {"name": "ocr_read", "arguments": arguments}, info, xml_mode=True))
            self.assertIn('"wrong"', message["content"])
            self.assertTrue(info["__trace__"]["tool_calls"][-1]["replayed_fault"])
        self.assertEqual(tool.executed, [])
        asyncio.run(call(callback, {"name": "ocr_read", "arguments": {
            "target_image": 2, "bbox_2d": [0, 0, 500, 500]}}, info, xml_mode=True))
        self.assertEqual(len(tool.executed), 1)

    def run_call(self, arguments, fault):
        tool = FakeCountTool()
        info = {"images": [], "__trace__": {"model_calls": [], "tool_calls": []}, "reliance_fault": fault}
        message = asyncio.run(load_call_tool()(
            SimpleNamespace(tools={"object_count": tool}),
            {"name": "object_count", "arguments": arguments}, info, xml_mode=True,
        ))
        return message["content"], info["__trace__"]["tool_calls"][0], tool.executed

    def test_repeating_the_faulted_call_returns_the_same_fault(self):
        arguments = {"query": "apples", "target_image": 0}
        fault = {"tool": "object_count", "arguments": dict(arguments), "observation": '{"count": 3}'}
        content, trace, executed = self.run_call(arguments, fault)
        self.assertEqual(content, '<tool_response>\n{"count": 3}\n</tool_response>')
        self.assertEqual((trace["status"], trace["replayed_fault"], trace["model_observation"]), ("success", True, '{"count": 3}'))
        self.assertEqual(executed, [])

    def test_other_calls_reach_the_real_tool(self):
        fault = {"tool": "object_count", "arguments": {"query": "apples", "target_image": 0}, "observation": '{"count": 3}'}
        for fault_spec in (fault, None):
            with self.subTest(fault=fault_spec is not None):
                arguments = {"query": "red apples", "target_image": 0}
                content, trace, executed = self.run_call(arguments, fault_spec)
                self.assertEqual(content, '<tool_response>\n{"count": 7}\n</tool_response>')
                self.assertNotIn("replayed_fault", trace)
                self.assertEqual(executed, [arguments])


if __name__ == "__main__":
    unittest.main()
