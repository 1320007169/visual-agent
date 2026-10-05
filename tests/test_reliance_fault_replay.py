import ast
import asyncio
import json
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


SCHEDULER = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/workers/rollout/chat_scheduler.py"


def load_call_tool():
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_call_tool")
    namespace = {"Any": object, "Dict": dict, "json": json, "time": time, "logger": logging.getLogger(__name__)}
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
