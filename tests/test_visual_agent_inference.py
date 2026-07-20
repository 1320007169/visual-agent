import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from visual_agent_inference import (  # noqa: E402
    HTTPVisualToolExecutor,
    InferenceError,
    ToolExecutionResult,
    ToolInvocation,
    VisualAgent,
    parse_tool_invocation,
)


class FakeModelClient:
    def __init__(self):
        self.responses = [
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"grounding_detect","arguments":{"query":"car","target_image":0}}</tool_call>',
            },
            {"role": "assistant", "content": "<answer>2</answer>"},
        ]

    def chat(self, messages, **kwargs):
        return self.responses.pop(0)


class FakeToolExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, invocation, images):
        self.calls.append((invocation, images))
        return ToolExecutionResult(output={"count": 2})


class FailingToolExecutor:
    def execute(self, invocation, images):
        raise InferenceError("target was not localized")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.request = None

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return FakeResponse({"status": "success", "result": {"boxes": []}})


class VisualAgentInferenceTest(unittest.TestCase):
    def test_parse_xml_tool_call(self):
        invocation = parse_tool_invocation(
            {
                "content": '<tool_call>{"name":"sam3_crop_zoom","arguments":{"query":"sign","target_image":0,"slack_ratio":0.1}}</tool_call>'
            }
        )
        self.assertEqual(invocation.name, "sam3_crop_zoom")
        self.assertEqual(invocation.arguments["target_image"], 0)

    def test_parse_qwen_attribute_tool_call(self):
        invocation = parse_tool_invocation(
            {
                "content": '<tool_call function="grounding_detect" arguments="{&quot;query&quot;:&quot;person&quot;,&quot;target_image&quot;:0}"></tool_call>'
            }
        )
        self.assertEqual(invocation.name, "grounding_detect")
        self.assertEqual(invocation.arguments, {"query": "person", "target_image": 0})

    def test_xml_multiturn_agent(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"not-a-real-jpeg-but-valid-for-transport")
            image.flush()
            executor = FakeToolExecutor()
            result = VisualAgent(FakeModelClient(), tool_executor=executor).run([image.name], "Count cars")

        self.assertEqual(result.response, "<answer>2</answer>")
        self.assertEqual(len(executor.calls), 1)
        self.assertIn("<tool_response>", result.messages[-2]["content"])

    def test_agent_recovers_from_tool_error(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"not-a-real-jpeg-but-valid-for-transport")
            image.flush()
            result = VisualAgent(
                FakeModelClient(), tool_executor=FailingToolExecutor()
            ).run([image.name], "Count cars")

        self.assertEqual(result.response, "<answer>2</answer>")
        self.assertIn('"status": "error"', result.messages[-2]["content"])
        self.assertEqual(result.tool_calls[0]["error"], "target was not localized")

    def test_http_executor_contract(self):
        session = FakeSession()
        executor = HTTPVisualToolExecutor("http://tools:9000", session=session)
        executor.execute(ToolInvocation("grounding_detect", {"query": "car", "target_image": 0}), ["data:image/jpeg;base64,eA=="])

        url, kwargs = session.request
        self.assertEqual(url, "http://tools:9000/execute")
        self.assertEqual(kwargs["json"]["name"], "grounding_detect")
        self.assertTrue(kwargs["json"]["instance_id"])
        self.assertEqual(len(kwargs["json"]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
