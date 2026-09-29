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
        return ToolExecutionResult(output={
            "query": "car", "boxes": [[0, 0, 100, 100], [200, 200, 300, 300]],
            "confidence": [0.9, 0.8], "count": 2, "source": "groundingdino",
            "labels": ["car", "truck"],
        })


class FailingToolExecutor:
    def execute(self, invocation, images):
        raise InferenceError("target was not localized")


class RepeatingToolModelClient:
    def chat(self, messages, **kwargs):
        return {
            "role": "assistant",
            "content": '<tool_call>{"name":"grounding_detect","arguments":{"query":"car","target_image":0}}</tool_call>',
        }


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
    def test_failure_trace_preserves_prior_tools_and_malformed_assistant(self):
        model = FakeModelClient()
        model.responses[1]["content"] = "<tool_call>{bad-json}</tool_call>"
        trace = {}
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            with self.assertRaisesRegex(InferenceError, "invalid <tool_call> JSON"):
                VisualAgent(model, tool_executor=FakeToolExecutor()).run(
                    [image.name], "Read formula", trace_sink=trace,
                )
        self.assertEqual(trace["turn"], 2)
        self.assertEqual(trace["tool_calls"][0]["name"], "grounding_detect")
        self.assertEqual(trace["messages"][-1]["content"], "<tool_call>{bad-json}</tool_call>")
        self.assertEqual(trace["last_assistant"]["content"], "<tool_call>{bad-json}</tool_call>")

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
        self.assertNotIn('"source"', result.messages[-2]["content"])
        self.assertNotIn('"count"', result.messages[-2]["content"])
        self.assertIn('"labels": ["car", "truck"]', result.messages[-2]["content"])
        self.assertEqual(result.tool_calls[0]["result"]["count"], 2)
        self.assertEqual(result.tool_calls[0]["result"]["source"], "groundingdino")

    def test_compact_observations_keep_full_trace_and_crop_image(self):
        box = [0, 0, 100, 100]
        crop_image = "data:image/jpeg;base64,eA=="
        cases = [
            ("crop_zoom", {"bbox_2d": box, "target_image": 0},
             {"crop_zoom": {"target_image": 1, "crop_path": "tool://crop.jpg"}, "source": "crop_zoom"},
             {"target_image": 1}, [crop_image]),
            ("depth_measure", {"bboxes_2d": [box], "target_image": 0},
             {"bbox_2d": box, "depth_m": 1.5, "coordinate_space": "relative_0_1000"},
             {"regions": [{"bbox_2d": box, "depth_m": 1.5}]}, []),
            ("object_count", {"query": "cars", "target_image": 0},
             {"count": 2, "boxes": [box], "points_2d": [[50, 50]], "query": "cars", "source": "object_count"},
             {"count": 2, "boxes": [box]}, []),
            ("text_detect", {"target_image": 0},
             {"regions": [{"bbox_2d": box, "confidence": 0.9}], "count": 1},
             {"regions": [{"bbox_2d": box}]}, []),
            ("text_recognize", {"bboxes_2d": [box], "target_image": 0},
             {"regions": [{"bbox_2d": box, "text": "OPEN", "confidence": 0.9}], "count": 1},
             {"regions": [{"bbox_2d": box, "text": "OPEN"}]}, []),
            ("depth_measure", {"bboxes_2d": [box], "target_image": 0},
             {"status": "error", "message": "No valid depth pixels", "recoverable": True},
             {"status": "error", "message": "No valid depth pixels", "recoverable": True}, []),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            for name, arguments, original, expected, returned_images in cases:
                with self.subTest(tool=name, output=original):
                    model = FakeModelClient()
                    model.responses[0]["content"] = "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
                    executor = FakeToolExecutor()
                    executor.execute = lambda invocation, images: ToolExecutionResult(output=original, images=returned_images)
                    result = VisualAgent(model, tool_executor=executor, allowed_tool_names={name}).run([image.name], "Inspect image")
                    content = result.messages[-2]["content"]
                    text = content[0]["text"] if isinstance(content, list) else content
                    observed = json.loads(text.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>"))
                    self.assertEqual(observed, expected)
                    self.assertEqual(result.tool_calls[0]["result"], original)
                    if returned_images:
                        self.assertEqual(content[1]["image_url"]["url"], crop_image)

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

    def test_agent_returns_result_after_max_turns(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"not-a-real-jpeg-but-valid-for-transport")
            image.flush()
            result = VisualAgent(
                RepeatingToolModelClient(),
                tool_executor=FakeToolExecutor(),
                max_turns=2,
            ).run([image.name], "Count cars")

        self.assertEqual(result.response, "Agent exceeded the maximum of 2 turns")
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(result.tool_calls), 2)

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
