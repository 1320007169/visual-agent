import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from visual_agent_inference import (  # noqa: E402
    HTTPVisualToolExecutor,
    InferenceError,
    ToolExecutionResult,
    ToolInvocation,
    VisualAgent,
    load_training_tool_schemas,
    parse_tool_invocations,
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
    def test_malformed_tool_call_ends_like_rl_rollout(self):
        model = FakeModelClient()
        model.responses[1]["content"] = "<tool_call>{bad-json}</tool_call>"
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            result = VisualAgent(model, tool_executor=FakeToolExecutor()).run([image.name], "Read formula")
        # The malformed turn is the final response; scoring marks it wrong instead of an API failure.
        self.assertEqual(result.response, "<tool_call>{bad-json}</tool_call>")
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.tool_calls[0]["name"], "grounding_detect")

    def test_parse_skips_text_calls_the_rl_rollout_skips(self):
        invocations = parse_tool_invocations({"content": (
            '<tool_call>{bad-json}</tool_call>'
            '<tool_call>["not", "an", "object"]</tool_call>'
            '<tool_call>{"name":"crop_zoom","arguments":[1]}</tool_call>'
            '<tool_call>{"name":"crop_zoom","arguments":{"target_image":0}}</tool_call>'
        )})
        self.assertEqual(invocations, [ToolInvocation("crop_zoom", {"target_image": 0})])

    def test_parse_xml_tool_call(self):
        [invocation] = parse_tool_invocations(
            {
                "content": '<tool_call>{"name":"sam3_crop_zoom","arguments":{"query":"sign","target_image":0,"slack_ratio":0.1}}</tool_call>'
            }
        )
        self.assertEqual(invocation.name, "sam3_crop_zoom")
        self.assertEqual(invocation.arguments["target_image"], 0)

    def test_parse_qwen_attribute_tool_call(self):
        [invocation] = parse_tool_invocations(
            {
                "content": '<tool_call function="grounding_detect" arguments="{&quot;query&quot;:&quot;person&quot;,&quot;target_image&quot;:0}"></tool_call>'
            }
        )
        self.assertEqual(invocation.name, "grounding_detect")
        self.assertEqual(invocation.arguments, {"query": "person", "target_image": 0})

    def test_every_text_call_is_answered_in_order(self):
        model = FakeModelClient()
        model.responses[0]["content"] = (
            '<tool_call>{"name":"grounding_detect","arguments":{"query":"car","target_image":0}}</tool_call>'
            '<tool_call>{"name":"grounding_detect","arguments":{"query":"bus","target_image":0}}</tool_call>'
        )
        executor = FakeToolExecutor()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            result = VisualAgent(model, tool_executor=executor).run([image.name], "Count")
        self.assertEqual([call[0].arguments["query"] for call in executor.calls], ["car", "bus"])
        self.assertEqual([message["role"] for message in result.messages[2:]], ["assistant", "user", "user", "assistant"])

    def test_native_bad_arguments_and_unknown_tools_get_error_observations(self):
        calls = [
            {"id": "call_0", "type": "function", "function": {"name": "grounding_detect", "arguments": "{bad"}},
            {"id": "call_1", "type": "function", "function": {"name": "grounding_detect", "arguments": "[1]"}},
            {"id": "call_2", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}},
        ]
        model = FakeModelClient()
        model.responses[0] = {"role": "assistant", "content": None, "tool_calls": calls}
        executor = FakeToolExecutor()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            result = VisualAgent(model, tool_executor=executor, use_native_tools=True,
                                 allowed_tool_names={"grounding_detect"}).run([image.name], "Count")
        self.assertEqual(result.response, "<answer>2</answer>")
        self.assertEqual(executor.calls, [])
        observations = result.messages[3:6]
        self.assertEqual([message["tool_call_id"] for message in observations], ["call_0", "call_1", "call_2"])
        self.assertTrue(all(message["role"] == "tool" for message in observations))
        self.assertIn("JSONDecodeError", observations[0]["content"])
        self.assertIn("must be a JSON object", observations[1]["content"])
        self.assertIn("not allowed", observations[2]["content"])

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

    def test_native_tools_send_given_schemas_and_use_tool_role_history(self):
        schemas = [{"type": "function", "function": {"name": "grounding_detect", "description": "d",
                    "parameters": {"type": "object", "properties": {}, "required": []}}}]
        call = {"id": "call_0", "type": "function", "function": {
            "name": "grounding_detect", "arguments": '{"query":"car","target_image":0}'}}

        class NativeModelClient:
            def __init__(self):
                self.tools = []
                self.responses = [{"role": "assistant", "content": "Look.", "tool_calls": [call]},
                                  {"role": "assistant", "content": "<answer>2</answer>"}]

            def chat(self, messages, **kwargs):
                self.tools.append(kwargs["tools"])
                return self.responses.pop(0)

        client = NativeModelClient()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"not-a-real-jpeg-but-valid-for-transport")
            image.flush()
            result = VisualAgent(client, tool_executor=FakeToolExecutor(), use_native_tools=True,
                                 allowed_tool_names={"grounding_detect"}, tool_schemas=schemas,
                                 ).run([image.name], "Count cars")

        self.assertEqual(result.response, "<answer>2</answer>")
        self.assertEqual(client.tools, [schemas, schemas])
        self.assertEqual(result.messages[2], {"role": "assistant", "content": "Look.", "tool_calls": [call]})
        observation = result.messages[3]
        self.assertEqual((observation["role"], observation["tool_call_id"]), ("tool", "call_0"))
        self.assertNotIn("<tool_response>", observation["content"])
        self.assertIn('"labels": ["car", "truck"]', observation["content"])

    def test_training_tool_schemas_match_rl_loader_byte_for_byte(self):
        root = SCRIPTS_DIR.parent
        sys.path.insert(0, str(root / "reinforcement_learning"))
        from verl.tools.schemas import load_tool_schemas_from_config

        config = root / "reinforcement_learning/examples/sglang_multiturn/config/tool_config/visual_tool_multitool_vlocr_config.yaml"
        schemas = load_training_tool_schemas(config)
        # Key order matters: it changes the rendered <tools> block text.
        self.assertEqual(json.dumps(schemas), json.dumps(load_tool_schemas_from_config(str(config))))
        self.assertEqual([schema["function"]["name"] for schema in schemas],
                         ["grounding_detect", "crop_zoom", "depth_measure", "object_count", "ocr_read"])

    def test_compact_observations_keep_full_trace_and_crop_image(self):
        box = [0, 0, 100, 100]
        crop_image = "data:image/jpeg;base64,eA=="
        cases = [
            ("ocr_read", {"target_image": 0, "mode": "chart"},
             {"text": "A | 10", "truncated": False, "mode": "chart", "bbox_2d": box,
              "target_image": 0, "source": "paddleocr_vl"},
             {"text": "A | 10", "truncated": False}, []),
            ("chart_parse", {"target_image": 0},
             {"text": "A | 10", "truncated": True, "source": "paddleocr_vl_chart", "target_image": 0},
             {"text": "A | 10", "truncated": True}, []),
            ("crop_zoom", {"bbox_2d": box, "target_image": 0},
             {"crop_zoom": {"target_image": 1, "crop_path": "tool://crop.jpg"}, "source": "crop_zoom"},
             {"target_image": 1}, [crop_image]),
            ("depth_measure", {"bboxes_2d": [box], "target_image": 0},
             {"bbox_2d": box, "depth_m": 1.5, "coordinate_space": "relative_0_1000"},
             {"regions": [{"bbox_2d": box, "depth_m": 1.5}]}, []),
            ("object_count", {"query": "cars", "target_image": 0},
             {"count": 2, "boxes": [box], "points_2d": [[50, 50]], "query": "cars", "source": "object_count"},
             {"count": 2}, []),
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

    def test_raw_ocr_backslashes_keep_newline_and_quote_escapes(self):
        text = '\\neq \\frac{a}{b}\n"中文"\t'
        output = {"text": text, "truncated": False, "source": "paddleocr_vl"}
        for raw in ("0", "1"):
            for call_id in (None, "ocr_0"):
                with self.subTest(raw=raw, native=call_id is not None):
                    executor = FakeToolExecutor()
                    executor.execute = lambda invocation, images: ToolExecutionResult(output=output)
                    with patch.dict(os.environ, {"VISUAL_AGENT_OCR_RAW_BACKSLASH": raw}):
                        agent = VisualAgent(FakeModelClient(), tool_executor=executor)
                    trace = []
                    observation = agent._execute_tool(
                        ToolInvocation("ocr_read", {"target_image": 0}, call_id), [], trace,
                    )
                    content = observation["content"]
                    if call_id is None:
                        content = content.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
                    if raw == "1":
                        self.assertEqual(content, r'{"text": "\neq \frac{a}{b}\n\"中文\"\t", "truncated": false}')
                        self.assertIn(r'\neq', content)
                        self.assertNotIn(r'\\neq', content)
                    else:
                        self.assertEqual(json.loads(content), {"text": text, "truncated": False})
                        self.assertIn(r'\\frac', content)
                    self.assertIn(r'\n\"中文\"\t', content)
                    self.assertNotIn('\n', content)
                    self.assertNotIn('\t', content)
                    self.assertEqual(trace[0]["result"], output)
                    self.assertEqual(json.loads(json.dumps(observation)), observation)

    def test_raw_ocr_observation_does_not_change_other_tools_or_errors(self):
        for name, output in [
            ("chart_parse", {"text": r"\\frac{a}{b}", "truncated": False}),
            ("ocr_read", {"status": "error", "text": r"bad \\input"}),
        ]:
            with self.subTest(tool=name):
                executor = FakeToolExecutor()
                executor.execute = lambda invocation, images: ToolExecutionResult(output=output)
                with patch.dict(os.environ, {"VISUAL_AGENT_OCR_RAW_BACKSLASH": "1"}):
                    agent = VisualAgent(FakeModelClient(), tool_executor=executor)
                observation = agent._execute_tool(ToolInvocation(name, {}, "call_0"), [], [])
                self.assertEqual(json.loads(observation["content"]), output)

    def test_fault_injection_replays_same_parameters_without_executing_again(self):
        outputs = [
            {"status": "error", "message": "no target"},
            {"count": 7, "query": "cars", "source": "object_count"},
            {"count": 7, "query": "cars", "source": "object_count"},
        ]
        executor = FakeToolExecutor()
        executor.execute = lambda invocation, images: ToolExecutionResult(output=outputs.pop(0))
        with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "object_count"}):
            agent = VisualAgent(FakeModelClient(), tool_executor=executor)
        trace = []
        observed = [
            json.loads(agent._execute_tool(ToolInvocation("object_count", {}, "call_0"), [], trace)["content"])
            for _ in range(3)
        ]
        self.assertEqual(observed[0], {"status": "error", "message": "no target"})
        self.assertNotEqual(observed[1]["count"], 7)
        self.assertEqual(observed[2], observed[1])
        self.assertNotIn("fault", trace[0])
        self.assertEqual(trace[1]["fault"], {"original": {"count": 7}, "injected": observed[1]})
        self.assertEqual(trace[1]["result"]["count"], 7)
        self.assertEqual(trace[2]["fault"], trace[1]["fault"])
        self.assertTrue(trace[2]["replayed_fault"])
        self.assertEqual(len(outputs), 1)

    def test_fault_replay_leaves_changed_parameters_unfaulted(self):
        executor = FakeToolExecutor()
        executor.execute = lambda invocation, images: ToolExecutionResult(output={"count": 7})
        with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "object_count"}):
            agent = VisualAgent(FakeModelClient(), tool_executor=executor)
        trace = []
        agent._execute_tool(ToolInvocation("object_count", {"target_image": 0}, "first"), [], trace)
        response = agent._execute_tool(ToolInvocation("object_count", {"target_image": 1}, "next"), [], trace)
        self.assertEqual(json.loads(response["content"]), {"count": 7})
        self.assertNotIn("fault", trace[-1])

    def test_confusable_replay_preserves_serialization_and_new_call_id(self):
        executor = FakeToolExecutor()
        executor.execute = lambda invocation, images: ToolExecutionResult(output={"text": r"\(0\)"})
        with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "ocr_read",
                                   "VISUAL_AGENT_FAULT_TYPE": "ocr_confusable", "VISUAL_AGENT_OCR_RAW_BACKSLASH": "1"}):
            agent = VisualAgent(FakeModelClient(), tool_executor=executor)
        trace = []
        first = agent._execute_tool(ToolInvocation("ocr_read", {"target_image": 0}, "first"), [], trace)
        executor.execute = lambda invocation, images: self.fail("Replay must not execute the tool")
        replay = agent._execute_tool(ToolInvocation("ocr_read", {"target_image": 0}, "second"), [], trace)
        self.assertEqual(first["content"], replay["content"])
        self.assertEqual(replay["tool_call_id"], "second")
        self.assertIn("O", replay["content"])

    def test_ocr_replay_normalizes_default_mode_and_full_image_bbox(self):
        executor = FakeToolExecutor()
        executor.execute = lambda invocation, images: ToolExecutionResult(output={"text": "0"})
        with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "ocr_read",
                                   "VISUAL_AGENT_FAULT_TYPE": "ocr_confusable"}):
            agent = VisualAgent(FakeModelClient(), tool_executor=executor)
        trace = []
        agent._execute_tool(ToolInvocation("ocr_read", {"target_image": 0, "mode": "text"}, "first"), [], trace)
        executor.execute = lambda invocation, images: self.fail("Equivalent arguments must replay the fault")
        for arguments in [{"target_image": 0}, {"target_image": 0, "bbox_2d": [0, 0, 1000, 1000]}]:
            observed = agent._execute_tool(ToolInvocation("ocr_read", arguments, "again"), [], trace)
            self.assertEqual(json.loads(observed["content"]), {"text": "O"})
            self.assertTrue(trace[-1]["replayed_fault"])

    def test_ocr_replay_follows_full_crop_aliases_but_allows_real_regions(self):
        executor = FakeToolExecutor()
        executor.execute = lambda invocation, images: ToolExecutionResult(output={"text": "0"})
        with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "ocr_read",
                                   "VISUAL_AGENT_FAULT_TYPE": "ocr_confusable"}):
            agent = VisualAgent(FakeModelClient(), tool_executor=executor)
        images, trace = ["original", "original"], []
        first = agent._execute_tool(ToolInvocation("ocr_read", {"target_image": 0}, "first"), images, trace)
        for target, box in ((1, [0, 0, 1000, 1000]), (2, [50, 50, 950, 950])):
            executor.execute = lambda invocation, images: ToolExecutionResult(
                output={"crop_zoom": {"target_image": len(images), "bbox_2d": [0, 0, 1000, 1000]}},
                images=[f"crop-{len(images)}"],
            )
            agent._execute_tool(ToolInvocation("crop_zoom", {"target_image": target, "bbox_2d": box}), images, trace)
        executor.execute = lambda invocation, images: self.fail("Image aliases must replay")
        for target in (1, 2, 3):
            replay = agent._execute_tool(ToolInvocation("ocr_read", {"target_image": target}, "again"), images, trace)
            self.assertEqual(first["content"], replay["content"])
        executor.execute = lambda invocation, images: ToolExecutionResult(output={"text": "real region"})
        response = agent._execute_tool(ToolInvocation("ocr_read", {"target_image": 3, "bbox_2d": [0, 0, 500, 500]}), images, trace)
        self.assertIn("real region", response["content"])

    def test_fault_injection_is_off_by_default_and_seeded_by_question(self):
        call = '<tool_call>{"name":"object_count","arguments":{"query":"cars","target_image":0}}</tool_call>'
        faults = []
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image-for-transport")
            for tools in ("", "object_count", "object_count"):
                model = FakeModelClient()
                model.responses[0]["content"] = call
                executor = FakeToolExecutor()
                executor.execute = lambda invocation, images: ToolExecutionResult(output={"count": 12})
                with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": tools}):
                    agent = VisualAgent(model, tool_executor=executor)
                faults.append(agent.run([str(image)], "How many cars?").tool_calls[0].get("fault"))
        self.assertIsNone(faults[0])
        self.assertEqual(faults[1], faults[2])
        self.assertNotEqual(faults[1]["injected"]["count"], 12)

    def test_fault_replay_is_reset_between_samples(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"image-for-transport")
            image.flush()
            executor = FakeToolExecutor()
            model = FakeModelClient()
            responses = [
                {"role": "assistant", "content": '<tool_call>{"name":"object_count","arguments":{}}</tool_call>'},
                {"role": "assistant", "content": "<answer>2</answer>"},
            ]
            model.responses = responses + responses
            with patch.dict(os.environ, {"VISUAL_AGENT_FAULT_TOOLS": "object_count"}):
                agent = VisualAgent(model, tool_executor=executor)
            agent.run([image.name], "First sample")
            agent.run([image.name], "Second sample")
            self.assertEqual(len(executor.calls), 2)

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
