import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visual_tool_extensions import EXTENSION_TOOL_NAMES, EXTENSION_TOOL_SCHEMAS
from visual_tool_server import SUPPORTED_TOOLS, ToolServerError, ToolService, create_app, decode_image, encode_image
from visual_tools import DEFAULT_VISUAL_TOOL_NAMES, get_visual_tool_schemas


class SegmentBridge:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class VisualToolExtensionsTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), "white")
        self.image.putpixel((0, 0), (0, 0, 0))
        self.service = ToolService(None, None)

    def test_resize_chains_into_derived_image(self):
        result, returned = self.service.execute("image_resize", {"target_image": 0, "max_side": 100}, [self.image])
        self.assertEqual(result, {"target_image": 1, "source_image": 0, "width": 100, "height": 50})
        resized = decode_image(returned[0])
        result, returned = self.service.execute("image_rotate", {"target_image": 1, "angle": 90}, [self.image, resized])
        self.assertEqual(result["target_image"], 2)
        self.assertEqual(result["source_image"], 1)
        self.assertEqual(decode_image(returned[0]).size, (50, 100))
        result, returned = self.service.execute("image_resize", {"target_image": 0, "width": 60, "height": 80}, [self.image])
        self.assertEqual(decode_image(returned[0]).size, (60, 80))

    def test_flip_and_enhance_preserve_input_pixels(self):
        for direction, position in (("horizontal", (199, 0)), ("vertical", (0, 99)), ("both", (199, 99))):
            with self.subTest(direction=direction):
                _, returned = self.service.execute("image_flip", {"target_image": 0, "direction": direction}, [self.image])
                self.assertEqual(decode_image(returned[0]).getpixel(position), (0, 0, 0))
        _, returned = self.service.execute("image_enhance", {"target_image": 0, "brightness": 0}, [self.image])
        self.assertEqual(decode_image(returned[0]).getextrema(), ((0, 0), (0, 0), (0, 0)))
        self.assertEqual(self.image.getpixel((50, 50)), (255, 255, 255))
        _, returned = self.service.execute("image_enhance", {"target_image": 0, "gamma": 2, "denoise": True}, [self.image])
        self.assertEqual(decode_image(returned[0]).getpixel((0, 0)), (255, 255, 255))

    def test_draw_converts_each_axis_using_image_dimensions(self):
        _, returned = self.service.execute("image_draw", {
            "target_image": 0, "boxes": [[100, 100, 300, 400]],
            "points": [[750, 500]], "lines": [[500, 900, 500, 100]], "width": 1,
        }, [self.image])
        output = decode_image(returned[0])
        for position in ((20, 10), (60, 40), (150, 50), (100, 90), (100, 10)):
            self.assertEqual(output.getpixel(position), (255, 0, 0))
        self.assertEqual(self.image.getpixel((20, 10)), (255, 255, 255))

    def test_geometry_uses_pixels_for_distance_on_rectangular_image(self):
        result, returned = self.service.execute("bbox_geometry", {
            "target_image": 0, "bbox_a": [0, 0, 500, 500], "bbox_b": [250, 250, 750, 750],
        }, [self.image])
        self.assertEqual(returned, [])
        self.assertEqual(result["x_relation"], "left_of")
        self.assertEqual(result["y_relation"], "above")
        self.assertTrue(result["overlap"])
        self.assertEqual(result["iou"], round(1 / 7, 6))
        self.assertEqual(result["center_distance_px"], round(math.hypot(50, 25), 6))

    def test_segment_converts_coordinates_and_appends_two_images(self):
        mask = encode_image(Image.new("L", self.image.size, 255), image_format="PNG")
        overlay = encode_image(self.image, image_format="PNG")
        bridge = SegmentBridge(SimpleNamespace(result={
            "status": "success", "structured": {"score": 0.8, "area_pixels": 600, "area_ratio": 0.03},
        }, images=[mask, overlay]))
        service = ToolService(None, None, segment_bridge=bridge)
        result, returned = service.execute("sam_segment", {
            "target_image": 1, "bbox_2d": [100, 200, 300, 500],
        }, [Image.new("RGB", (400, 400)), self.image], instance_id="segment-7")
        self.assertEqual(bridge.calls[0]["arguments"], {"image_id": 1, "bbox": [20, 20, 60, 50]})
        self.assertEqual(bridge.calls[0]["instance_id"], "segment-7")
        self.assertEqual((result["mask_image"], result["overlay_image"]), (2, 3))
        self.assertEqual(result["target_image"], 1)
        self.assertEqual(result["area_ratio"], 0.03)
        self.assertEqual(returned, [mask, overlay])
        bridge.response = SimpleNamespace(result={"status": "failed", "error_code": "empty_mask", "error_message": "empty"}, images=[])
        result, returned = service.execute("sam_segment", {"target_image": 0, "bbox_2d": [0, 0, 1000, 1000]}, [self.image])
        self.assertEqual(result["status"], "error")
        self.assertEqual(returned, [])

    def test_invalid_arguments_and_missing_segment_service_are_reported(self):
        cases = [
            ("image_resize", {"target_image": 0}),
            ("image_resize", {"target_image": 0, "max_side": 0}),
            ("image_resize", {"target_image": 0, "width": 50, "height": 50, "max_side": 50}),
            ("image_resize", {"target_image": 0, "max_side": 100, "image_id": 0}),
            ("image_rotate", {"target_image": 1, "angle": 90}),
            ("image_rotate", {"target_image": True, "angle": 90}),
            ("image_rotate", {"target_image": 0, "angle": float("nan")}),
            ("image_rotate", {"target_image": 0, "angle": 90, "expand": "false"}),
            ("image_enhance", {"target_image": 0, "gamma": 0}),
            ("image_flip", {"target_image": 0, "direction": "diagonal"}),
            ("image_draw", {"target_image": 0}),
            ("image_draw", {"target_image": 0, "points": [[1001, 0]]}),
            ("bbox_geometry", {"target_image": 0, "bbox_a": [500, 0, 0, 500], "bbox_b": [0, 0, 1000, 1000]}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ToolServerError):
                self.service.execute(name, arguments, [self.image])
        with self.assertRaisesRegex(ToolServerError, "VTS_SEGMENT_ENDPOINT"):
            self.service.execute("sam_segment", {"target_image": 0, "bbox_2d": [0, 0, 1000, 1000]}, [self.image])

    def test_http_contract_returns_image_urls_and_rejects_invalid_arguments(self):
        with TestClient(create_app(self.service)) as client:
            response = client.post("/execute", json={
                "name": "image_resize", "arguments": {"target_image": 0, "max_side": 100},
                "images": [encode_image(self.image)], "instance_id": "http-7",
            })
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "success")
            self.assertEqual(body["result"]["target_image"], 1)
            self.assertEqual(decode_image(body["images"][0]).size, (100, 50))
            response = client.post("/execute", json={
                "name": "image_rotate", "arguments": {"target_image": 0, "angle": 361},
                "images": [encode_image(self.image)],
            })
            self.assertEqual(response.status_code, 422)

    def test_full_config_preserves_five_tools_and_registers_all_extensions(self):
        directory = REPO_ROOT / "reinforcement_learning/examples/sglang_multiturn/config/tool_config"
        original = yaml.safe_load((directory / "visual_tool_multitool_vlocr_config.yaml").read_text())
        full = yaml.safe_load((directory / "visual_tool_multitool_full_config.yaml").read_text())
        self.assertEqual(full["tools"][:5], original["tools"])
        self.assertEqual(len(full["tools"]), 12)
        self.assertEqual([tool["tool_schema"] for tool in full["tools"][5:]], EXTENSION_TOOL_SCHEMAS)
        names = {tool["tool_schema"]["function"]["name"] for tool in full["tools"]}
        self.assertTrue(names <= SUPPORTED_TOOLS)
        self.assertEqual(get_visual_tool_schemas(EXTENSION_TOOL_NAMES), EXTENSION_TOOL_SCHEMAS)
        self.assertEqual(len(get_visual_tool_schemas()), len(DEFAULT_VISUAL_TOOL_NAMES))

    def test_new_tool_calls_receive_format_credit_in_rl(self):
        path = REPO_ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py"
        spec = importlib.util.spec_from_file_location("extension_reward_test", path)
        reward = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward)
        for name in EXTENSION_TOOL_NAMES:
            call = json.dumps({"name": name, "arguments": {"target_image": 0}})
            transcript = (
                f"<tool_call>{call}</tool_call><|im_end|>\n"
                "<|im_start|>user\n<tool_response>{}</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<answer>4</answer><|im_end|>"
            )
            with self.subTest(name=name), patch.dict("os.environ", {"VISUAL_AGENT_FORMAT_PROTOCOL": "reason_act"}):
                result = reward.compute_score(transcript, "4", {"data_source": "visual-agent-tallyqa"})
                self.assertEqual(result["format"], 1.0)
                self.assertEqual(result["score"], 1.0)
                self.assertFalse(reward.has_strict_answer_format(transcript.replace(name, "unknown_tool")))


if __name__ == "__main__":
    unittest.main()
