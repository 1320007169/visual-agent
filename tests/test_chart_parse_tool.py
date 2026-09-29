import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import nullcontext

import numpy as np

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from visual_tool_server import ToolService, ToolServerError
from visual_tools import get_visual_tool_schemas
from paddleocr_vl_chart_server import ChartRecognizer, create_app


class Bridge:
    def __init__(self, result=None):
        self.result = result or {"status": "success", "structured": {"text": "A | 10\nB | 20", "truncated": False}}
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result, images=[])


class ChartParseTest(unittest.TestCase):
    def test_fixed_chart_task_and_decode_only_generated_tokens(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return Batch(input_ids=np.array([[9, 8, 7]]))

            def batch_decode(self, generated, **kwargs):
                self.generated = generated.tolist()
                return [" A | 10 "]

        class Model:
            generation_config = SimpleNamespace(eos_token_id=2)

            def generate(self, **kwargs):
                self.options = kwargs
                return np.array([[9, 8, 7, 4, 2]])

        recognizer = ChartRecognizer.__new__(ChartRecognizer)
        recognizer.torch = SimpleNamespace(inference_mode=nullcontext)
        recognizer.device = "cpu"
        recognizer.max_new_tokens = 2
        recognizer.model = Model()
        recognizer.processor = Processor()
        result = recognizer.read(Image.new("RGB", (100, 60)))
        self.assertEqual(result, {"text": "A | 10", "truncated": False})
        self.assertEqual(recognizer.processor.generated, [[4, 2]])
        self.assertEqual(recognizer.processor.messages[0]["content"][1]["text"], "Chart Recognition:")
        self.assertFalse(recognizer.model.options["do_sample"])
        recognizer.model.generation_config.eos_token_id = [3]
        self.assertTrue(recognizer.read(Image.new("RGB", (100, 60)))["truncated"])

    def test_crop_only_selected_image_and_preserve_coordinate_reference(self):
        bridge = Bridge()
        images = [Image.new("RGB", (40, 40)), Image.new("RGB", (200, 100))]
        service = ToolService(None, None, chart_bridge=bridge)
        result, returned_images = service.execute(
            "chart_parse", {"target_image": 1, "bbox_2d": [250, 200, 750, 800]}, images,
        )
        self.assertEqual(bridge.calls[0]["images"][0].size, (100, 60))
        self.assertEqual(len(bridge.calls[0]["images"]), 1)
        self.assertEqual(bridge.calls[0]["arguments"], {"image_id": 0})
        self.assertEqual(result["bbox_2d"], [250, 200, 750, 800])
        self.assertEqual(result["target_image"], 1)
        self.assertEqual(result["text"], "A | 10\nB | 20")
        self.assertFalse(result["truncated"])
        self.assertEqual(returned_images, [])
        self.assertEqual(images[1].size, (200, 100))

    def test_whole_chart_and_truncation(self):
        bridge = Bridge({"status": "success", "structured": {"text": "partial", "truncated": True}})
        result, _ = ToolService(None, None, chart_bridge=bridge).execute(
            "chart_parse", {"target_image": 0}, [Image.new("RGB", (200, 100))],
        )
        self.assertTrue(result["truncated"])
        self.assertEqual(result["bbox_2d"], [0, 0, 1000, 1000])

    def test_invalid_boxes_or_question_rejected_before_backend(self):
        bridge = Bridge()
        service = ToolService(None, None, chart_bridge=bridge)
        for extra in ({"bbox_2d": [500, 0, 100, 100]}, {"query": "What is the maximum?"}, {"target_image": 3}):
            with self.subTest(extra=extra), self.assertRaises(ToolServerError):
                service.execute("chart_parse", {"target_image": 0, **extra}, [Image.new("RGB", (200, 100))])
        self.assertEqual(bridge.calls, [])

    def test_failures_are_not_successful_empty_parses(self):
        for body in ({"status": "failed", "error_code": "oom", "error_message": "Out of memory"},
                     {"status": "success", "structured": {"text": " "}}):
            result, _ = ToolService(None, None, chart_bridge=Bridge(body)).execute(
                "chart_parse", {"target_image": 0}, [Image.new("RGB", (200, 100))],
            )
            self.assertEqual(result["status"], "error")
        with self.assertRaisesRegex(ToolServerError, "VTS_CHART_ENDPOINT"):
            ToolService(None, None).execute("chart_parse", {"target_image": 0}, [])

    def test_schema_is_opt_in_and_has_no_question(self):
        schema = get_visual_tool_schemas({"chart_parse"})[0]["function"]
        self.assertEqual(schema["parameters"]["required"], ["target_image"])
        self.assertEqual(set(schema["parameters"]["properties"]), {"target_image", "bbox_2d"})
        self.assertNotIn("chart_parse", {row["function"]["name"] for row in get_visual_tool_schemas()})

    def test_chart_service_contract_and_path_validation(self):
        from fastapi import HTTPException

        class Recognizer:
            def read(self, image):
                self.size = image.size
                return {"text": "A | 10", "truncated": False}

        with tempfile.TemporaryDirectory() as directory:
            recognizer = Recognizer()
            app = create_app(recognizer, Path(directory))
            execute = next(route.endpoint for route in app.routes if route.path == "/execute")
            path = Path(directory) / "chart.png"
            Image.new("RGB", (100, 60)).save(path)
            payload = {"tool": "chart_parse", "args": {"image_id": 0},
                       "context": {"images": [{"path": str(path)}]}}
            result = execute(payload)
            self.assertEqual(result["structured"], {"text": "A | 10", "truncated": False})
            self.assertEqual(recognizer.size, (100, 60))
            for bad in ({"tool": "ocr_read"}, {"args": {"image_id": -1}},
                        {"args": {"image_id": 0, "query": "answer this"}},
                        {"context": {"images": [{"path": "/etc/passwd"}]}}):
                with self.subTest(bad=bad), self.assertRaises(HTTPException):
                    execute({**payload, **bad})


if __name__ == "__main__":
    unittest.main()
