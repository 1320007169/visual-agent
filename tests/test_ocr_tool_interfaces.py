import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

import jsonschema
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from visual_tool_server import ToolService, ToolServerError
from visual_tools import get_visual_tool_schemas


class Bridge:
    def __init__(self, regions=(), error=False):
        self.result = {"status": "error" if error else "success",
                       "structured": {"results": list(regions)}}
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result, images=[])


class OCRInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), "white")

    def test_find_literal_text_after_fifty_unrelated_regions(self):
        regions = [{"text": str(i), "bbox": [0, 0, 20, 10]} for i in range(60)]
        regions += [{"text": "ＯＰＥＮ  NOW", "bbox": [20, 10, 100, 30]},
                    {"text": "Open now", "bbox": [100, 50, 180, 80]}]
        bridge = Bridge(regions)
        service = ToolService(None, None, ocr_bridge=bridge)
        result, images = service.execute("text_locate", {"target_image": 0, "query": "open\nnow"}, [self.image])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["regions"][0]["bbox_2d"], [100, 100, 500, 300])
        self.assertEqual(result["regions"][1]["bbox_2d"], [500, 500, 900, 800])
        self.assertEqual(result["match_scope"], "text_region")
        self.assertEqual(images, [])

    def test_contains_returns_whole_region_and_exact_does_not_match_substring(self):
        service = ToolService(None, None, ocr_bridge=Bridge([
            {"text": "营业时间 9:00", "bbox": [20, 10, 100, 30]},
            {"text": "营业时间", "bbox": None},
        ]))
        arguments = {"target_image": 0, "query": "营业时间"}
        exact, _ = service.execute("text_locate", arguments, [self.image])
        self.assertEqual(exact["regions"], [])
        contains, _ = service.execute("text_locate", dict(arguments, match_mode="contains"), [self.image])
        self.assertEqual(contains["regions"], [{"text": "营业时间 9:00", "bbox_2d": [100, 100, 500, 300]}])

    def test_roi_coordinates_refer_to_selected_image(self):
        for name in ("ocr_read", "text_locate"):
            with self.subTest(tool=name):
                bridge = Bridge([{"text": "OPEN", "bbox": [[10, 5], [30, 5], [30, 15], [10, 15]]}])
                service = ToolService(None, None, ocr_bridge=bridge)
                images = [Image.new("RGB", (40, 40)), self.image]
                result, _ = service.execute(name, {
                    "target_image": 1, "query": "OPEN", "bbox_2d": [250, 200, 750, 800],
                }, images)
                self.assertEqual(result["regions"][0]["bbox_2d"], [300, 250, 400, 350])
                self.assertEqual(result["target_image"], 1)
                self.assertEqual(bridge.calls[0]["images"][1].size, (100, 60))
                self.assertEqual(bridge.calls[0]["arguments"]["image_id"], 1)
                self.assertIs(bridge.calls[0]["images"][0], images[0])
                self.assertEqual(images[1].size, (200, 100))

    def test_invalid_inputs_fail_before_backend_and_errors_are_not_empty_matches(self):
        bridge = Bridge(error=True)
        service = ToolService(None, None, ocr_bridge=bridge)
        for extra in ({"query": " "}, {"query": []}, {"match_mode": "fuzzy"},
                      {"match_mode": []}, {"bbox_2d": [50, 50, 20, 20]}, {"target_image": 2}):
            with self.subTest(extra=extra), self.assertRaises(ToolServerError):
                service.execute("text_locate", {"query": "OPEN", "target_image": 0, **extra}, [self.image])
        self.assertEqual(bridge.calls, [])
        result, _ = service.execute("text_locate", {"query": "OPEN", "target_image": 0}, [self.image])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tool"], "text_locate")

    def test_new_schemas_can_be_selected_without_changing_old_schemas(self):
        names = {"grounding_detect", "crop_zoom", "depth_measure", "object_count", "text_detect", "text_recognize"}
        selected = get_visual_tool_schemas(names | {"text_locate", "ocr_read"})
        self.assertEqual(len(selected), 8)
        self.assertEqual(get_visual_tool_schemas(names), [s for s in selected if s["function"]["name"] in names])
        self.assertNotIn("text_locate", {s["function"]["name"] for s in get_visual_tool_schemas()})
        for schema in selected:
            jsonschema.Draft7Validator.check_schema(schema["function"]["parameters"])
        locate = next(s["function"]["parameters"] for s in selected if s["function"]["name"] == "text_locate")
        jsonschema.validate({"query": "OPEN", "target_image": 0}, locate)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"target_image": 0}, locate)


class Detector:
    def __init__(self):
        self.calls = 0
        self.boxes = [[2, 3, 20, 13], [30, 20, 50, 30]]

    def predict(self, path):
        self.calls += 1
        yield {"dt_polys": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in self.boxes],
               "dt_scores": [0.9] * len(self.boxes)}


class Recognizer:
    def __init__(self):
        self.calls = 0
        self.shapes = []

    def predict(self, input, batch_size):
        self.calls += 1
        self.shapes = [crop.shape for crop in input]
        return iter({"rec_text": f"word{i}", "rec_score": 0.95} for i in range(len(input)))


class PaddleOCRInterfaceTest(unittest.TestCase):
    def setUp(self):
        fake = ModuleType("paddlex")
        self.detector, self.recognizer = Detector(), Recognizer()
        fake.create_model = lambda **kw: self.detector if kw["model_name"].endswith("det") else self.recognizer
        spec = importlib.util.spec_from_file_location("test_paddleocr_backend", SCRIPTS / "paddleocr_split_server.py")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get("paddlex")
        sys.modules["paddlex"] = fake
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                sys.modules.pop("paddlex", None)
            else:
                sys.modules["paddlex"] = previous
        self.app = module.create_app(Path("unused"), cache_size=2)
        self.execute = next(route.endpoint for route in self.app.routes if route.path == "/execute")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "image.png"
        Image.new("RGB", (100, 50), "white").save(self.path)

    def call(self, name="ocr_read", path=None, **arguments):
        return self.execute({"tool": name, "args": {"image_id": 0, **arguments},
                             "context": {"images": [{"path": str(path or self.path)}]}})

    def test_combined_read_and_content_cache_across_temporary_paths(self):
        first = self.call()
        regions = first["structured"]["results"]
        self.assertEqual([r["text"] for r in regions], ["word0", "word1"])
        self.assertEqual(regions[0]["bbox"], [2, 3, 20, 13])
        self.assertEqual(self.recognizer.shapes, [(10, 18, 3), (10, 20, 3)])
        copy = self.path.with_name("copy.png")
        shutil.copyfile(self.path, copy)
        self.assertEqual(self.call(path=copy), first)
        self.assertEqual(self.detector.calls, 1)
        self.assertEqual(self.recognizer.calls, 1)
        # Filtering a cached result must not delete the cache's raw regions.
        self.assertEqual(self.call(minimum_confidence=0.99)["structured"]["results"], [])
        self.assertEqual(self.call(), first)
        Image.new("RGB", (100, 50), "red").save(copy)
        self.call(path=copy)
        self.assertEqual(self.detector.calls, 2)

    def test_cache_is_bounded(self):
        for color in ("white", "red", "blue", "white"):
            Image.new("RGB", (100, 50), color).save(self.path)
            self.call()
        self.assertEqual(self.detector.calls, 4)

    def test_empty_detection_skips_recognizer_and_degenerate_boxes_are_ignored(self):
        self.detector.boxes = [[20, 20, 20, 30], [-20, 0, -10, 10], [100, 0, 110, 10]]
        self.assertEqual(self.call()["structured"]["results"], [])
        self.assertEqual(self.recognizer.calls, 0)

    def test_combined_read_keeps_more_than_fifty_regions(self):
        self.detector.boxes = [[0, 0, 10, 10]] * 65
        self.assertEqual(len(self.call()["structured"]["results"]), 65)

    def test_legacy_detection_and_recognition_remain_independent(self):
        detected = self.call("text_detect")
        self.assertNotIn("text", detected["structured"]["results"][0])
        self.assertEqual(self.recognizer.calls, 0)
        read = self.call("text_recognize", bboxes=[[2, 3, 20, 13]])
        self.assertEqual(read["structured"]["results"][0]["text"], "word0")
        self.assertEqual(self.detector.calls, 1)
        self.assertEqual(self.recognizer.calls, 1)


if __name__ == "__main__":
    unittest.main()
