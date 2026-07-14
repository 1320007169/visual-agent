import sys
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from visual_tool_server import ToolService, decode_image, encode_image  # noqa: E402


class FakeSam3:
    def segment(self, image, query):
        return {
            "boxes": [[10.0, 20.0, 30.0, 50.0]],
            "confidence": [0.9],
            "mask_area_px": [321],
        }


class FakeGroundingDino:
    def detect(self, image, query):
        return {
            "boxes": [[1.0, 2.0, 20.0, 30.0]],
            "confidence": [0.8],
            "labels": [query],
        }


class VisualToolServerTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), "white")
        self.service = ToolService(FakeSam3(), FakeGroundingDino())

    def test_image_data_url_roundtrip(self):
        decoded = decode_image(encode_image(self.image))
        self.assertEqual(decoded.size, self.image.size)

    def test_grounding_contract(self):
        result, images = self.service.execute(
            "grounding_detect", {"query": "person", "target_image": 0}, [self.image]
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["source"], "groundingdino")
        self.assertEqual(images, [])

    def test_sam3_segment_contract(self):
        result, _ = self.service.execute(
            "sam3_segment_multi",
            {"queries": [{"role": "target", "query": "person"}], "target_image": 0},
            [self.image],
        )
        self.assertEqual(result["queries"][0]["mask_area_px"], [321])
        self.assertEqual(result["queries"][0]["count"], 1)

    def test_crop_returns_336_image(self):
        result, images = self.service.execute(
            "sam3_crop_zoom",
            {"query": "sign", "target_image": 0, "slack_ratio": 0.35},
            [self.image],
        )
        crop = decode_image(images[0])
        self.assertEqual(crop.size, (336, 336))
        self.assertEqual(result["crop_zoom"]["target_image"], 1)
        self.assertEqual(result["target"]["selected_box"], [10.0, 20.0, 30.0, 50.0])


if __name__ == "__main__":
    unittest.main()
