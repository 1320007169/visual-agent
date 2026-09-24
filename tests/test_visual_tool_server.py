import sys
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from visual_tool_server import (  # noqa: E402
    ToolServerError,
    ToolService,
    _resolve_sam3_image_checkpoint,
    decode_image,
    encode_image,
)


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


class FakeBridge:
    def __init__(self, result, images=None):
        self.result = result
        self.images = images or []
        self.calls = []

    def execute(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result, images=self.images)


class VisualToolServerTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), "white")
        self.service = ToolService(FakeSam3(), FakeGroundingDino())

    def test_image_data_url_roundtrip(self):
        decoded = decode_image(encode_image(self.image))
        self.assertEqual(decoded.size, self.image.size)

    def test_sam31_video_checkpoint_is_rejected_for_image_tools(self):
        with self.assertRaisesRegex(ToolServerError, "video tracking"):
            _resolve_sam3_image_checkpoint("/tmp/sam3.1_multiplex.pt")

    def test_grounding_contract(self):
        result, images = self.service.execute(
            "grounding_detect", {"query": "person", "target_image": 0}, [self.image]
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["boxes"], [[5.0, 20.0, 100.0, 300.0]])
        self.assertEqual(result["coordinate_space"], "relative_0_1000")
        self.assertEqual(result["source"], "groundingdino")
        self.assertEqual(images, [])

    def test_long_grounding_query_returns_recoverable_error(self):
        query = "one two three four five six seven eight nine ten eleven twelve thirteen"
        with patch.dict("os.environ", {"GROUNDING_DINO_MAX_QUERY_WORDS": "12"}):
            result, images = self.service.execute(
                "grounding_detect", {"query": query, "target_image": 0}, [self.image]
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["word_count"], 13)
        self.assertEqual(result["max_words"], 12)
        self.assertEqual(images, [])

    def test_sam3_segment_contract(self):
        result, _ = self.service.execute(
            "sam3_segment_multi",
            {"queries": [{"role": "target", "query": "person"}], "target_image": 0},
            [self.image],
        )
        self.assertEqual(result["queries"][0]["mask_area_px"], [321])
        self.assertEqual(result["queries"][0]["boxes"], [[50.0, 200.0, 150.0, 500.0]])
        self.assertEqual(result["coordinate_space"], "relative_0_1000")
        self.assertEqual(result["queries"][0]["count"], 1)
        self.assertEqual(result["source"], "sam3")
        self.assertNotIn("selected_box", result["queries"][0])

    def test_crop_returns_336_image(self):
        result, images = self.service.execute(
            "sam3_crop_zoom",
            {"query": "sign", "target_image": 0, "slack_ratio": 0.35},
            [self.image],
        )
        crop = decode_image(images[0])
        self.assertEqual(crop.size, (336, 336))
        self.assertEqual(result["crop_zoom"]["target_image"], 1)
        self.assertEqual(result["target"]["selected_box"], [50.0, 200.0, 150.0, 500.0])
        self.assertEqual(result["crop_zoom"]["requested_bbox_2d"], [50.0, 200.0, 150.0, 500.0])
        self.assertEqual(result["crop_zoom"]["bbox_2d"], [0.0, 0.0, 480.0, 960.0])
        self.assertEqual(result["coordinate_space"], "relative_0_1000")
        self.assertNotIn("crop_zoom", result["target"])
        self.assertEqual(
            result["crop_zoom"]["text_summary"],
            "Localized the target and cropped the region for closer visual inspection.",
        )
        self.assertEqual(result["crop_zoom"]["crop_path"], "tool://images/1/request_crop_zoom.jpg")

    def test_crop_zoom_maps_qwen3_relative_bbox_to_pixels(self):
        result, images = ToolService(None, None).execute(
            "crop_zoom",
            {"bbox_2d": [50, 200, 150, 500], "target_image": 0, "label": "sign"},
            [self.image],
            instance_id="rollout-8",
        )

        crop = decode_image(images[0])
        self.assertEqual(crop.size, (96, 96))
        self.assertEqual(result["bbox_2d"], [50.0, 200.0, 150.0, 500.0])
        self.assertEqual(result["pixel_bbox_2d"], [10.0, 20.0, 30.0, 50.0])
        self.assertEqual(result["coordinate_space"], "relative_0_1000")
        self.assertEqual(result["crop_zoom"]["target_image"], 1)
        self.assertEqual(result["crop_zoom"]["requested_bbox_2d"], [50.0, 200.0, 150.0, 500.0])
        self.assertEqual(result["crop_zoom"]["slack_ratio"], 0.1)
        self.assertEqual(result["crop_zoom"]["coordinate_space"], "relative_0_1000")
        self.assertEqual(
            result["crop_zoom"]["text_summary"],
            "Cropped the specified image region at its native resolution.",
        )
        self.assertEqual(
            result["crop_zoom"]["crop_path"], "tool://images/1/rollout-8_crop_zoom.jpg"
        )
        self.assertEqual(result["source"], "crop_zoom")

    def test_crop_zoom_clamps_relative_bbox(self):
        result, _ = self.service.execute(
            "crop_zoom", {"bbox_2d": [-5, 100, 1050, 1100], "target_image": 0}, [self.image]
        )
        self.assertEqual(result["bbox_2d"], [0.0, 100.0, 1000.0, 1000.0])
        self.assertEqual(result["pixel_bbox_2d"], [0.0, 10.0, 200.0, 100.0])

    def test_crop_zoom_rejects_invalid_bbox(self):
        with self.assertRaisesRegex(ToolServerError, "valid relative region"):
            self.service.execute(
                "crop_zoom", {"bbox_2d": [300, 200, 100, 500], "target_image": 0}, [self.image]
            )

    def test_multi_crop_matches_sft_shape_and_image_indices(self):
        result, images = self.service.execute(
            "sam3_crop_zoom_multi",
            {
                "queries": [
                    {"role": "target", "query": "backpack"},
                    {"role": "anchor", "query": "apple"},
                ],
                "target_image": 0,
                "slack_ratio": 0.35,
            },
            [self.image],
            instance_id="rollout-7",
        )
        self.assertEqual(len(images), 2)
        self.assertEqual([item["target_image"] for item in result["image_outputs"]], [1, 2])
        self.assertEqual(result["queries"][0]["crop_zoom"]["target_image"], 1)
        self.assertEqual(result["queries"][1]["crop_zoom"]["target_image"], 2)
        self.assertEqual(result["queries"][1]["role"], "anchor")
        self.assertEqual(
            result["queries"][0]["crop_zoom"]["crop_path"],
            "tool://images/1/rollout-7_crop_zoom_t1.jpg",
        )

    def test_ocr_contract_is_compact_and_relative(self):
        bridge = FakeBridge(
            {
                "status": "success",
                "success": True,
                "structured": {
                    "text": "EXIT",
                    "results": [
                        {
                            "text": "EXIT",
                            "confidence": 0.96,
                            "bbox": [[20, 10], [100, 10], [100, 30], [20, 30]],
                        }
                    ],
                },
            },
            [encode_image(self.image)],
        )
        service = ToolService(None, None, ocr_bridge=bridge)
        result, images = service.execute(
            "ocr_read",
            {"target_image": 0},
            [self.image],
        )

        self.assertEqual(result["text"], "EXIT")
        self.assertEqual(result["regions"][0]["bbox_2d"], [100.0, 100.0, 500.0, 300.0])
        self.assertEqual(result["annotated_image"], 1)
        self.assertEqual(len(images), 1)
        self.assertEqual(bridge.calls[0]["arguments"], {"image_id": 0, "minimum_confidence": 0.0})

    def test_count_contract_converts_points_and_hides_backend_name(self):
        bridge = FakeBridge(
            {
                "status": "success",
                "success": True,
                "structured": {"count": 2, "points": [[20, 10], [100, 50]]},
            },
            [encode_image(self.image)],
        )
        service = ToolService(None, None, count_bridge=bridge)
        result, images = service.execute(
            "object_count",
            {"query": "apples", "target_image": 0},
            [self.image],
        )

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["points_2d"], [[100.0, 100.0], [500.0, 500.0]])
        self.assertEqual(result["source"], "object_count")
        self.assertNotIn("countgd", str(result).lower())
        self.assertEqual(len(images), 1)
        self.assertEqual(bridge.calls[0]["arguments"], {"image_id": 0, "query": "apples"})

    def test_count_contract_keeps_pseudo_exemplar_fields(self):
        bridge = FakeBridge(
            {
                "status": "success",
                "success": True,
                "structured": {
                    "count": 4,
                    "points": [[20, 10]],
                    "count_mode": "pseudo_exemplar",
                    "confidence": [0.91, 0.8],
                    "first_pass_count": 3,
                    "pseudo_exemplar_count": 3,
                    "pseudo_exemplar_boxes": [[1, 2, 3, 4]],
                },
            },
            [encode_image(self.image)],
        )
        service = ToolService(None, None, count_bridge=bridge)
        result, _ = service.execute(
            "object_count",
            {"query": "apples", "target_image": 0},
            [self.image],
        )

        self.assertEqual(result["count_mode"], "pseudo_exemplar")
        self.assertEqual(result["first_pass_count"], 3)
        self.assertEqual(result["pseudo_exemplar_count"], 3)
        self.assertEqual(result["pseudo_exemplar_boxes"], [[5.0, 20.0, 15.0, 40.0]])
        self.assertEqual(result["confidence"], [0.91, 0.8])

    def test_count_contract_omits_pseudo_fields_for_legacy_backend(self):
        bridge = FakeBridge(
            {"status": "success", "success": True, "structured": {"count": 1, "points": [[20, 10]]}},
        )
        service = ToolService(None, None, count_bridge=bridge)
        result, _ = service.execute(
            "object_count",
            {"query": "apples", "target_image": 0},
            [self.image],
        )

        self.assertNotIn("count_mode", result)
        self.assertNotIn("first_pass_count", result)
        self.assertNotIn("pseudo_exemplar_boxes", result)

    def test_depth_measure_contract_uses_one_bbox(self):
        bridge = FakeBridge(
            {
                "status": "success",
                "success": True,
                "structured": {
                    "statistics": {"metric_depth": True, "unit": "meter"},
                    "region_depths": [{"median_depth_m": 1.4, "mean_depth_m": 1.5}],
                },
                "provenance": {"metric_depth": True},
            },
            [encode_image(self.image)],
        )
        service = ToolService(None, None, depth_bridge=bridge)
        result, images = service.execute(
            "depth_measure",
            {"target_image": 0, "bbox_2d": [500, 100, 900, 800]},
            [self.image],
        )

        self.assertEqual(result["depth_m"], 1.4)
        self.assertEqual(result["bbox_2d"], [500.0, 100.0, 900.0, 800.0])
        self.assertEqual(result["depth_image"], 1)
        self.assertEqual(len(images), 1)
        self.assertEqual(
            bridge.calls[0]["arguments"]["bboxes"],
            [[100.0, 10.0, 180.0, 80.0]],
        )

    def test_ground_depth_contract_localizes_one_query(self):
        bridge = FakeBridge(
            {
                "status": "success",
                "success": True,
                "structured": {
                    "statistics": {"metric_depth": True},
                    "region_depths": [{"median_depth_m": 2.7}],
                },
                "provenance": {"metric_depth": True},
            },
            [encode_image(self.image)],
        )
        service = ToolService(None, FakeGroundingDino(), depth_bridge=bridge)
        result, _ = service.execute(
            "ground_depth", {"target_image": 0, "query": "person"}, [self.image]
        )

        self.assertEqual(result["query"], "person")
        self.assertEqual(result["depth_m"], 2.7)
        self.assertEqual(result["bbox_2d"], [5.0, 20.0, 100.0, 300.0])
        self.assertEqual(bridge.calls[0]["arguments"]["bboxes"], [[1.0, 2.0, 20.0, 30.0]])

    def test_remote_tool_failure_becomes_recoverable_observation(self):
        bridge = FakeBridge(
            {
                "status": "failed",
                "success": False,
                "error_code": "no_text",
                "error_message": "No readable text was found.",
            }
        )
        service = ToolService(None, None, ocr_bridge=bridge)
        result, images = service.execute("ocr_read", {"target_image": 0}, [self.image])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "no_text")
        self.assertTrue(result["recoverable"])
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()
