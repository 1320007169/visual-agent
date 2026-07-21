"""Visual tool names and schemas for offline SFT data.

These schemas make the visual tool family explicit without wiring any online
SAM3 or GroundingDINO executor into training.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


VISUAL_TOOL_NAMES = {
    "crop_zoom",
    "sam3_segment_multi",
    "sam3_crop_zoom",
    "sam3_crop_zoom_multi",
    "grounding_detect",
}


VISUAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "crop_zoom",
            "description": "Crop a region selected with Qwen3-VL relative coordinates without resampling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Relative [x1, y1, x2, y2] coordinates, each from 0 to 1000.",
                    },
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label for the selected region.",
                    },
                    "slack_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Optional extra context around the bounding box; defaults to 0.",
                    },
                },
                "required": ["bbox_2d", "target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sam3_segment_multi",
            "description": "Segment one or more target or anchor objects in an image using natural-language queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "Objects to segment. Each entry has role target or anchor and a natural-language query.",
                    },
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                },
                "required": ["queries", "target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grounding_detect",
            "description": "Detect boxes for an object in an image using a natural-language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Object or region to detect."},
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                },
                "required": ["query", "target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sam3_crop_zoom",
            "description": "Crop and zoom one target object or region from an image for closer visual inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Target object or region to crop."},
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "slack_ratio": {
                        "type": "number",
                        "description": "Extra context around the detected target box.",
                    },
                },
                "required": ["query", "target_image", "slack_ratio"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sam3_crop_zoom_multi",
            "description": "Crop and zoom multiple target objects or regions from an image for closer visual inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "Targets to crop. Each entry has role target and a natural-language query.",
                    },
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "slack_ratio": {
                        "type": "number",
                        "description": "Extra context around each detected target box.",
                    },
                },
                "required": ["queries", "target_image", "slack_ratio"],
            },
        },
    },
]


def get_visual_tool_schemas() -> list[dict[str, Any]]:
    return deepcopy(VISUAL_TOOL_SCHEMAS)
