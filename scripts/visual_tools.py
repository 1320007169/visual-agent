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
    "ocr_read",
    "depth_measure",
    "ground_depth",
    "object_count",
}

DEFAULT_VISUAL_TOOL_NAMES = {
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
                        "default": 0.1,
                        "description": "Extra context around the bounding box; defaults to 0.1 (10%).",
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
            "description": "Segment one or more objects and return boxes as relative 0-1000 coordinates.",
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
            "description": "Detect an object and return boxes as relative 0-1000 coordinates.",
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
            "description": "Localize and crop one target; returned boxes use relative 0-1000 coordinates.",
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
                        "default": 0.1,
                        "description": "Extra context around the detected target box; defaults to 0.1 (10%).",
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
            "description": "Localize and crop multiple targets; returned boxes use relative 0-1000 coordinates.",
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
                        "default": 0.1,
                        "description": "Extra context around each detected target box; defaults to 0.1 (10%).",
                    },
                },
                "required": ["queries", "target_image", "slack_ratio"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocr_read",
            "description": "Read visible text and return text regions in relative 0-1000 coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                },
                "required": ["target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "depth_measure",
            "description": "Return the median metric depth inside one relative 0-1000 bbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Relative xyxy region whose median depth should be measured.",
                    },
                },
                "required": ["bbox_2d", "target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ground_depth",
            "description": "Ground one concrete object and return its median metric depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short concrete object noun phrase."},
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
            "name": "object_count",
            "description": "Count instances of a concrete object and return spatial evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concrete object category to count."},
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                },
                "required": ["query", "target_image"],
            },
        },
    },
]


def get_visual_tool_schemas(tool_names: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict[str, Any]]:
    if tool_names is None:
        tool_names = DEFAULT_VISUAL_TOOL_NAMES
    requested = set(tool_names)
    unknown = requested - VISUAL_TOOL_NAMES
    if unknown:
        raise ValueError(f"Unknown visual tools: {sorted(unknown)}")
    return deepcopy([
        schema
        for schema in VISUAL_TOOL_SCHEMAS
        if schema["function"]["name"] in requested
    ])
