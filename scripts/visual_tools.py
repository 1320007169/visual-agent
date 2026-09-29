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
    "chart_parse",
    "text_locate",
    "text_detect",
    "text_recognize",
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
            "name": "chart_parse",
            "description": "Parse a chart with PaddleOCR-VL and return its data as text. Select the whole chart, including axes and legend. This tool does not answer questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {"type": "integer", "minimum": 0, "description": "Zero-based image index."},
                    "bbox_2d": {
                        "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4, "maxItems": 4,
                        "description": "Optional chart area [x1,y1,x2,y2]; omit for a whole-image chart.",
                    },
                },
                "required": ["target_image"],
                "additionalProperties": False,
            },
        },
    },
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
            "description": "Detect and read text in the whole image or optional bbox_2d. Return text and text-region boxes in relative 0-1000 coordinates of the selected image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "bbox_2d": {
                        "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4, "maxItems": 4,
                        "description": "Optional reading area [x1,y1,x2,y2] on target_image; omit to read the whole image.",
                    },
                },
                "required": ["target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_locate",
            "description": "Find literal text in an image using OCR and return matching text-region boxes. Exact matching ignores case and normalizes whitespace; contains matches within a region. Boxes cover the whole recognized region, not individual characters. Semantic descriptions and cross-region phrases are not supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {"type": "integer", "description": "Zero-based image index."},
                    "query": {"type": "string", "minLength": 1, "description": "Literal text to find, e.g. Corn or 营业时间; not a question or region description."},
                    "match_mode": {"type": "string", "enum": ["exact", "contains"], "default": "exact"},
                    "bbox_2d": {
                        "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4, "maxItems": 4,
                        "description": "Optional search area [x1,y1,x2,y2] on target_image.",
                    },
                },
                "required": ["query", "target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "depth_measure",
            "description": "Return regions with bbox_2d and median depth_m for one or more located boxes in the same image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {
                        "type": "integer",
                        "description": "Zero-based index into the sample images array.",
                    },
                    "bboxes_2d": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["bboxes_2d", "target_image"],
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
    {
        "type": "function",
        "function": {
            "name": "text_detect",
            "description": "Locate text without reading it; return regions with bbox_2d in relative 0-1000 coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {"type": "integer", "description": "Zero-based image index."},
                },
                "required": ["target_image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_recognize",
            "description": "Read selected relative 0-1000 boxes without running detection; return regions with bbox_2d and text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_image": {"type": "integer", "description": "Zero-based image index."},
                    "bboxes_2d": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["bboxes_2d", "target_image"],
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
