"""Additional RL tools using the existing relative-coordinate image protocol."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


RELATIVE_BOX = {
    "type": "array",
    "items": {"type": "number", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}
RELATIVE_POINT = {**RELATIVE_BOX, "minItems": 2, "maxItems": 2}
TARGET_IMAGE = {"type": "integer", "minimum": 0, "description": "Zero-based input image index."}
FACTOR = {"type": "number", "minimum": 0, "maximum": 4}

EXTENSION_PARAMETERS = {
    "image_resize": {
        "target_image": TARGET_IMAGE,
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "max_side": {"type": "integer", "minimum": 1},
    },
    "image_enhance": {
        "target_image": TARGET_IMAGE,
        "contrast": FACTOR,
        "sharpness": FACTOR,
        "brightness": FACTOR,
        "gamma": {"type": "number", "exclusiveMinimum": 0, "maximum": 4},
        "denoise": {"type": "boolean"},
    },
    "image_rotate": {
        "target_image": TARGET_IMAGE,
        "angle": {"type": "number", "minimum": -360, "maximum": 360},
        "expand": {"type": "boolean"},
    },
    "image_flip": {
        "target_image": TARGET_IMAGE,
        "direction": {"type": "string", "enum": ["horizontal", "vertical", "both"]},
    },
    "image_draw": {
        "target_image": TARGET_IMAGE,
        "boxes": {"type": "array", "items": RELATIVE_BOX},
        "points": {"type": "array", "items": RELATIVE_POINT},
        "lines": {"type": "array", "items": RELATIVE_BOX},
        "color": {"type": "string", "description": "Pillow color name or hex color."},
        "width": {"type": "integer", "minimum": 1, "maximum": 32},
    },
    "sam_segment": {"target_image": TARGET_IMAGE, "bbox_2d": RELATIVE_BOX},
    "bbox_geometry": {
        "target_image": TARGET_IMAGE,
        "bbox_a": RELATIVE_BOX,
        "bbox_b": RELATIVE_BOX,
    },
}
EXTENSION_REQUIRED = {
    "image_resize": ["target_image"],
    "image_enhance": ["target_image"],
    "image_rotate": ["target_image", "angle"],
    "image_flip": ["target_image", "direction"],
    "image_draw": ["target_image"],
    "sam_segment": ["target_image", "bbox_2d"],
    "bbox_geometry": ["target_image", "bbox_a", "bbox_b"],
}
EXTENSION_DESCRIPTIONS = {
    "image_resize": "Resize using width and height, or max_side preserving aspect ratio. Returns a new image index.",
    "image_enhance": "Adjust contrast, sharpness, brightness, gamma or denoise. Factors default to 1. Returns a new image index.",
    "image_rotate": "Rotate counterclockwise by angle in degrees. expand defaults to true. Returns a new image index.",
    "image_flip": "Flip horizontally, vertically or both. Returns a new image index.",
    "image_draw": "Draw boxes, points or lines in relative 0-1000 coordinates. width is in pixels. Returns a new image index.",
    "sam_segment": "Segment one relative 0-1000 box with SAM3. Returns mask and overlay image indices, score and area.",
    "bbox_geometry": "Compare two relative 0-1000 boxes on the same image. Returns image-plane relations, IoU and center distance in pixels; does not determine depth.",
}
EXTENSION_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": EXTENSION_DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": EXTENSION_REQUIRED[name],
                "additionalProperties": False,
            },
        },
    }
    for name, properties in EXTENSION_PARAMETERS.items()
]
EXTENSION_TOOL_NAMES = set(EXTENSION_PARAMETERS)


def validate_extension_arguments(name: str, arguments: dict[str, Any], images: list[Image.Image]) -> int:
    missing = set(EXTENSION_REQUIRED[name]) - set(arguments)
    unknown = set(arguments) - set(EXTENSION_PARAMETERS[name])
    if missing or unknown:
        raise ValueError(f"{name}: missing arguments {sorted(missing)}, unknown arguments {sorted(unknown)}")
    target = arguments["target_image"]
    if type(target) is not int or not 0 <= target < len(images):
        raise ValueError("target_image must identify an existing image")
    for key in ("expand", "denoise"):
        if key in arguments and type(arguments[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    return target


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _relative_values(value: Any, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"relative coordinates must contain {length} numbers")
    return [_number(item, "coordinate", 0, 1000) for item in value]


def relative_box_to_pixels(value: Any, size: tuple[int, int]) -> list[float]:
    box = _relative_values(value, 4)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("bbox must have positive area")
    return [number * size[index % 2] / 1000 for index, number in enumerate(box)]


def execute_extension(
    name: str, arguments: dict[str, Any], images: list[Image.Image],
) -> tuple[dict, list[Image.Image]]:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

    target = validate_extension_arguments(name, arguments, images)
    source = images[target]
    if name == "bbox_geometry":
        a = relative_box_to_pixels(arguments["bbox_a"], source.size)
        b = relative_box_to_pixels(arguments["bbox_b"], source.size)
        center_a = [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]
        center_b = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return {
            "target_image": target,
            "x_relation": "left_of" if center_a[0] < center_b[0] else "right_of" if center_a[0] > center_b[0] else "aligned",
            "y_relation": "above" if center_a[1] < center_b[1] else "below" if center_a[1] > center_b[1] else "aligned",
            "iou": round(intersection / (area_a + area_b - intersection), 6),
            "overlap": intersection > 0,
            "center_distance_px": round(math.dist(center_a, center_b), 6),
        }, []

    output = source.convert("RGB")
    if name == "image_resize":
        dimensions = {key: arguments[key] for key in ("width", "height", "max_side") if key in arguments}
        if set(dimensions) not in ({"width", "height"}, {"max_side"}):
            raise ValueError("provide width and height, or max_side")
        if any(type(value) is not int or value < 1 for value in dimensions.values()):
            raise ValueError("resize dimensions must be positive integers")
        if "max_side" in dimensions:
            scale = dimensions["max_side"] / max(source.size)
            size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        else:
            size = (dimensions["width"], dimensions["height"])
        output = output.resize(size, Image.Resampling.LANCZOS)
    elif name == "image_enhance":
        for key, enhancer in (("contrast", ImageEnhance.Contrast), ("sharpness", ImageEnhance.Sharpness), ("brightness", ImageEnhance.Brightness)):
            factor = _number(arguments.get(key, 1), key, 0, 4)
            output = enhancer(output).enhance(factor)
        gamma = _number(arguments.get("gamma", 1), "gamma", 0, 4)
        if gamma == 0:
            raise ValueError("gamma must be positive")
        if gamma != 1:
            table = [round(255 * ((value / 255) ** (1 / gamma))) for value in range(256)]
            output = output.point(table * 3)
        if arguments.get("denoise", False):
            output = output.filter(ImageFilter.MedianFilter(size=3))
    elif name == "image_rotate":
        angle = _number(arguments["angle"], "angle", -360, 360)
        output = output.rotate(angle, resample=Image.Resampling.BICUBIC, expand=arguments.get("expand", True))
    elif name == "image_flip":
        direction = arguments["direction"]
        if direction not in {"horizontal", "vertical", "both"}:
            raise ValueError("direction must be horizontal, vertical, or both")
        if direction in {"horizontal", "both"}:
            output = ImageOps.mirror(output)
        if direction in {"vertical", "both"}:
            output = ImageOps.flip(output)
    elif name == "image_draw":
        boxes, points, lines = (arguments.get(key, []) for key in ("boxes", "points", "lines"))
        if any(not isinstance(value, list) for value in (boxes, points, lines)):
            raise ValueError("boxes, points and lines must be arrays")
        if not boxes and not points and not lines:
            raise ValueError("provide at least one box, point or line")
        color, width = arguments.get("color", "red"), arguments.get("width", 3)
        if not isinstance(color, str) or type(width) is not int or not 1 <= width <= 32:
            raise ValueError("color must be a string and width an integer in [1, 32]")
        draw = ImageDraw.Draw(output)
        for box in boxes:
            draw.rectangle(relative_box_to_pixels(box, output.size), outline=color, width=width)
        for point in points:
            x, y = _relative_values(point, 2)
            x, y = x * output.width / 1000, y * output.height / 1000
            radius = max(2, width * 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        for line in lines:
            coordinates = _relative_values(line, 4)
            pixels = [value * output.size[index % 2] / 1000 for index, value in enumerate(coordinates)]
            draw.line(pixels, fill=color, width=width)
    else:
        raise ValueError(f"{name} requires the segmentation service")
    return {
        "target_image": len(images),
        "source_image": target,
        "width": output.width,
        "height": output.height,
    }, [output]
