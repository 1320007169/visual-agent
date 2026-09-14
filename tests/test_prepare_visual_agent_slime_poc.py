import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_visual_agent_slime_poc.py"
SPEC = importlib.util.spec_from_file_location("prepare_visual_agent_slime_poc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_convert_row_builds_chat_and_exact_image_placeholders():
    converted = MODULE.convert_row(
        {
            "question": "Where is the cup?",
            "solution": "RIGHT OF",
            "images": ["/data/a.jpg", "/data/b.jpg"],
            "bbox": [1, 2, 3, 4],
            "source_index": 17,
        },
        "Use tools.",
        0,
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "<image>\n<image>\nWhere is the cup?"},
    ]
    assert converted["images"] == ["/data/a.jpg", "/data/b.jpg"]
    assert converted["solution"] == "right of"
    assert converted["metadata"]["source_index"] == 17


def test_convert_row_rejects_missing_multimodal_input():
    with pytest.raises(ValueError, match="no images"):
        MODULE.convert_row({"question": "q", "solution": "above", "images": []}, "prompt", 3)
