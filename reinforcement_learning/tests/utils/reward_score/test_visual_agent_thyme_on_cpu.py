import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "verl/utils/reward_score/visual_agent_thyme.py"
SPEC = importlib.util.spec_from_file_location("visual_agent_thyme", MODULE_PATH)
visual_agent_thyme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visual_agent_thyme)


def test_strict_answer_format_accepts_chat_terminators():
    assert visual_agent_thyme.has_strict_answer_format("<answer>left of</answer><|im_end|>")
    assert visual_agent_thyme.has_strict_answer_format(
        "<answer>left of</answer>\n<|im_end|><|endoftext|>\n"
    )


def test_strict_answer_format_accepts_final_tool_assistant_turn():
    text = """assistant
<tool_call>{"name":"grounding_detect","arguments":{}}</tool_call>
user
<tool_response>{"boxes":[]}</tool_response>
assistant
<answer>below</answer><|im_end|>
"""
    assert visual_agent_thyme.has_strict_answer_format(text)


def test_strict_answer_format_rejects_non_protocol_suffixes():
    assert not visual_agent_thyme.has_strict_answer_format(
        "<answer>left of</answer> additional explanation<|im_end|>"
    )
    assert not visual_agent_thyme.has_strict_answer_format(
        "<answer>left</answer><answer>right</answer><|im_end|>"
    )
