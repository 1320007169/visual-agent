"""Counterfactual first-decision branches for multi-turn visual-agent rollouts.

A decision is the first action of the first assistant turn: one tool name, or
"direct" for answering without a tool. Forced branches prefill the canonical
decision prefix and let the policy continue on-policy from there.
"""

import json
import re
from typing import Any, Dict, List
from uuid import uuid4

import torch

DIRECT = "direct"
OTHER = "other"
# The RL prompts require the first turn to be either a tool call or <answer>...</answer>.
# "<answer" stops before ">" so the prefix ends on a tokenizer pre-token boundary.
DIRECT_PREFIX = "<answer"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
HERMES_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
_ARGS_MARKER = "__DECISION_BRANCH_ARGS__"


def decision_prefixes(tokenizer, tool_names: List[str]) -> Dict[str, str]:
    """Render each decision prefix exactly as the chat template renders it in training.

    Tool prefixes stop after '"arguments":' (the following space starts the next
    pre-token), so continuation tokens re-encode identically.
    """
    user = {"role": "user", "content": "x"}
    generation_prompt = tokenizer.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
    prefixes = {DIRECT: DIRECT_PREFIX}
    for name in tool_names:
        call = {"type": "function", "function": {"name": name, "arguments": _ARGS_MARKER}}
        rendered = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": "", "tool_calls": [call]}], tokenize=False
        )
        if not rendered.startswith(generation_prompt):
            raise ValueError("chat template does not render the assistant turn after the generation prompt")
        prefix = rendered[len(generation_prompt): rendered.index(_ARGS_MARKER)].rstrip()
        if not prefix.startswith("<tool_call>") or not prefix.endswith('"arguments":'):
            raise ValueError(f"unexpected tool-call rendering for decision prefix: {prefix!r}")
        prefixes[name] = prefix
    # vLLM continues the final assistant message; the continued prompt must equal
    # the generation prompt followed by the prefix, or forced context differs from training.
    for prefix in prefixes.values():
        continued = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": prefix}],
            add_generation_prompt=False, continue_final_message=True, tokenize=False,
        )
        if continued != generation_prompt + prefix:
            raise ValueError(f"continue_final_message does not reproduce the generation prompt for {prefix!r}")
    return prefixes


def assign_forced_decisions(n: int, decisions: List[str], forced_per_decision: int) -> List[str | None]:
    """Forced decision for each sample index within one prompt's rollout group."""
    forced = [decision for decision in decisions for _ in range(forced_per_decision)]
    if len(forced) >= n:
        raise ValueError(
            f"rollout.n={n} must exceed forced branches {len(forced)} to keep on-policy samples"
        )
    return forced + [None] * (n - len(forced))


def merge_forced_completion(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Prepend the forced prefix and re-parse tool calls as vLLM's hermes parser does.

    The server-side parser only sees the continuation, so a forced tool call would
    otherwise be lost. Sampled log probabilities cover only the continuation and
    are dropped, so rollout/training consistency skips this turn.
    """
    choice = data["choices"][0]
    message = choice["message"]
    text = prefix + (message.get("content") or "")
    choice["logprobs"] = None
    message.pop("tool_calls", None)
    message["content"] = text
    if "<tool_call>" not in text:
        return data
    calls = []
    try:
        for match in HERMES_CALL_RE.findall(text):
            payload = json.loads(match[0] or match[1])
            calls.append({
                "id": f"chatcmpl-tool-{uuid4().hex}",
                "type": "function",
                "function": {"name": payload["name"], "arguments": json.dumps(payload["arguments"], ensure_ascii=False)},
            })
    except (json.JSONDecodeError, KeyError, TypeError):
        # hermes returns the whole text as content when any call fails to parse.
        return data
    message["content"] = text[: text.find("<tool_call>")] or None
    message["tool_calls"] = calls
    choice["finish_reason"] = "tool_calls"
    return data


def first_decision(responses: List[Dict[str, Any]], decisions: List[str]) -> str:
    """Classify the first assistant turn of an on-policy trajectory."""
    first = next((message for message in responses if message.get("role") == "assistant"), None)
    if first is None:
        return OTHER
    if first.get("tool_calls"):
        name = first["tool_calls"][0]["function"]["name"]
        return name if name in decisions else OTHER
    content = first.get("content") or ""
    if "<tool_call>" in content:
        match = TOOL_CALL_RE.search(content)
        try:
            name = json.loads(match.group(1))["name"] if match else None
        except (json.JSONDecodeError, KeyError, TypeError):
            name = None
        return name if name in decisions else OTHER
    return DIRECT


def decision_mask(response_ids: torch.Tensor, prefix_ids: List[int] | None, eos_id: int) -> torch.Tensor:
    """Mask the first occurrence of the decision prefix inside the first assistant turn."""
    mask = torch.zeros_like(response_ids, dtype=torch.long)
    if not prefix_ids:
        return mask
    ids = response_ids.tolist()
    first_end = ids.index(eos_id) if eos_id in ids else len(ids)
    width = len(prefix_ids)
    for start in range(first_end - width + 1):
        if ids[start: start + width] == prefix_ids:
            mask[start: start + width] = 1
            break
    return mask
