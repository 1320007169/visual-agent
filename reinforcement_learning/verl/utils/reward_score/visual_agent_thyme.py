"""Outcome reward for Visual-Agent training on the Thyme RL questions."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def has_strict_answer_format(text: str) -> bool:
    """Require exactly one non-empty answer tag at the end of the final turn."""
    if len(re.findall(r"<answer>", text, flags=re.IGNORECASE)) != 1:
        return False
    if len(re.findall(r"</answer>", text, flags=re.IGNORECASE)) != 1:
        return False

    final_message = text
    if re.search(r"<tool_response>", text, flags=re.IGNORECASE):
        assistant_markers = list(
            re.finditer(r"(?:<\|im_start\|>|^|\s)assistant\s*", text, flags=re.IGNORECASE)
        )
        if not assistant_markers:
            return False
        final_message = text[assistant_markers[-1].end() :]

    # Reward managers decode response IDs with special tokens enabled. Accept
    # chat terminators emitted by common model templates, but no ordinary text,
    # after the final answer tag.
    final_message = re.sub(
        r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|</s>))+\s*$",
        "",
        final_message,
        flags=re.IGNORECASE,
    )

    return re.search(
        r"<answer>\s*[^<>]+?\s*</answer>\s*$",
        final_message,
        flags=re.DOTALL | re.IGNORECASE,
    ) is not None


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?")


RELATION_LABELS = frozenset(
    {
        "above", "below", "left of", "right of", "in front of",
        "behind", "inside", "contain", "overlap", "next to",
    }
)
RELATION_ALIASES = {
    "left": "left of",
    "to the left of": "left of",
    "on the left of": "left of",
    "right": "right of",
    "to the right of": "right of",
    "on the right of": "right of",
    "under": "below",
    "beneath": "below",
    "inside of": "inside",
    "within": "inside",
    "contains": "contain",
    "containing": "contain",
    "overlaps": "overlap",
    "overlapping": "overlap",
    "beside": "next to",
    "adjacent to": "next to",
}


def normalize_relation_answer(text: str) -> str | None:
    """Normalize a complete relation label, rejecting ambiguous or free-form text."""
    if not isinstance(text, str):
        return None
    normalized = normalize_answer(text)
    normalized = RELATION_ALIASES.get(normalized, normalized)
    return normalized if normalized in RELATION_LABELS else None


def relation_match(prediction: str, ground_truth: str) -> bool:
    prediction = normalize_relation_answer(prediction)
    ground_truth = normalize_relation_answer(ground_truth)
    return prediction is not None and ground_truth is not None and prediction == ground_truth


def _is_zwz_relation_task(extra_info) -> bool:
    if not isinstance(extra_info, dict):
        return False
    return (
        extra_info.get("source") == "zwz_rl_vqa/original_images"
        or extra_info.get("data_source") == "visual-agent-zwz-relation"
        or extra_info.get("task_type") == "zwz_original_relation"
    )


def _is_vision_opd_task(extra_info) -> bool:
    if not isinstance(extra_info, dict):
        return False
    return (
        extra_info.get("source") == "vision-opd/original_images"
        or extra_info.get("data_source") == "visual-agent-vision-opd"
    )


def grounding_query_penalty(text: str) -> tuple[int, float]:
    max_words = int(os.environ.get("GROUNDING_QUERY_MAX_WORDS", "0"))
    penalty_per_query = float(os.environ.get("GROUNDING_QUERY_PENALTY", "0"))
    if max_words < 1 or penalty_per_query <= 0:
        return 0, 0.0

    invalid_queries = 0
    for raw_call in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(raw_call)
        except json.JSONDecodeError:
            continue
        if call.get("name") != "grounding_detect":
            continue
        arguments = call.get("arguments")
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not isinstance(query, str) or len(query.split()) > max_words:
            invalid_queries += 1

    max_penalty = float(os.environ.get("GROUNDING_QUERY_MAX_PENALTY", "0.2"))
    return invalid_queries, min(max_penalty, penalty_per_query * invalid_queries)


def rule_match(prediction: str, ground_truth: str) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if pred == gold:
        return True

    # Accept an option label followed by its text, e.g. gold "C", pred "C. S QT 911".
    if len(gold) == 1 and gold in "abcdefghijklmnopqrstuvwxyz" and re.match(
        rf"^{re.escape(gold)}(?:\b|[.)：:])", pred
    ):
        return True

    try:
        from math_verify import parse, verify

        parsed_gold = parse(ground_truth)
        parsed_pred = parse(prediction)
        if parsed_gold and parsed_pred and verify(parsed_gold, parsed_pred):
            return True
    except Exception:
        pass
    return False


@lru_cache(maxsize=1)
def _judge_client_and_model():
    base_url = os.environ.get("LLM_AS_A_JUDGE_BASE", "").rstrip("/")
    if not base_url:
        return None, None
    from openai import OpenAI

    if not base_url.endswith("/v1"):
        base_url += "/v1"
    timeout = float(os.environ.get("LLM_AS_A_JUDGE_TIMEOUT", "20"))
    client = OpenAI(
        api_key=os.environ.get("LLM_AS_A_JUDGE_KEY", "EMPTY"),
        base_url=base_url,
        timeout=timeout,
        # Retries are controlled below so an unavailable judge cannot multiply
        # SDK retries and stall every reward worker for a long time.
        max_retries=0,
    )
    model = os.environ.get("LLM_AS_A_JUDGE_MODEL", "").strip()
    if not model:
        models = client.models.list()
        model = models.data[0].id
    return client, model


def judge_match(question: str, prediction: str, ground_truth: str) -> bool:
    client, model = _judge_client_and_model()
    if client is None:
        return False

    prompt = f"""Judge whether the candidate answer is semantically equivalent to the reference answer for the question.
Return exactly TRUE or FALSE and nothing else.

Question:
{question}

Reference answer:
{ground_truth}

Candidate answer:
{prediction}
"""
    attempts = max(1, int(os.environ.get("LLM_AS_A_JUDGE_RETRIES", "2")))
    for _ in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
                extra_body={"thinking": {"type": "disabled"}},
            )
            verdict = response.choices[0].message.content.strip().upper()
            if verdict.startswith("TRUE"):
                return True
            if verdict.startswith("FALSE"):
                return False
        except Exception as exc:
            print(f"[visual-agent-thyme reward] judge request failed: {exc}")
    return False


def compute_score(solution_str: str, ground_truth: str, extra_info=None):
    answer = extract_answer(solution_str)
    format_reward = 1.0 if has_strict_answer_format(solution_str) else 0.0
    invalid_queries, query_penalty = grounding_query_penalty(solution_str)
    if not answer:
        result = {"score": 0.0, "acc": 0.0, "format": 0.0, "tool_used": 0.0}
        if int(os.environ.get("GROUNDING_QUERY_MAX_WORDS", "0")) > 0:
            result["invalid_grounding_queries"] = float(invalid_queries)
            result["query_penalty"] = query_penalty
        return result

    if _is_zwz_relation_task(extra_info):
        # Closed relation labels must not receive a semantic-judge fallback:
        # ambiguous answers such as "on" can otherwise match several labels.
        correct = relation_match(answer, ground_truth)
    elif _is_vision_opd_task(extra_info):
        # Vision-OPD is closed A-D multiple choice. Keep its reward exact and
        # deterministic rather than sending malformed/free-form answers to a judge.
        normalized_answer = normalize_answer(answer).upper()
        normalized_ground_truth = normalize_answer(ground_truth).upper()
        correct = normalized_answer in {"A", "B", "C", "D"} and normalized_answer == normalized_ground_truth
    else:
        question = str((extra_info or {}).get("question", ""))
        correct = rule_match(answer, ground_truth)
        if not correct:
            correct = judge_match(question, answer, ground_truth)
    accuracy_reward = 1.0 if correct else 0.0
    tool_used = 1.0 if "<tool_call>" in solution_str else 0.0
    result = {
        "score": max(0.0, 0.9 * accuracy_reward + 0.1 * format_reward - query_penalty),
        "acc": accuracy_reward,
        "format": format_reward,
        "tool_used": tool_used,
    }
    if int(os.environ.get("GROUNDING_QUERY_MAX_WORDS", "0")) > 0:
        result["invalid_grounding_queries"] = float(invalid_queries)
        result["query_penalty"] = query_penalty
    return result
