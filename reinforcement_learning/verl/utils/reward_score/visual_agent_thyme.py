"""Outcome reward for Visual-Agent training on the Thyme RL questions."""

from __future__ import annotations

import os
import re
from functools import lru_cache


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?")


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
    format_reward = 1.0 if answer else 0.0
    if not answer:
        return {"score": 0.0, "acc": 0.0, "format": 0.0, "tool_used": 0.0}

    question = str((extra_info or {}).get("question", ""))
    correct = rule_match(answer, ground_truth)
    if not correct:
        correct = judge_match(question, answer, ground_truth)
    accuracy_reward = 1.0 if correct else 0.0
    tool_used = 1.0 if "<tool_call>" in solution_str else 0.0
    return {
        "score": 0.9 * accuracy_reward + 0.1 * format_reward,
        "acc": accuracy_reward,
        "format": format_reward,
        "tool_used": tool_used,
    }
