"""Outcome reward for Visual-Agent training on the Thyme RL questions."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from threading import Lock


_judge_stats = Counter()
_judge_stats_lock = Lock()


def _record_judge_stat(name):
    with _judge_stats_lock:
        _judge_stats[name] += 1


def judge_stats_snapshot():
    with _judge_stats_lock:
        return dict(_judge_stats)


def judge_batch_summary(before, samples):
    after = judge_stats_snapshot()
    counts = {key: after.get(key, 0) - before.get(key, 0) for key in after}
    for key in ("judge_samples", "final_unresolved", "fallbacks"):
        counts.setdefault(key, 0)
    for endpoint in ("primary", "backup"):
        for metric in ("requests", "valid", "true", "false", "errors", "invalid", "rate_limited", "timeouts"):
            counts.setdefault(f"{endpoint}_{metric}", 0)
    counts["samples"] = samples
    counts["judge_sample_ratio"] = counts["judge_samples"] / samples if samples else 0.0
    counts["final_unresolved_ratio"] = counts["final_unresolved"] / counts["judge_samples"] if counts["judge_samples"] else 0.0
    return counts


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def has_think_action_format(text: str) -> bool:
    """Validate Reason-Act syntax, with optional think tags or plain analysis.

    Qwen response IDs retain im_start/im_end delimiters; the first assistant
    header is already part of the prompt and may be absent here.
    """
    turns = []
    for chunk in text.split("<|im_end|>"):
        chunk = chunk.strip()
        if not chunk or chunk == "<|endoftext|>":
            continue
        match = re.match(r"<\|im_start\|>(assistant|user|tool)\n", chunk)
        if match:
            if match[1] != "assistant":
                continue
            chunk = chunk[match.end():]
        turns.append(chunk.strip())
    if not turns:
        return False
    for index, turn in enumerate(turns):
        match = re.fullmatch(
            r"(.*?)<(tool_call|answer)>\s*(.+?)\s*</\2>",
            turn, re.DOTALL,
        )
        if not match or not match[3].strip():
            return False
        analysis = match[1].strip()
        if "<think>" in analysis or "</think>" in analysis:
            thought = re.fullmatch(r"<think>(.*?)</think>", analysis, re.DOTALL)
            if not thought or not thought[1].strip():
                return False
            analysis = thought[1]
        if re.search(r"</?(?:think|tool_call|answer|tool_response)\b", analysis, re.IGNORECASE):
            return False
        if match[2] == "answer":
            if index != len(turns) - 1 or re.search(r"[<>]", match[3]):
                return False
        else:
            if index == len(turns) - 1:
                return False
            try:
                call = json.loads(match[3])
            except (ValueError, TypeError):
                return False
            if not isinstance(call, dict) or call.get("name") not in {
                "grounding_detect", "crop_zoom", "depth_measure", "object_count"
            } or not isinstance(call.get("arguments"), dict):
                return False
    return True


def has_strict_answer_format(text: str) -> bool:
    """Require exactly one non-empty answer tag at the end of the final turn."""
    # Accept the old switch for existing launchers, but no longer require tags:
    # reasoning content/length must not be a prerequisite for format credit.
    if (os.environ.get("VISUAL_AGENT_FORMAT_PROTOCOL") == "reason_act"
            or os.environ.get("VISUAL_AGENT_REQUIRE_THINK", "0") == "1"):
        return has_think_action_format(text)
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


def multiple_choice_match(prediction: str, ground_truth: str) -> bool:
    """Match a closed A-D answer, optionally followed by its option text."""
    gold = normalize_answer(ground_truth).upper()
    if gold not in {"A", "B", "C", "D"}:
        return False
    candidate = prediction.strip()
    match = re.fullmatch(r"([A-Da-d])(?:\s*[.)：:](?:\s*\S.*)?)?", candidate, flags=re.DOTALL)
    return match is not None and match.group(1).upper() == gold


@lru_cache(maxsize=2)
def _judge_client_and_model(backup=False):
    prefix = "LLM_AS_A_JUDGE_BACKUP" if backup else "LLM_AS_A_JUDGE"
    base_url = os.environ.get(f"{prefix}_BASE", "").rstrip("/")
    if not base_url:
        return None, None
    from openai import OpenAI

    if not base_url.endswith("/v1"):
        base_url += "/v1"
    timeout = float(os.environ.get("LLM_AS_A_JUDGE_TIMEOUT", "20"))
    client = OpenAI(
        api_key=os.environ.get(f"{prefix}_KEY", "EMPTY"),
        base_url=base_url,
        timeout=timeout,
        # Retries are controlled below so an unavailable judge cannot multiply
        # SDK retries and stall every reward worker for a long time.
        max_retries=0,
    )
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    if not model:
        models = client.models.list()
        model = models.data[0].id
    return client, model


def _judge_endpoint(question_prompt: str, *, backup: bool, attempts: int) -> bool | None:
    endpoint = "backup" if backup else "primary"
    client, model = None, None
    for _ in range(attempts):
        try:
            if client is None:
                client, model = _judge_client_and_model(backup=backup)
            if client is None:
                return None
            _record_judge_stat(f"{endpoint}_requests")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a binary answer verifier. Output exactly one token: "
                            "TRUE or FALSE. Do not explain your decision."
                        ),
                    },
                    {"role": "user", "content": question_prompt},
                ],
                temperature=0.0,
                max_tokens=8,
                stop=["\n"],
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = (response.choices[0].message.content or "").strip()
            verdict = None
            normalized = content.strip("`\"' \t\r\n").upper()
            if normalized == "TRUE":
                verdict = True
            elif normalized == "FALSE":
                verdict = False
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and isinstance(payload.get("verdict"), bool):
                    verdict = payload["verdict"]
            except json.JSONDecodeError:
                pass
            if verdict is True:
                _record_judge_stat(f"{endpoint}_valid")
                _record_judge_stat(f"{endpoint}_true")
                return True
            if verdict is False:
                _record_judge_stat(f"{endpoint}_valid")
                _record_judge_stat(f"{endpoint}_false")
                return False
            _record_judge_stat(f"{endpoint}_invalid")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            _record_judge_stat(f"{endpoint}_errors")
            if status == 429:
                _record_judge_stat(f"{endpoint}_rate_limited")
            if isinstance(exc, TimeoutError) or type(exc).__name__ == "APITimeoutError":
                _record_judge_stat(f"{endpoint}_timeouts")
            # Never log raw provider errors: they may contain request credentials.
            print(
                f"[visual-agent-thyme reward] {endpoint} judge request failed: "
                f"{type(exc).__name__}, status={status}",
                flush=True,
            )
    return None


def judge_match(question: str, prediction: str, ground_truth: str) -> bool | None:
    _record_judge_stat("judge_samples")
    prompt = f"""Judge whether the candidate answer is semantically equivalent to the reference answer for the question.
Return exactly TRUE or FALSE and nothing else.

Question:
{question}

Reference answer:
{ground_truth}

Candidate answer:
{prediction}
"""
    primary_attempts = max(1, int(os.environ.get("LLM_AS_A_JUDGE_PRIMARY_RETRIES", "1")))
    result = _judge_endpoint(prompt, backup=False, attempts=primary_attempts)
    if result is not None:
        return result

    if os.environ.get("LLM_AS_A_JUDGE_BACKUP_BASE"):
        _record_judge_stat("fallbacks")
        backup_attempts = max(
            1,
            int(
                os.environ.get(
                    "LLM_AS_A_JUDGE_BACKUP_RETRIES",
                    os.environ.get("LLM_AS_A_JUDGE_RETRIES", "2"),
                )
            ),
        )
        result = _judge_endpoint(prompt, backup=True, attempts=backup_attempts)
        if result is not None:
            return result

    _record_judge_stat("final_unresolved")
    return None


def compute_score(solution_str: str, ground_truth: str, extra_info=None):
    answer = extract_answer(solution_str)
    format_reward = 1.0 if has_strict_answer_format(solution_str) else 0.0
    invalid_queries, query_penalty = grounding_query_penalty(solution_str)
    if not answer:
        result = {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "tool_used": 0.0,
            "reward_valid": 1.0,
        }
        if int(os.environ.get("GROUNDING_QUERY_MAX_WORDS", "0")) > 0:
            result["invalid_grounding_queries"] = float(invalid_queries)
            result["query_penalty"] = query_penalty
        return result

    if _is_zwz_relation_task(extra_info):
        # Closed relation labels must not receive a semantic-judge fallback:
        # ambiguous answers such as "on" can otherwise match several labels.
        correct = relation_match(answer, ground_truth)
    elif _is_vision_opd_task(extra_info) or (extra_info or {}).get("data_source") == "visual-agent-hrbench4k":
        # These are closed A-D tasks. Accept the option label with its displayed
        # text, as the benchmark evaluator does, without using a semantic judge.
        correct = multiple_choice_match(answer, ground_truth)
    elif (extra_info or {}).get("data_source") == "visual-agent-depth-raw":
        correct = (
            multiple_choice_match(answer, ground_truth)
            if (extra_info or {}).get("original_source") == "ca_vqa_multichoice"
            else normalize_answer(answer) == normalize_answer(ground_truth)
        )
    elif (extra_info or {}).get("data_source") == "visual-agent-tallyqa":
        correct = answer.strip() == ground_truth.strip()
    else:
        question = str((extra_info or {}).get("question", ""))
        correct = rule_match(answer, ground_truth)
        if not correct:
            correct = judge_match(question, answer, ground_truth)
            if correct is None:
                # Keep unresolved samples in GRPO, using the ordinary incorrect-
                # answer reward (including format reward and query penalties).
                correct = False
    accuracy_reward = 1.0 if correct else 0.0
    tool_used = 1.0 if "<tool_call>" in solution_str else 0.0
    result = {
        "score": max(0.0, 0.9 * accuracy_reward + 0.1 * format_reward - query_penalty),
        "acc": accuracy_reward,
        "format": format_reward,
        "tool_used": tool_used,
        "reward_valid": 1.0,
    }
    if int(os.environ.get("GROUNDING_QUERY_MAX_WORDS", "0")) > 0:
        result["invalid_grounding_queries"] = float(invalid_queries)
        result["query_penalty"] = query_penalty
    return result
