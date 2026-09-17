#!/usr/bin/env python3
"""Run the shared DeepSeek visual judge with a mixed-task prompt."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_JUDGE = (
    SCRIPT_DIR.parent
    / "zwz_relation_qwen3_kl_safe_steps001_165_success10362"
    / "judge_with_deepseek_v41.py"
)
SPEC = importlib.util.spec_from_file_location("shared_deepseek_visual_judge", SHARED_JUDGE)
judge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge)

judge.DEFAULT_INPUT = SCRIPT_DIR / "data/new_unique_correct_tool_trajectories.jsonl"
judge.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/deepseek_v41"
judge.SOURCE_ROLLOUT_DIR = {
    "16gpu_n8_2node": str(
        Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts")
        / "visual-agent-zwz-rl/zwz_deepeyesv2_3k_nocount_v1_n8_2node"
    ),
    "64gpu_b224_8node": str(
        Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts")
        / "visual-agent-zwz-rl/zwz_deepeyesv2_3k_nocount_v1_n8_b224_8node_scratch"
    ),
}
judge.SOURCE_ROLLOUT_STEPS = {
    "16gpu_n8_2node": "captured by prepare_report.json",
    "64gpu_b224_8node": "captured by prepare_report.json",
}
judge.PROMPT_VERSION = "deepseek-v41-visual-trajectory-judge-mixed-v1"
judge.SYSTEM_PROMPT = """You are a strict visual trajectory quality judge for rejection-sampling fine-tuning.
The candidate has already passed the training reward's answer and format checks. Do not accept it merely because its final answer resembles the reference. Decide whether the visible evidence and tool trajectory genuinely support the answer.

The task may ask about a spatial relation, a visual attribute, or text in the image. Only the original source image is supplied; no reconstructed crop image is sent. The original image is annotated with detector boxes and crop regions when applicable. Use both the original image and the recorded crop coordinates to evaluate:
1. Whether grounding queries or crop regions target only the entities or region needed by the question. A grounding query or crop label must not contain, imply, or presuppose the candidate/reference answer. For example, a relation query such as "cup left of plate", an attribute query such as "brown cushion" when color is being asked, or a text-region label containing the answer word is answer leakage.
2. Whether detector boxes visually identify the requested entities rather than proxies or unrelated objects.
3. Whether retries and crops are relevant and logically follow earlier observations.
4. Whether the visible observations support the answer for this task type. GroundingDINO cannot itself verify text or visual attributes; those require visual inspection of the original image or crop.
5. Whether the assistant's reasoning and final response are mutually consistent and grounded in the observations.

Use REVIEW when the image, text, attribute, or spatial relation is genuinely ambiguous. Use REJECT for a clear wrong entity, wrong box or crop, irrelevant call, unsupported conclusion, invented observation, contradictory reasoning, or materially broken trajectory. Also REJECT whenever a tool query or crop label leaks the answer or uses the predicted answer to identify the target, and include `answer_leakage_in_query` in reason_codes. Minor redundancy alone is not enough to reject an otherwise sound trajectory.

Return one JSON object only, with exactly these fields:
{
  "entity_queries_valid": true,
  "detections_visually_correct": true,
  "tool_sequence_reasonable": true,
  "evidence_supports_answer": true,
  "final_explanation_consistent": true,
  "confidence": 1,
  "verdict": "accept",
  "reason_codes": [],
  "brief_reason": "short factual reason"
}

confidence must be an integer from 1 to 5. verdict must be accept, reject, or review. Do not output markdown or hidden reasoning."""


def mixed_trajectory_text(item: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    messages = item.get("messages") or []
    metadata = item.get("metadata") or {}
    question = next(
        (
            str(message.get("content", "")).replace("<image>", "").strip()
            for message in messages
            if message.get("role") == "user"
            and "<tool_response>" not in str(message.get("content", ""))
        ),
        "",
    )
    final = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        "",
    )
    reference = str(metadata.get("reference_answer") or "MISSING")
    assistant_trajectory = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ]
    compact_pairs = []
    for number, pair in enumerate(pairs, start=1):
        call = pair["call"]
        response = pair["response"]
        if call.get("name") == "grounding_detect":
            compact_response = {
                key: response.get(key)
                for key in ("query", "boxes", "labels", "confidence", "count", "coordinate_space")
                if key in response
            }
        else:
            crop = response.get("crop_zoom") if isinstance(response, dict) else None
            compact_response = {
                "target_image": response.get("target_image") if isinstance(response, dict) else None,
                "coordinate_space": response.get("coordinate_space") if isinstance(response, dict) else None,
                "crop_zoom": {
                    key: crop.get(key)
                    for key in ("target_image", "requested_bbox_2d", "bbox_2d")
                    if isinstance(crop, dict) and key in crop
                },
            }
        compact_pairs.append({"step": number, "call": call, "observation": compact_response})
    return (
        "Judge this trajectory. Only IMAGE 0, the original source image, is supplied. "
        "No crop image is supplied; later target_image values refer to recorded crop results only. "
        "Boxes and crop regions visible on the annotated original image are numbered by tool step.\n\n"
        f"TASK TYPE:\n{metadata.get('ability', 'unknown')}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER (from the source dataset):\n{reference}\n\n"
        f"TOOL TRAJECTORY:\n{json.dumps(compact_pairs, ensure_ascii=False, indent=2)}\n\n"
        "ALL ORIGINAL ASSISTANT TURNS (including any reasoning):\n"
        f"{json.dumps(assistant_trajectory, ensure_ascii=False, indent=2)}\n\n"
        f"ORIGINAL FINAL RESPONSE:\n{final}\n"
    )


judge.trajectory_text = mixed_trajectory_text


if __name__ == "__main__":
    judge.main()
