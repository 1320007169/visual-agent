#!/usr/bin/env python3
"""Visually judge RL-to-SFT trajectories with the official DeepSeek V4.1 API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageOps
except ModuleNotFoundError:
    modelarts_sdk = Path("/opt/huawei/modelarts-dev/modelarts-sdk")
    if modelarts_sdk.is_dir():
        sys.path.insert(0, str(modelarts_sdk))
    from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "data/rl_distill/qwen3_groundingdino_kl_safe_success_sft.jsonl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/deepseek_v41"
DEFAULT_KEY_FILE = (
    Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx")
    / "secrets/deepseek_api_key.txt"
)
SOURCE_ROLLOUT_DIR = (
    Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts")
    / "visual-agent-zwz-rl"
    / "zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual"
)
SOURCE_ROLLOUT_STEPS: Any = [1, 165]
PROMPT_VERSION = "deepseek-v41-visual-trajectory-judge-v1"
ANSWER_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

SYSTEM_PROMPT = """You are a strict visual trajectory quality judge for rejection-sampling fine-tuning.
The reference answer has already passed deterministic answer checks. Do not reward a trajectory merely because its final answer matches the reference. Decide whether the visible evidence and tool trajectory genuinely support that answer.

Inspect the supplied images, including annotated images showing detector boxes. Evaluate:
1. Whether grounding queries refer to the subject and reference entities in the question.
2. Whether detector boxes visually identify the requested entities rather than proxies or unrelated objects.
3. Whether retries and crops are relevant and logically follow earlier observations.
4. Whether the observations support the claimed spatial relation.
5. Whether the final explanation is consistent with the answer and evidence.

Use REVIEW when the image or depth relation is genuinely ambiguous. Use REJECT for a clear wrong entity, wrong box, irrelevant call, unsupported relation, contradictory explanation, or materially broken trajectory. Minor redundancy alone is not enough to reject an otherwise sound trajectory.

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

COLORS = [
    (230, 57, 70),
    (29, 130, 246),
    (22, 163, 74),
    (245, 158, 11),
    (147, 51, 234),
    (8, 145, 178),
]


class RateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if delay:
            time.sleep(delay)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(index: int, raw_line: str) -> str:
    digest = hashlib.sha256(raw_line.rstrip("\n").encode("utf-8")).hexdigest()[:16]
    return f"line-{index:05d}-{digest}"


def parse_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\{", cleaned)):
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response contains no JSON object")


def normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    boolean_fields = [
        "entity_queries_valid",
        "detections_visually_correct",
        "tool_sequence_reasonable",
        "evidence_supports_answer",
        "final_explanation_consistent",
    ]
    for field in boolean_fields:
        if type(payload.get(field)) is not bool:
            raise ValueError(f"judge field {field} must be boolean")
    confidence = payload.get("confidence")
    if type(confidence) is not int or not 1 <= confidence <= 5:
        raise ValueError("judge confidence must be an integer from 1 to 5")
    provider_verdict = str(payload.get("verdict", "")).strip().lower()
    if provider_verdict not in {"accept", "reject", "review"}:
        raise ValueError("judge verdict must be accept, reject, or review")
    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list) or any(not isinstance(code, str) for code in reason_codes):
        raise ValueError("judge reason_codes must be a list of strings")

    all_critical_pass = all(payload[field] for field in boolean_fields)
    if provider_verdict == "accept" and all_critical_pass and confidence >= 4:
        verdict = "accept"
    elif provider_verdict == "reject" and not all_critical_pass and confidence >= 4:
        verdict = "reject"
    else:
        verdict = "review"
    return {
        **{field: payload[field] for field in boolean_fields},
        "confidence": confidence,
        "provider_verdict": provider_verdict,
        "verdict": verdict,
        "reason_codes": reason_codes[:12],
        "brief_reason": str(payload.get("brief_reason", "")).strip()[:500],
    }


def parse_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    pending: dict[str, Any] | None = None
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "assistant":
            match = TOOL_CALL_RE.search(content)
            if match:
                pending = json.loads(match.group(1))
        elif message.get("role") == "user" and pending is not None:
            match = TOOL_RESPONSE_RE.search(content)
            if match:
                pairs.append({"call": pending, "response": json.loads(match.group(1))})
                pending = None
    if pending is not None:
        raise ValueError("trajectory ends with an unmatched tool call")
    return pairs


def trajectory_text(item: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    messages = item.get("messages") or []
    question = next(
        (
            str(message.get("content", "")).replace("<image>", "").strip()
            for message in messages
            if message.get("role") == "user" and "<tool_response>" not in str(message.get("content", ""))
        ),
        "",
    )
    final = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "assistant" and "<answer>" in str(message.get("content", ""))
        ),
        "",
    )
    answer_match = ANSWER_RE.findall(final)
    reference = answer_match[-1].strip() if answer_match else "MISSING"
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
        "Judge this trajectory. The images follow this text. Image indices match target_image. "
        "Boxes shown on annotated images are numbered by tool step.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER (already rule-verified):\n{reference}\n\n"
        f"TOOL TRAJECTORY:\n{json.dumps(compact_pairs, ensure_ascii=False, indent=2)}\n\n"
        f"ORIGINAL FINAL RESPONSE:\n{final}\n"
    )


def collect_annotations(pairs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for step, pair in enumerate(pairs, start=1):
        call = pair["call"]
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if not isinstance(arguments, dict):
            continue
        target = arguments.get("target_image", 0)
        if type(target) is not int or target < 0:
            continue
        if call.get("name") == "grounding_detect":
            response = pair["response"]
            boxes = response.get("boxes") if isinstance(response, dict) else None
            if not isinstance(boxes, list):
                continue
            query = str(arguments.get("query", "object"))
            for box_index, box in enumerate(boxes, start=1):
                if isinstance(box, list) and len(box) == 4:
                    annotations[target].append(
                        {"box": box, "text": f"S{step}.{box_index} {query}", "color": COLORS[(step - 1) % len(COLORS)]}
                    )
        elif call.get("name") == "crop_zoom":
            box = arguments.get("bbox_2d")
            if isinstance(box, list) and len(box) == 4:
                label = str(arguments.get("label", "crop"))
                annotations[target].append(
                    {"box": box, "text": f"S{step} CROP {label}", "color": COLORS[(step - 1) % len(COLORS)]}
                )
    return annotations


def image_data_url(path: Path, annotations: list[dict[str, Any]], max_edge: int, quality: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(max(width, height) / 350))
    for annotation in annotations:
        left, top, right, bottom = [float(value) for value in annotation["box"]]
        rect = [left / 1000 * width, top / 1000 * height, right / 1000 * width, bottom / 1000 * height]
        color = annotation["color"]
        draw.rectangle(rect, outline=color, width=line_width)
        # Pillow's built-in bitmap font is Latin-1 only on the lightweight
        # ModelArts runtime. The full query remains in the textual transcript;
        # an ASCII-safe overlay is sufficient to associate boxes with steps.
        text = annotation["text"][:80].encode("ascii", "replace").decode("ascii")
        text_box = draw.textbbox((rect[0], rect[1]), text)
        text_height = text_box[3] - text_box[1] + 4
        label_top = max(0, rect[1] - text_height)
        label_box = [rect[0], label_top, min(width, rect[0] + text_box[2] - text_box[0] + 6), rect[1]]
        draw.rectangle(label_box, fill=color)
        draw.text((rect[0] + 3, label_top + 2), text, fill=(255, 255, 255))
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(item: dict[str, Any], max_edge: int, jpeg_quality: int) -> list[dict[str, Any]]:
    messages = item.get("messages")
    images = item.get("images")
    if not isinstance(messages, list) or not isinstance(images, list) or not images:
        raise ValueError("sample must contain messages and at least one image")
    pairs = parse_tool_pairs(messages)
    annotations = collect_annotations(pairs)
    content: list[dict[str, Any]] = [{"type": "text", "text": trajectory_text(item, pairs)}]
    for index, raw_path in enumerate(images):
        path = Path(str(raw_path))
        content.append(
            {
                "type": "text",
                "text": f"IMAGE {index} ({'annotated tool target' if annotations.get(index) else 'trajectory image'}):",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(path, annotations.get(index, []), max_edge, jpeg_quality),
                    "detail": "original",
                },
            }
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def endpoint_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def request_judgment(
    *,
    item: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    max_tokens: int,
    max_edge: int,
    jpeg_quality: int,
    limiter: RateLimiter,
) -> dict[str, Any]:
    messages = build_messages(item, max_edge=max_edge, jpeg_quality=jpeg_quality)
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    endpoint = endpoint_from_base(base_url)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        limiter.wait()
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            response_message = response_body["choices"][0]["message"]
            content = response_message.get("content") or response_message.get("reasoning_content") or ""
            decision = normalize_decision(parse_json_block(content))
            return {
                **decision,
                "attempts": attempt,
                "response_id": response_body.get("id"),
                "usage": response_body.get("usage"),
                "raw_response": content,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            status = getattr(exc, "code", None)
            retryable = status in {None, 408, 409, 429, 500, 502, 503, 504}
            if attempt >= retries or not retryable:
                break
            time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    status = getattr(last_error, "code", None)
    detail = re.sub(r"\s+", " ", str(last_error or "unknown error"))[:300]
    raise RuntimeError(
        f"judge request failed after {retries} attempts: "
        f"{type(last_error).__name__}, status={status}, detail={detail}"
    )


def load_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(args.api_key_env, "").strip()
    if not key and args.api_key_file.is_file():
        key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            f"API key is missing; set {args.api_key_env} or provide --api-key-file"
        )
    return key


def load_existing_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions = {}
    if not path.exists():
        return decisions
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = value.get("sample_id")
            if not isinstance(identifier, str):
                raise ValueError(f"invalid decision at {path}:{line_number}")
            decisions[identifier] = value
    return decisions


def rebuild_partitions(
    input_path: Path,
    output_dir: Path,
    decisions: dict[str, dict[str, Any]],
) -> dict[str, int]:
    handles: dict[str, tuple[Any, Path]] = {}
    counts = Counter()
    try:
        for verdict in ("accepted", "review", "rejected"):
            handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False)
            handles[verdict] = (handle, Path(handle.name))
        with input_path.open("r", encoding="utf-8") as source:
            for index, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                identifier = sample_id(index, raw_line)
                decision = decisions.get(identifier)
                if decision is None:
                    counts["pending"] += 1
                    continue
                verdict = decision["verdict"]
                partition = {"accept": "accepted", "review": "review", "reject": "rejected"}[verdict]
                handles[partition][0].write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                counts[partition] += 1
        for handle, _ in handles.values():
            handle.close()
        for verdict, (_, temporary) in handles.items():
            os.replace(temporary, output_dir / f"{verdict}.jsonl")
    finally:
        for handle, temporary in handles.values():
            if not handle.closed:
                handle.close()
            temporary.unlink(missing_ok=True)
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-image-edge", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="New samples to judge; 0 means all remaining samples.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and prepare one request without calling the API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.max_workers < 1 or args.retries < 1 or args.start_index < 1:
        raise ValueError("max-workers, retries, and start-index must be positive")
    if not 1 <= args.jpeg_quality <= 95 or args.max_image_edge < 256:
        raise ValueError("invalid image encoding settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_digest = sha256_file(args.input)
    config = {
        "prompt_version": PROMPT_VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": input_digest,
        "source_rollout_dir": str(SOURCE_ROLLOUT_DIR),
        "source_rollout_steps": SOURCE_ROLLOUT_STEPS,
        "model": args.model,
        "base_url": args.base_url,
        "max_image_edge": args.max_image_edge,
        "jpeg_quality": args.jpeg_quality,
    }
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(f"existing run configuration differs: {config_path}")
    else:
        atomic_write_json(config_path, config)

    raw_samples = []
    with args.input.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            if index < args.start_index or not raw_line.strip():
                continue
            raw_samples.append((index, raw_line, sample_id(index, raw_line)))

    decisions_path = args.output_dir / "decisions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    decisions = load_existing_decisions(decisions_path)
    pending = [sample for sample in raw_samples if sample[2] not in decisions]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        if not pending:
            print("No pending samples.")
            return
        index, raw_line, identifier = pending[0]
        item = json.loads(raw_line)
        messages = build_messages(item, args.max_image_edge, args.jpeg_quality)
        image_parts = sum(
            part.get("type") == "image_url"
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
        print(json.dumps({
            "status": "dry_run_ok",
            "sample_index": index,
            "sample_id": identifier,
            "image_parts": image_parts,
            "tool_pairs": len(parse_tool_pairs(item["messages"])),
            "model": args.model,
            "endpoint": endpoint_from_base(args.base_url),
        }, indent=2))
        return

    if not pending:
        partitions = rebuild_partitions(args.input, args.output_dir, decisions)
        print(json.dumps({"status": "nothing_to_do", **partitions}, indent=2))
        return

    api_key = load_api_key(args)
    limiter = RateLimiter(args.requests_per_minute)
    print(
        f"Judging {len(pending)} samples with {args.model}; "
        f"workers={args.max_workers}, rpm={args.requests_per_minute:g}",
        flush=True,
    )

    def worker(sample: tuple[int, str, str]) -> dict[str, Any]:
        index, raw_line, identifier = sample
        item = json.loads(raw_line)
        judged = request_judgment(
            item=item,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
            max_edge=args.max_image_edge,
            jpeg_quality=args.jpeg_quality,
            limiter=limiter,
        )
        return {
            "sample_id": identifier,
            "sample_index": index,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            **judged,
        }

    run_counts = Counter()
    with decisions_path.open("a", encoding="utf-8") as decision_handle, errors_path.open("a", encoding="utf-8") as error_handle:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(worker, sample): sample for sample in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                index, _, identifier = futures[future]
                try:
                    decision = future.result()
                except Exception as exc:
                    error = {
                        "sample_id": identifier,
                        "sample_index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                    error_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    error_handle.flush()
                    run_counts["error"] += 1
                else:
                    decision_handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
                    decision_handle.flush()
                    decisions[identifier] = decision
                    run_counts[decision["verdict"]] += 1
                if completed % 20 == 0 or completed == len(pending):
                    print(
                        f"Completed {completed}/{len(pending)}: "
                        f"accept={run_counts['accept']} review={run_counts['review']} "
                        f"reject={run_counts['reject']} error={run_counts['error']}",
                        flush=True,
                    )

    partitions = rebuild_partitions(args.input, args.output_dir, decisions)
    verdict_counts = Counter(decision["verdict"] for decision in decisions.values())
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": input_digest,
        "source_rollout_dir": str(SOURCE_ROLLOUT_DIR),
        "source_rollout_steps": SOURCE_ROLLOUT_STEPS,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "judged": len(decisions),
        "verdict_counts": dict(verdict_counts),
        "partitions": partitions,
        "this_run": dict(run_counts),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
