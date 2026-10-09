"""Levelled faults for online training tool observations."""

from __future__ import annotations

import random
import re
import string
import unicodedata
from typing import Any


def _normalized(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def _ocr_span(text: str, answers: list[str]) -> tuple[int, int] | None:
    normalized = ""
    positions = []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char).casefold():
            if folded.isalnum():
                normalized += folded
                positions.append(index)
    for answer in sorted(answers, key=lambda value: len(_normalized(value)), reverse=True):
        needle = _normalized(answer)
        start = normalized.find(needle) if needle else -1
        if start >= 0:
            return positions[start], positions[start + len(needle) - 1] + 1
    return None


def _ocr_fault(observation: dict, level: int, rng: random.Random, spec: dict) -> dict | None:
    text = observation.get("text")
    if not isinstance(text, str):
        return None
    answers = [str(value) for value in [spec["answer"], *spec.get("aliases", [])] if str(value).strip()]
    span = _ocr_span(text, answers)
    if span is None:
        return None
    start, end = span
    commands = [match.group() for match in re.finditer(r"\\[A-Za-z]+", text)]
    command_positions = {index for match in re.finditer(r"\\[A-Za-z]+", text)
                         for index in range(match.start(), match.end())}
    visible = [index for index in range(start, end)
               if index not in command_positions and text[index].isascii() and text[index].isalnum()]
    confusable = {"0": "O", "O": "0", "1": "l", "l": "1", "2": "Z", "Z": "2",
                  "5": "S", "S": "5", "8": "B", "B": "8", "m": "rn", "c": "e",
                  "e": "c", "b": "d", "d": "b", "p": "q", "q": "p", "u": "v",
                  "v": "u", "i": "l"}
    if spec.get("variant") == "hme":
        confusable.update({"3": "8", "6": "b", "9": "g", "x": "y", "y": "x", "c": "e", "e": "c"})
    for _ in range(8):
        if level == 0:
            candidates = [index for index in visible if text[index] in confusable]
            if not candidates:
                return None
            index = rng.choice(candidates)
            faulty = text[:index] + confusable[text[index]] + text[index + 1:]
        elif level == 1:
            if not visible:
                return None
            chars = list(text)
            for index in rng.sample(visible, min(len(visible), rng.randint(1, 3))):
                char = chars[index]
                pool = string.digits if char.isdigit() else string.ascii_uppercase if char.isupper() else string.ascii_lowercase
                chars[index] = rng.choice(pool.replace(char, ""))
            faulty = "".join(chars)
        elif level == 2:
            if spec.get("variant") == "hme":
                tokens = [match for match in re.finditer(r"[A-Za-z]+|\d+", text[start:end])
                          if all(start + index not in command_positions
                                 for index in range(match.start(), match.end()))]
                if not tokens:
                    return None
                token = rng.choice(tokens)
                old = token.group()
                replacement = str(int(old) + rng.randint(1, 9)) if old.isdigit() else rng.choice(("x", "y", "z", "a", "b"))
                if replacement == old:
                    continue
                left, right = start + token.start(), start + token.end()
            else:
                old = text[start:end]
                if re.fullmatch(r"\d+(?:[.,]\d+)?", old.strip()):
                    replacement = str(int(float(old.replace(",", ""))) + rng.randint(1, 9))
                else:
                    other_words = [word for word in re.findall(r"[A-Za-z]+", text[:start] + " " + text[end:])
                                   if _normalized(word) not in {_normalized(value) for value in answers}]
                    replacement = rng.choice(other_words or ["London", "Monday", "silver", "pending"])
                left, right = start, end
            faulty = text[:left] + replacement + text[right:]
        else:
            raise ValueError(f"Unsupported OCR level: {level}")
        if faulty != text and re.findall(r"\\[A-Za-z]+", faulty) == commands and all(
            _normalized(answer) not in _normalized(faulty) for answer in answers if _normalized(answer)
        ):
            return {**observation, "text": faulty}
    return None


def _count_fault(observation: dict, level: int, rng: random.Random, spec: dict) -> dict | None:
    count = observation.get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        return None
    truth = int(spec["count_truth"])
    if level == 0:
        candidates = [count + change for change in (-2, -1, 1, 2)]
    elif level == 1:
        candidates = [round(count * factor) for factor in (0.67, 1.5)]
    elif level == 2:
        candidates = [round(count * factor) for factor in (0.5, 2.0)]
    elif level == 3:
        candidates = [0, max(5, 5 * max(count, truth))]
    else:
        raise ValueError(f"Unsupported count level: {level}")
    candidates = [value for value in candidates if value >= 0 and value not in {count, truth}]
    return {**observation, "count": rng.choice(candidates)} if candidates else None


def _grounding_fault(observation: dict, level: int, rng: random.Random) -> dict | None:
    boxes = observation.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        return None
    if level == 2:
        return {**observation, "boxes": [], "confidence": [], "labels": []}
    if level == 1:
        if len(boxes) < 2:
            return None
        shifted = list(boxes)
        shifted[0], shifted[1] = shifted[1], shifted[0]
        return {**observation, "boxes": shifted} if shifted != boxes else None
    if level != 0:
        raise ValueError(f"Unsupported grounding level: {level}")
    shifted = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            return None
        x1, y1, x2, y2 = box
        dx = max(-x1, min(1000 - x2, rng.choice((-40, -20, 20, 40))))
        dy = max(-y1, min(1000 - y2, rng.choice((-40, -20, 20, 40))))
        shifted.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    return {**observation, "boxes": shifted} if shifted != boxes else None


def inject_fault(observation: Any, spec: dict) -> dict | None:
    """Return a schema-preserving wrong observation, or None if the first call cannot be faulted."""
    if not isinstance(observation, dict) or observation.get("status") in {"error", "failed"}:
        return None
    rng = random.Random(spec["seed"])
    tool, level = spec["tool"], spec["level"]
    if tool == "ocr_read":
        return _ocr_fault(observation, level, rng, spec)
    if tool == "object_count":
        return _count_fault(observation, level, rng, spec)
    if tool == "grounding_detect":
        return _grounding_fault(observation, level, rng)
    raise ValueError(f"Unsupported online fault tool: {tool}")
