#!/usr/bin/env python3
"""Render top-level runtime overrides into a LLaMA-Factory YAML config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, TypeAlias


Scalar: TypeAlias = str | int | float | bool | None
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCALAR_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s.*)?$")


def format_yaml_scalar(value: Scalar) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def render_config(source: str, overrides: Mapping[str, Scalar]) -> str:
    for key in overrides:
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"invalid override key: {key}")

    remaining = dict(overrides)
    rendered: list[str] = []
    for line in source.splitlines():
        match = SCALAR_LINE_RE.fullmatch(line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            rendered.append(f"{key}: {format_yaml_scalar(remaining.pop(key))}")
        else:
            rendered.append(line)

    if remaining:
        rendered.extend(("", "### runtime overrides"))
        for key, value in remaining.items():
            rendered.append(f"{key}: {format_yaml_scalar(value)}")

    return "\n".join(rendered) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--set", dest="strings", action="append", nargs=2, default=[])
    parser.add_argument("--set-int", dest="integers", action="append", nargs=2, default=[])
    parser.add_argument("--set-null", dest="nulls", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Scalar] = dict(args.strings)
    for key, value in args.integers:
        overrides[key] = int(value)
    for key in args.nulls:
        overrides[key] = None

    source = args.source.read_text(encoding="utf-8")
    args.destination.write_text(render_config(source, overrides), encoding="utf-8")


if __name__ == "__main__":
    main()
