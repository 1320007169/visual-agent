#!/usr/bin/env python3
"""Rejudge old relation-query suspects with the answer-leakage prompt."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MIXED_JUDGE = SCRIPT_DIR.parent / "mixed_rl_16gpu_and_64gpu_new_unique/judge_with_deepseek_v41.py"
SPEC = importlib.util.spec_from_file_location("mixed_deepseek_visual_judge", MIXED_JUDGE)
mixed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mixed)

judge = mixed.judge
judge.DEFAULT_INPUT = SCRIPT_DIR / "data/suspicious834_original_only.jsonl"
judge.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/deepseek_v41"
judge.SOURCE_ROLLOUT_DIR = str(
    Path("/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/rollouts")
    / "visual-agent-zwz-rl"
    / "zwz_original_relation_qwen3_base_2node_groundingdino_kl_safe_manual"
)
judge.SOURCE_ROLLOUT_STEPS = [1, 165]
judge.PROMPT_VERSION = "deepseek-v41-relation-query-leakage-rejudge-v1"
judge.trajectory_text = mixed.mixed_trajectory_text


if __name__ == "__main__":
    judge.main()
