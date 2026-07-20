"""VERL runtime adapter for zwz_rl_vqa original-image relation data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parents[4] / "prompts/visual_agent_rl_system.txt"
SYSTEM_PROMPT_FILE = Path(os.environ.get("VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE", DEFAULT_SYSTEM_PROMPT_FILE))
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()


class ZwzOriginalRelationDataset(RLHFDataset):
    """Use unboxed original images and clean spatial-relation questions."""

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        example.pop(self.prompt_key, None)
        question = str(example.get("question", "")).strip()
        images = example.get(self.image_key) or []
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        question = str(row.get("question", "")).strip()
        solution = str(row.get("solution", "")).strip().lower()
        row["data_source"] = "visual-agent-zwz-relation"
        row["ability"] = "spatial_relation"
        row["reward_model"] = {"style": "rule", "ground_truth": solution}
        row["extra_info"] = {
            "split": "train",
            "index": item,
            "question": question,
            "answer": solution,
            "bbox": row.get("bbox"),
            "source": "zwz_rl_vqa/original_images",
        }
        row["index"] = item
        return row
