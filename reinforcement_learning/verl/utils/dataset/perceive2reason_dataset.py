"""Perceive2Reason mixed dataset for visual agent RL training.

Mixes P2R samples (fixed perception prefix) with standard training samples.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parents[4] / "prompts/visual_agent_rl_system_groundingdino.txt"
SYSTEM_PROMPT_FILE = Path(os.environ.get("VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE", DEFAULT_SYSTEM_PROMPT_FILE))
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
IMAGE_TRANSPORT_MODE = os.environ.get("VISUAL_AGENT_IMAGE_TRANSPORT", "pil_png")
SUPPORTED_IMAGE_TRANSPORT_MODES = {"pil_png", "source_cached"}

if IMAGE_TRANSPORT_MODE not in SUPPORTED_IMAGE_TRANSPORT_MODES:
    raise ValueError(
        f"Unsupported VISUAL_AGENT_IMAGE_TRANSPORT={IMAGE_TRANSPORT_MODE!r}; "
        f"expected one of {sorted(SUPPORTED_IMAGE_TRANSPORT_MODES)}"
    )


class Perceive2ReasonDataset(RLHFDataset):
    """Mixed dataset: P2R samples (fixed prefix) + standard samples.

    Environment variables:
        PERCEIVE2REASON_DATA_PATH: Path to p2r_train.jsonl
        PERCEIVE2REASON_MIX_RATIO: Fraction of P2R samples (default 0.3)
        STANDARD_TRAIN_DATA_PATH: Path to standard training data
    """

    def __init__(self, **kwargs):
        # Load P2R data
        p2r_path = os.environ.get("PERCEIVE2REASON_DATA_PATH", "")
        if not p2r_path or not Path(p2r_path).exists():
            raise FileNotFoundError(
                f"P2R data not found: {p2r_path}. "
                "Set PERCEIVE2REASON_DATA_PATH to p2r_train.jsonl"
            )

        # Use the files supplied by the trainer so train and validation datasets
        # remain independent. Keep the environment variable as a standalone
        # fallback for direct construction.
        standard_path = kwargs.pop("standard_data_path", None)
        if standard_path is None:
            standard_path = kwargs.pop("data_files", None)
        if standard_path is None:
            standard_path = os.environ.get("STANDARD_TRAIN_DATA_PATH", "")
        if not standard_path:
            raise ValueError(
                "Standard training data path required. "
                "Set STANDARD_TRAIN_DATA_PATH or pass data_files"
            )
        data_config = kwargs.get("config")
        is_validation = data_config is not None and standard_path == data_config.get("val_files")

        mix_ratio = float(os.environ.get("PERCEIVE2REASON_MIX_RATIO", "0.3"))
        if not 0 < mix_ratio < 1:
            raise ValueError(f"PERCEIVE2REASON_MIX_RATIO must be in (0, 1), got {mix_ratio}")

        # Initialize parent with standard data
        super().__init__(data_files=standard_path, **kwargs)

        # Load P2R samples
        self.p2r_samples = [] if is_validation else self._load_p2r_data(p2r_path)
        self.mix_ratio = mix_ratio

        # Build mixed index: maps global index to (source, local_index)
        # source: 'p2r' or 'standard'
        self.mixed_index = self._build_mixed_index()

        print(f"Perceive2ReasonDataset initialized:")
        print(f"  P2R samples: {len(self.p2r_samples)}")
        print(f"  Standard samples: {len(self.dataframe)}")
        print(f"  Mixed total: {len(self.mixed_index)}")
        print(f"  P2R ratio: {mix_ratio:.1%}")

    def _load_p2r_data(self, path: str) -> list[dict[str, Any]]:
        """Load P2R training data from JSONL."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _build_mixed_index(self) -> list[tuple[str, int]]:
        """Build mixed index with desired P2R ratio.

        Returns list of (source, local_idx) tuples.
        """
        n_p2r = len(self.p2r_samples)
        n_standard = len(self.dataframe)

        # Calculate target counts to achieve mix_ratio
        # mix_ratio = n_p2r_selected / (n_p2r_selected + n_standard_selected)
        # Solve for n_p2r_selected given we want to use all standard samples
        n_standard_selected = n_standard
        n_p2r_selected = int(n_standard_selected * self.mix_ratio / (1 - self.mix_ratio))

        # Limit to available P2R samples
        n_p2r_selected = min(n_p2r_selected, n_p2r)

        # Build index
        index = []
        index.extend([('p2r', i) for i in range(n_p2r_selected)])
        index.extend([('standard', i) for i in range(n_standard_selected)])

        # Shuffle for random mixing
        random.shuffle(index)

        return index

    def __len__(self) -> int:
        return len(self.mixed_index)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build messages for standard samples (reuse parent logic)."""
        example.pop(self.prompt_key, None)
        question = str(example.get("question", "")).strip()
        images = example.get(self.image_key) or []
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_messages_p2r(self, p2r_sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Build messages for P2R samples."""
        question = str(p2r_sample.get("question", "")).strip()
        images = p2r_sample.get("image", [])
        if isinstance(images, str):
            images = [images]

        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def __getitem__(self, item: int) -> dict[str, Any]:
        source, local_idx = self.mixed_index[item]

        if source == 'standard':
            # Standard sample: use parent logic
            source_image_values = list(self.dataframe[local_idx].get(self.image_key) or [])
            # Call parent's __getitem__ with local_idx
            row = super().__getitem__(local_idx)

            if IMAGE_TRANSPORT_MODE == "source_cached" and source_image_values:
                invalid_paths = [value for value in source_image_values if not isinstance(value, str)]
                if invalid_paths:
                    raise TypeError("source_cached image transport requires filesystem image paths")
                row["origin_multi_modal_data"]["image"] = [
                    str(Path(value).expanduser()) for value in source_image_values
                ]

            question = str(row.get("question", "")).strip()
            solution = str(row.get("solution", "")).strip().lower()
            row["data_source"] = "visual-agent-zwz-relation"
            row["ability"] = "spatial_relation"
            row["reward_model"] = {"style": "rule", "ground_truth": solution}
            row["extra_info"] = {
                "split": "train",
                "index": local_idx,
                "question": question,
                "answer": solution,
                "bbox": row.get("bbox"),
                "source": "zwz_rl_vqa/original_images",
                "sample_type": "standard",
            }
            row["index"] = item
            # No perception_prefix for standard samples

        else:  # source == 'p2r'
            # P2R sample: build from p2r_samples
            p2r_sample = self.p2r_samples[local_idx]

            # Build row structure
            messages = self._build_messages_p2r(p2r_sample)
            images = p2r_sample.get("image", [])
            if isinstance(images, str):
                images = [images]

            row = {
                "prompt": messages,
                "data_source": "visual-agent-perceive2reason",
                "ability": "reasoning_given_perception",
                "index": item,
            }

            # Image handling
            if IMAGE_TRANSPORT_MODE == "source_cached":
                row["origin_multi_modal_data"] = {"image": [str(Path(img).expanduser()) for img in images]}
            elif IMAGE_TRANSPORT_MODE == "pil_png":
                # Parent class handles PIL loading
                row[self.image_key] = images

            # Reward model
            gt_answer = str(p2r_sample.get("gt_answer", "")).strip().lower()
            row["reward_model"] = {"style": "rule", "ground_truth": gt_answer}

            # Extra info
            row["extra_info"] = {
                "split": "train",
                "index": local_idx,
                "question": p2r_sample.get("question", ""),
                "answer": gt_answer,
                "source": "perceive2reason",
                "sample_type": "p2r",
                "num_tool_calls": p2r_sample.get("num_tool_calls", 0),
            }

            # **KEY**: Add perception_prefix for rollout worker
            row["perception_prefix"] = p2r_sample.get("perception_prefix", [])

        return row
