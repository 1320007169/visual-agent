"""VERL runtime adapter for zwz_rl_vqa original-image relation data."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask


DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parents[4] / "prompts/visual_agent_rl_system.txt"
SYSTEM_PROMPT_FILE = Path(os.environ.get("VISUAL_AGENT_RL_SYSTEM_PROMPT_FILE", DEFAULT_SYSTEM_PROMPT_FILE))
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
DUAL_STREAM_ENABLED = os.environ.get("VISUAL_AGENT_DUAL_STREAM", "0") == "1"
DEFAULT_NATIVE_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parents[4] / "prompts/visual_agent_native_system.txt"
NATIVE_SYSTEM_PROMPT_FILE = Path(
    os.environ.get("VISUAL_AGENT_NATIVE_SYSTEM_PROMPT_FILE", DEFAULT_NATIVE_SYSTEM_PROMPT_FILE)
)
NATIVE_SYSTEM_PROMPT = NATIVE_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
IMAGE_TRANSPORT_MODE = os.environ.get("VISUAL_AGENT_IMAGE_TRANSPORT", "pil_png")
SUPPORTED_IMAGE_TRANSPORT_MODES = {"pil_png", "source_cached"}

if IMAGE_TRANSPORT_MODE not in SUPPORTED_IMAGE_TRANSPORT_MODES:
    raise ValueError(
        f"Unsupported VISUAL_AGENT_IMAGE_TRANSPORT={IMAGE_TRANSPORT_MODE!r}; "
        f"expected one of {sorted(SUPPORTED_IMAGE_TRANSPORT_MODES)}"
    )


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
        source_image_values = list(self.dataframe[item].get(self.image_key) or [])
        row = super().__getitem__(item)
        if DUAL_STREAM_ENABLED:
            self._add_native_prompt(row)
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
            "index": item,
            "question": question,
            "answer": solution,
            "bbox": row.get("bbox"),
            "source": "zwz_rl_vqa/original_images",
        }
        row["index"] = item
        return row

    def _add_native_prompt(self, row: dict[str, Any]) -> None:
        """Build a tool-free prompt with the same user question and image."""
        if self.processor is None:
            raise RuntimeError(
                "dual-stream visual training requires a multimodal processor; "
                "check that the RL environment supports Qwen3VLProcessor"
            )
        native_messages = copy.deepcopy(row["raw_prompt"])
        if native_messages and native_messages[0].get("role") == "system":
            native_messages[0]["content"] = NATIVE_SYSTEM_PROMPT
        else:
            native_messages.insert(0, {"role": "system", "content": NATIVE_SYSTEM_PROMPT})

        native_prompt = self._apply_chat_template(native_messages, tools_enabled=False)
        multi_modal_data = row.get("multi_modal_data", {})
        model_inputs = self.processor(
            text=[native_prompt],
            images=multi_modal_data.get("image"),
            videos=multi_modal_data.get("video"),
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        second_per_grid_ts = model_inputs.pop("second_per_grid_ts", None)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = torch.stack(
                [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask[0],
                    )
                ]
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["native_input_ids"] = input_ids[0]
        row["native_attention_mask"] = attention_mask[0]
        row["native_position_ids"] = position_ids[0]
        row["native_raw_prompt"] = native_messages


class VisionOpdDataset(ZwzOriginalRelationDataset):
    """Use clean, unmarked Vision-OPD images for Agent/Native dual-stream RL."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        source_row = self.dataframe[item]
        source_image_values = list(source_row.get(self.image_key) or [])
        row = RLHFDataset.__getitem__(self, item)
        if DUAL_STREAM_ENABLED:
            self._add_native_prompt(row)
        if IMAGE_TRANSPORT_MODE == "source_cached" and source_image_values:
            invalid_paths = [value for value in source_image_values if not isinstance(value, str)]
            if invalid_paths:
                raise TypeError("source_cached image transport requires filesystem image paths")
            row["origin_multi_modal_data"]["image"] = [
                str(Path(value).expanduser()) for value in source_image_values
            ]

        question = str(row.get("question", "")).strip()
        solution = str(row.get("solution", "")).strip().upper()
        if solution not in {"A", "B", "C", "D"}:
            raise ValueError(f"Vision-OPD solution must be A, B, C, or D; got {solution!r}")
        row["data_source"] = "visual-agent-vision-opd"
        row["ability"] = "fine_grained_vqa"
        row["reward_model"] = {"style": "rule", "ground_truth": solution}
        row["extra_info"] = {
            "split": "train",
            "index": item,
            "question": question,
            "answer": solution,
            "answer_text": row.get("answer_text"),
            "bbox": row.get("bbox"),
            "source": "vision-opd/original_images",
            "data_source": "visual-agent-vision-opd",
            "source_image": row.get("source_image"),
        }
        row["index"] = item
        return row
