import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


TRAINER_PATH = (
    Path(__file__).parents[1]
    / "reinforcement_learning/verl/trainer/ppo/ray_trainer.py"
)


def load_dump_methods():
    # Exercise the real logging methods without importing the Ray/GPU stack.
    module = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    trainer = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    methods = {"_generation_source_metadata", "_dump_generations"}
    trainer.body = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {"np": np, "json": json, "os": os}
    exec(compile(ast.Module(body=[trainer], type_ignores=[]), str(TRAINER_PATH), "exec"), namespace)
    return namespace["RayPPOTrainer"]


Trainer = load_dump_methods()


class Batch:
    def __init__(self, rows):
        self.rows = rows
        self.non_tensor_batch = {
            key: np.array([row.get(key) for row in rows], dtype=object)
            for key in set().union(*(row.keys() for row in rows))
        }

    def __len__(self):
        return len(self.rows)


class RolloutSourceMetadataTest(unittest.TestCase):
    def test_source_identity_follows_repeated_reordered_rows(self):
        first = {
            "source_index": np.int64(55062),
            "index": 0,
            "source_image": "image-a.jpeg",
            "question": "Where is the cup?",
            "solution": "left of",
            "reward_model": {"ground_truth": "left of"},
        }
        second = {
            "source_index": np.int64(88019),
            "index": 1,
            "source_image": "image-b.jpeg",
            "question": "Where is the cup?",
            "solution": "right of",
            "reward_model": {"ground_truth": "right of"},
        }
        batch = Batch([second, first, first, second])

        metadata = Trainer._generation_source_metadata(batch, "train")

        self.assertEqual([row["source_index"] for row in metadata], [88019, 55062, 55062, 88019])
        self.assertEqual([row["source_image"] for row in metadata], ["image-b.jpeg", "image-a.jpeg", "image-a.jpeg", "image-b.jpeg"])
        self.assertEqual([row["ground_truth"] for row in metadata], ["right of", "left of", "left of", "right of"])
        self.assertTrue(all(type(row["source_index"]) is int for row in metadata))
        self.assertTrue(all("index" not in row for row in metadata))

    def test_allowlist_omits_images_and_does_not_invent_source_id(self):
        metadata = Trainer._generation_source_metadata(
            Batch([{
                "index": 7,
                "source_image": "data:image/png;base64,not-an-image-path",
                "images": [b"image bytes"],
                "extra_info": {"index": 7, "question": "Which side?", "api_key": "private"},
                "solution": "below",
            }]),
            "validation",
        )

        self.assertEqual(metadata, [{"dataset_split": "validation", "question": "Which side?", "ground_truth": "below"}])

    def test_dump_roundtrip_keeps_source_and_output_together(self):
        trainer = Trainer()
        trainer.global_steps = 12
        metadata = [
            {"source_index": 20, "source_image": "b.jpeg"},
            {"source_index": 10, "source_image": "a.jpeg"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trainer._dump_generations(
                ["same question", "same question"], ["right of", "left of"],
                [1.0, 1.0], {"acc": [1.0, 1.0], "source_metadata": ["untrusted", "untrusted"]}, directory,
                source_metadata=metadata,
            )
            rows = [json.loads(line) for line in (Path(directory) / "12.jsonl").read_text().splitlines()]
        self.assertEqual([(row["source_metadata"]["source_index"], row["output"]) for row in rows], [(20, "right of"), (10, "left of")])

    def test_dump_without_metadata_is_backward_compatible(self):
        trainer = Trainer()
        trainer.global_steps = 1
        with tempfile.TemporaryDirectory() as directory:
            trainer._dump_generations(["question"], ["answer"], [1.0], {}, directory)
            row = json.loads((Path(directory) / "1.jsonl").read_text())
        self.assertEqual(row, {"input": "question", "output": "answer", "score": 1.0, "step": 1})

    def test_metadata_count_mismatch_does_not_write_a_dump(self):
        trainer = Trainer()
        trainer.global_steps = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "metadata count"):
                trainer._dump_generations(["question"], ["answer"], [1.0], {}, directory, source_metadata=[])
            self.assertFalse((Path(directory) / "1.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
