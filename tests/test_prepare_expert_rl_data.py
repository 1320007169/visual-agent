import sys
from collections import Counter
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_expert_rl_data import row, select, split_for, write_dataset


class ExpertDataSelectionTests(unittest.TestCase):
    def test_excludes_benchmarks_duplicates_and_conflicting_labels(self):
        candidates = [
            row("unused.png", question, answer, str(index), "source", "visual-agent-ocr", "text_reading", index,
                image_digest=digest, split="train")
            for index, (digest, question, answer) in enumerate([
                ("heldout", "read", "text"),
                ("conflict", "read", "one"), ("conflict", "read", "two"),
                ("valid", "read", "correct"), ("valid", "read", "correct"),
                ("other", "read", "correct"),
            ])
        ]
        stats = Counter()
        selected = select(candidates, {("train", "source"): 2}, {"heldout"}, stats, fixed_split=True)
        self.assertEqual({r["image_digest"] for r in selected["train"]}, {"valid", "other"})
        self.assertFalse(selected["val"])

    def test_image_split_keeps_multiple_questions_together(self):
        digests = [str(index) for index in range(200)]
        train_digest = next(d for d in digests if split_for(d) == "train")
        val_digest = next(d for d in digests if split_for(d) == "val")
        candidates = [
            row("unused.png", question, "answer", f"{digest}_{question}", "source", "visual-agent-ocr", "text_reading", 0,
                image_digest=digest)
            for digest in (train_digest, val_digest) for question in ("first", "second")
        ]
        selected = select(candidates, {("train", "source"): 2, ("val", "source"): 2}, set(), Counter())
        self.assertEqual({r["image_digest"] for r in selected["train"]}, {train_digest})
        self.assertEqual({r["image_digest"] for r in selected["val"]}, {val_digest})

    def test_does_not_repeat_samples_to_fill_a_shortage(self):
        with self.assertRaisesRegex(ValueError, "Insufficient eligible samples"):
            select([], {("train", "source"): 1}, set(), Counter())

    def test_rejects_identical_pixels_in_different_split_files_before_export(self):
        train = row("train.jpg", "count geese", "33", "train", "fsc147", "visual-agent-fsc147", "counting", 0,
                    image_digest="same_decoded_pixels")
        val = row("val.jpg", "count seagulls", "33", "val", "fsc147", "visual-agent-fsc147", "counting", 1,
                  image_digest="same_decoded_pixels", split="val")
        with self.assertRaisesRegex(ValueError, "Train/val image overlap: 1"):
            write_dataset(Path("unused_output"), {"train": [train], "val": [val]}, Counter(), [], [])


if __name__ == "__main__":
    unittest.main()
