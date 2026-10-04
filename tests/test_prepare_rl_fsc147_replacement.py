from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fsc147_replacement", ROOT / "scripts/prepare_rl_fsc147_replacement.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Fsc147ReplacementTest(unittest.TestCase):
    def fixture(self, root):
        base, fsc = root / "base", root / "fsc"
        base.mkdir()
        (fsc / "images_384_VarV2").mkdir(parents=True)
        rows = []
        for index, source in enumerate(("visual-agent-tallyqa", "visual-agent-tallyqa", "visual-agent-ocr")):
            rows.append({
                "images": ["/base.png"], "source_image": "/base.png", "question": "Original question",
                "solution": "3", "bbox": [], "source_index": index, "data_source": source,
                "ability": "counting", "split": "train", "uid": f"base_{index}",
                "original_source": "original", "image_digest": "", "answer_aliases": ["3"], "transform": "",
            })
        pq.write_table(pa.Table.from_pylist(rows), base / "train.parquet")
        pq.write_table(pa.Table.from_pylist([{**rows[0], "split": "val"}]), base / "val.parquet")
        (base / "val.jsonl").write_text(json.dumps({**rows[0], "split": "val"}) + "\n")
        splits = {"train": ["7.jpg", "8.jpg"], "val": ["9.jpg"], "test": ["10.jpg"]}
        annotations = {name: {"points": [[1, 2]] * count} for name, count in (("7.jpg", 13), ("8.jpg", 101), ("9.jpg", 999), ("10.jpg", 999))}
        (fsc / "Train_Test_Val_FSC_147.json").write_text(json.dumps(splits))
        (fsc / "annotation_FSC147_384.json").write_text(json.dumps(annotations))
        (fsc / "ImageClasses_FSC147.txt").write_text("7.jpg\tsea shells\n8.jpg\tapples\n9.jpg\tcars\n10.jpg\tpeople\n")
        for name in splits["train"]:
            (fsc / "images_384_VarV2" / name).write_bytes(b"fixture")
        return base, fsc, rows

    def test_replacement_uses_only_training_labels_and_preserves_other_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, fsc, original = self.fixture(root)
            before = {name: (base / name).read_bytes() for name in ("train.parquet", "val.parquet", "val.jsonl")}
            output = root / "output"
            manifest = prepare.prepare(base, fsc, output, fsc_count=2)
            rows = pq.read_table(output / "train.parquet").to_pylist()
            self.assertEqual(Counter(row["data_source"] for row in rows), {"visual-agent-fsc147": 2, "visual-agent-tallyqa": 1, "visual-agent-ocr": 1})
            self.assertIn(original[-1], rows)
            added = [row for row in rows if row["data_source"] == "visual-agent-fsc147"]
            self.assertEqual({row["solution"] for row in added}, {"13", "101"})
            self.assertEqual({Path(row["images"][0]).name for row in added}, {"7.jpg", "8.jpg"})
            self.assertTrue(all(row["bbox"] == [] and row["split"] == "train" for row in added))
            self.assertEqual(manifest["removed_tallyqa_rows"], 1)
            self.assertEqual(manifest["retained_tallyqa_rows"], 1)
            self.assertEqual(manifest["added_fsc147_rows"], 2)
            retained = [row for row in rows if row["data_source"] == "visual-agent-tallyqa"]
            removed = [json.loads(line) for line in (output / "removed_tallyqa.jsonl").read_text().splitlines()]
            self.assertEqual({row["uid"] for row in retained + removed}, {row["uid"] for row in original[:2]})
            self.assertNotEqual(retained[0]["uid"], removed[0]["uid"])
            for name, payload in before.items():
                self.assertEqual((base / name).read_bytes(), payload)
                if name != "train.parquet":
                    self.assertEqual((output / name).read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                prepare.prepare(base, fsc, output)

    def test_random_subsets_are_reproducible_and_preserve_official_source_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, fsc, _ = self.fixture(root)
            results = []
            for name in ("first", "second"):
                output = root / name
                prepare.prepare(base, fsc, output, fsc_count=1)
                rows = pq.read_table(output / "train.parquet").to_pylist()
                added = [row for row in rows if row["data_source"] == "visual-agent-fsc147"]
                self.assertEqual(len(added), 1)
                official = json.loads((fsc / "Train_Test_Val_FSC_147.json").read_text())["train"]
                self.assertEqual(Path(added[0]["images"][0]).name, official[added[0]["source_index"]])
                results.append(rows)
            self.assertEqual(results[0], results[1])

    def test_rejects_training_and_test_image_overlap_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, fsc, _ = self.fixture(root)
            split_path = fsc / "Train_Test_Val_FSC_147.json"
            splits = json.loads(split_path.read_text())
            splits["test"].append("7.jpg")
            split_path.write_text(json.dumps(splits))
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "held-out"):
                prepare.prepare(base, fsc, output)
            self.assertFalse(output.exists())

    def test_fsc147_is_registered_and_counting_reward_has_no_tolerance(self):
        parent = ModuleType("verl.utils.dataset.zwz_original_relation_dataset")

        class ParentDataset:
            def __getitem__(self, item):
                return {"extra_info": {}}

        parent.ZwzOriginalRelationDataset = ParentDataset
        utils = ROOT / "reinforcement_learning/verl/utils"
        dataset_spec = importlib.util.spec_from_file_location("fsc147_dataset_test", utils / "dataset/zwz_deepeyesv2_dataset.py")
        dataset_module = importlib.util.module_from_spec(dataset_spec)
        with patch.dict(sys.modules, {parent.__name__: parent}):
            dataset_spec.loader.exec_module(dataset_module)
        dataset = dataset_module.ZwzDeepEyesV2Dataset()
        dataset.dataframe = [{
            "data_source": "visual-agent-fsc147", "split": "train", "solution": "101",
            "question": "Count every instance of apples in the image.", "ability": "counting",
            "source_image": "/8.jpg", "uid": "fsc147_train_8", "original_source": "fsc147",
        }]
        row = dataset[0]
        self.assertEqual(row["data_source"], "visual-agent-fsc147")
        self.assertEqual(row["reward_model"]["ground_truth"], "101")
        self.assertEqual(row["extra_info"]["original_source"], "fsc147")
        self.assertEqual(row["extra_info"]["uid"], "fsc147_train_8")
        import_utils = ModuleType("verl.utils.import_utils")
        import_utils.deprecated = lambda name: lambda function: function
        name = "_fsc147_registered_rewards"
        registry_spec = importlib.util.spec_from_file_location(name, utils / "reward_score/__init__.py", submodule_search_locations=[str(utils / "reward_score")])
        registry = importlib.util.module_from_spec(registry_spec)
        reward_spec = importlib.util.spec_from_file_location(name + ".visual_agent_thyme", utils / "reward_score/visual_agent_thyme.py")
        reward = importlib.util.module_from_spec(reward_spec)
        reward_spec.loader.exec_module(reward)
        modules = {import_utils.__name__: import_utils, name: registry, reward_spec.name: reward}
        with patch.dict(sys.modules, modules), patch.object(reward, "judge_match", return_value=False) as judge:
            registry_spec.loader.exec_module(registry)
            for answer, expected in (("101", 1), ("100", 0)):
                score = registry.default_compute_score("visual-agent-fsc147", f"<answer>{answer}</answer>", "101", row["extra_info"])
                self.assertEqual(score["acc"], expected)
            judge.assert_called_once()
