import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mme_candidates", ROOT / "scripts/prepare_rl_mme_candidates.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class MmeCandidatesTest(unittest.TestCase):
    def test_sampling_preserves_base_data_answers_options_and_provenance(self):
        root = Path(tempfile.mkdtemp(prefix="mme-candidates-test-"))
        base, candidates = root / "base", root / "candidates"
        base.mkdir()
        candidates.mkdir()
        original = {
            "images": ["/original.jpg"], "source_image": "/original.jpg", "question": "Original",
            "solution": "3", "bbox": [], "source_index": 0, "data_source": "visual-agent-fsc147",
            "ability": "counting", "split": "train", "uid": "original", "original_source": "fsc147",
            "image_digest": "", "answer_aliases": ["3"], "transform": "",
        }
        pq.write_table(pa.Table.from_pylist([original]), base / "train.parquet")
        pq.write_table(pa.Table.from_pylist([{**original, "split": "val"}]), base / "val.parquet")
        (base / "val.jsonl").write_text(json.dumps({**original, "split": "val"}) + "\n")
        records = []
        for index in range(6):
            (candidates / f"{index}.jpg").write_bytes(b"fixture")
            driving = index < 2
            raw = ({"questions": "Can I turn?" if index == 0 else ["Can I turn?"],
                    "possible_answers": {"A": "Yes", "B": "No"}, "true_answers": ["B"],
                    "has_multiple_questions": False, "explanation": "SECRET_EXPLANATION"}
                   if driving else {"question": "Count the cars.", "answer": "12",
                                    "annotations": ["SECRET_ANNOTATION"]})
            records.append({"source": "DrivingVQA" if driving else "GRAID-BDD", "split": "train",
                            "source_id": str(index), "source_shard": "train.parquet", "source_row": index,
                            "image_path": f"{index}.jpg", "image_sha256": "fixture",
                            "question_type": "single_question_single_answer" if driving else "HowMany",
                            "raw": raw})
        candidate_file = candidates / "candidates_5000.jsonl"
        candidate_file.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        before = {name: (base / name).read_bytes() for name in ("train.parquet", "val.parquet", "val.jsonl")}
        output = root / "output"
        manifest = prepare.prepare(base, candidates, output, count=4)
        selected_indices = random.Random(20261009).sample(range(6), 4)
        self.assertEqual(manifest["selected_indices"], selected_indices)
        rows = pq.read_table(output / "train.parquet").to_pylist()
        self.assertEqual(rows[0], original)
        self.assertEqual(len(rows), 5)
        for row, index in zip(rows[1:], selected_indices):
            self.assertEqual(row["source_index"], index)
            self.assertEqual(row["images"], [str(candidates / f"{index}.jpg")])
            self.assertEqual(row["bbox"], [])
            self.assertNotIn("SECRET", row["question"])
            if index < 2:
                self.assertIn("A. Yes\nB. No", row["question"])
                self.assertEqual(row["solution"], "B")
                self.assertEqual(row["ability"], "multiple_choice")
            else:
                self.assertEqual(row["solution"], "12")
                self.assertEqual(row["ability"], "counting")
        selected = [json.loads(line) for line in (output / "selected_candidates.jsonl").read_text().splitlines()]
        self.assertEqual(selected, [records[index] for index in selected_indices])
        for name, content in before.items():
            self.assertEqual((base / name).read_bytes(), content)
            if name != "train.parquet":
                self.assertEqual((output / name).read_bytes(), content)
        repeated = root / "repeated"
        prepare.prepare(base, candidates, repeated, count=4)
        self.assertEqual(pq.read_table(repeated / "train.parquet").to_pylist(), rows)
        with self.assertRaises(FileExistsError):
            prepare.prepare(base, candidates, output, count=4)
        records[selected_indices[0]]["split"] = "test"
        candidate_file.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        with self.assertRaisesRegex(ValueError, "official training split"):
            prepare.prepare(base, candidates, root / "invalid", count=4)
        self.assertFalse((root / "invalid").exists())

    def test_new_sources_reach_dataset_and_reward_with_correct_answer_semantics(self):
        utils = ROOT / "reinforcement_learning/verl/utils"
        parent = ModuleType("verl.utils.dataset.zwz_original_relation_dataset")

        class ParentDataset:
            def __getitem__(self, item):
                return {"extra_info": {}}

        parent.ZwzOriginalRelationDataset = ParentDataset
        dataset_spec = importlib.util.spec_from_file_location("mme_dataset", utils / "dataset/zwz_deepeyesv2_dataset.py")
        dataset_module = importlib.util.module_from_spec(dataset_spec)
        with patch.dict(sys.modules, {parent.__name__: parent}):
            dataset_spec.loader.exec_module(dataset_module)
        import_utils = ModuleType("verl.utils.import_utils")
        import_utils.deprecated = lambda name: lambda function: function
        name = "_mme_registered_rewards"
        registry_spec = importlib.util.spec_from_file_location(name, utils / "reward_score/__init__.py",
                                                              submodule_search_locations=[str(utils / "reward_score")])
        registry = importlib.util.module_from_spec(registry_spec)
        reward_spec = importlib.util.spec_from_file_location(name + ".visual_agent_thyme", utils / "reward_score/visual_agent_thyme.py")
        reward = importlib.util.module_from_spec(reward_spec)
        reward_spec.loader.exec_module(reward)
        modules = {import_utils.__name__: import_utils, name: registry, reward_spec.name: reward}
        with patch.dict(sys.modules, modules), patch.object(reward, "judge_match", return_value=False):
            registry_spec.loader.exec_module(registry)
            for source, ability, answer, predictions in (
                ("visual-agent-drivingvqa", "multiple_choice", "B", (("B", 1), ("B. No", 1), ("A", 0))),
                ("visual-agent-graid-bdd", "multiple_choice", "A", (("A", 1), ("B", 0))),
                ("visual-agent-graid-bdd", "counting", "101", (("101", 1), ("100", 0))),
                ("visual-agent-graid-bdd", "perception", "car", (("car", 1), ("bus", 0))),
            ):
                dataset = dataset_module.ZwzDeepEyesV2Dataset()
                dataset.dataframe = [{"data_source": source, "solution": answer, "question": "Question",
                                      "ability": ability, "source_image": "/image.jpg", "uid": "sample",
                                      "original_source": "official_train"}]
                row = dataset[0]
                self.assertEqual(row["reward_model"]["ground_truth"], answer)
                self.assertEqual(row["extra_info"]["uid"], "sample")
                self.assertEqual(row["extra_info"]["ability"], ability)
                for prediction, expected in predictions:
                    result = registry.default_compute_score(source, f"<answer>{prediction}</answer>", answer, row["extra_info"])
                    self.assertEqual(result["acc"], expected)
