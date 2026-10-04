import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "reinforcement_learning/verl/utils"


class OcrChartRegistrationTest(unittest.TestCase):
    def test_dataset_keeps_sources_splits_and_answer_aliases(self):
        parent_module = ModuleType("verl.utils.dataset.zwz_original_relation_dataset")

        class ParentDataset:
            def __getitem__(self, item):
                return {"extra_info": {}}

        parent_module.ZwzOriginalRelationDataset = ParentDataset
        spec = importlib.util.spec_from_file_location(
            "ocr_chart_dataset_test", ROOT / "dataset/zwz_deepeyesv2_dataset.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {parent_module.__name__: parent_module}):
            spec.loader.exec_module(module)
        for source in ("visual-agent-ocr", "visual-agent-chartqa"):
            for split in ("train", "val"):
                with self.subTest(source=source, split=split):
                    dataset = module.ZwzDeepEyesV2Dataset()
                    dataset.dataframe = [{
                        "data_source": source, "split": split, "solution": "NYC",
                        "question": "Which city?", "ability": "text_reading",
                        "source_image": "/image.png", "uid": "sample1",
                        "original_source": "textvqa", "answer_aliases": ["NYC", "New York"],
                    }]
                    row = dataset[0]
                    self.assertEqual(row["data_source"], source)
                    self.assertEqual(row["reward_model"]["ground_truth"], "NYC")
                    self.assertEqual(row["extra_info"]["split"], split)
                    self.assertEqual(row["extra_info"]["answer_aliases"], ["NYC", "New York"])
                    self.assertEqual(row["extra_info"]["original_source"], "textvqa")
                    self.assertEqual(row["extra_info"]["uid"], "sample1")

    def test_reward_dispatch_reaches_ocr_and_chart_judge_fallback(self):
        import_utils = ModuleType("verl.utils.import_utils")
        import_utils.deprecated = lambda name: lambda function: function
        name = "_ocr_chart_registered_rewards"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "reward_score/__init__.py",
            submodule_search_locations=[str(ROOT / "reward_score")],
        )
        registry = importlib.util.module_from_spec(spec)
        reward_spec = importlib.util.spec_from_file_location(
            name + ".visual_agent_thyme", ROOT / "reward_score/visual_agent_thyme.py",
        )
        reward = importlib.util.module_from_spec(reward_spec)
        reward_spec.loader.exec_module(reward)
        modules = {import_utils.__name__: import_utils, name: registry, reward_spec.name: reward}
        with patch.dict(sys.modules, modules), patch.object(reward, "judge_match", return_value=False) as judge:
            spec.loader.exec_module(registry)
            cases = [("visual-agent-ocr", "NYC", "New York", ["NYC"], 1),
                     ("visual-agent-ocr", "Boston", "New York", ["NYC"], 0),
                     ("visual-agent-chartqa", "100.0", "100", [], 1),
                     ("visual-agent-chartqa", "105", "100", [], 0),
                     ("visual-agent-chartqa", "106", "100", [], 0)]
            for source, answer, gold, aliases, expected in cases:
                with self.subTest(source=source, answer=answer):
                    extra = {"data_source": source, "answer_aliases": aliases}
                    result = registry.default_compute_score(source, f"<answer>{answer}</answer>", gold, extra)
                    self.assertEqual(result["acc"], expected)
            self.assertEqual(judge.call_count, 3)
