"""Text-specified, tool-assisted counting on the FSC147 test split."""

from pathlib import Path

import pandas as pd

from .image_base import ImageBaseDataset
from .utils.fsc147 import counting_metrics
from ..smp import LMUDataRoot, dump, get_intermediate_file_path, load


class FSC147(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {"FSC147_TEST": ""}

    def load_data(self, dataset):
        path = Path(LMUDataRoot()) / f"{dataset}.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/prepare_fsc147_eval.py first")
        data = load(str(path))
        for column in ("index", "image_path", "question", "answer", "category"):
            if column not in data:
                raise ValueError(f"Missing FSC147 column: {column}")
        if data.empty or data["index"].duplicated().any():
            raise ValueError("FSC147 test data must be nonempty with unique indices")
        for path in data["image_path"]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing FSC147 test image: {path}")
        return data

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if len(data) != len(self.data) or data["index"].astype(str).duplicated().any():
            raise ValueError("FSC147 predictions must cover the full test split exactly once")
        reference = self.data.set_index(self.data["index"].astype(str))
        if set(data["index"].astype(str)) != set(reference.index):
            raise ValueError("FSC147 prediction indices do not match the test split")
        answers = [reference.loc[str(index), "answer"] for index in data["index"]]
        metrics, extracted = counting_metrics(answers, list(data["prediction"]))
        data["parsed_count"] = extracted
        data["ground_truth_count"] = answers
        dump(data, get_intermediate_file_path(eval_file, "_score", "xlsx"))
        result = pd.DataFrame([metrics])
        dump(result, get_intermediate_file_path(eval_file, "_acc", "csv"))
        return result
