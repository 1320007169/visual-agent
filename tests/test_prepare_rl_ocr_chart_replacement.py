import base64
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_rl_ocr_chart_replacement as prepare


def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 12), color).save(buffer, format="PNG")
    return buffer.getvalue()


class OcrChartReplacementTest(unittest.TestCase):
    def fixture(self, root):
        base, hme, chart, benchmark = (root / name for name in ("base", "hme", "chart", "benchmark"))
        base.mkdir()
        hme.joinpath("data").mkdir(parents=True)
        benchmark.mkdir()
        val_image = root / "val.png"
        val_image.write_bytes(png_bytes("green"))
        fields = ["images", "question", "solution", "bbox", "source_index", "source_image", "data_source", "ability", "split", "image_digest", "cycle_category", "uid", "original_source", "count_complexity", "answer_aliases", "transform"]
        rows = []
        for index in range(22):
            value = {
                "images": [str(val_image)], "question": f"question {index}", "solution": "above",
                "bbox": [], "source_index": index, "source_image": str(val_image),
                "data_source": "visual-agent-zwz-relation" if index < 20 else "visual-agent-tallyqa",
                "ability": "spatial_relation" if index < 20 else "counting", "split": "train",
                "image_digest": "", "uid": f"old-{index}", "answer_aliases": ["above"],
                "original_source": "zwz" if index < 20 else "tallyqa", "transform": "",
            }
            rows.append({key: value.get(key) for key in fields})
        pq.write_table(pa.Table.from_pylist(rows), base / "train.parquet")
        prepare.write_jsonl(base / "train.jsonl", rows)
        val = [{**rows[-1], "uid": "heldout", "split": "val"}]
        pq.write_table(pa.Table.from_pylist(val, schema=pq.read_schema(base / "train.parquet")), base / "val.parquet")
        prepare.write_jsonl(base / "val.jsonl", val)
        hme_rows = [{"image": {"bytes": png_bytes(color), "path": None}, "label": f"x ^ {{ {index} }}", "image_path": f"train-{index}.jpg"}
                    for index, color in enumerate(("blue", "green", "red", "purple", "orange", "pink", "cyan", "gray"))]
        hme_rows.append({"image": {"bytes": png_bytes("black"), "path": None}, "label": "x < 1", "image_path": "invalid.jpg"})
        pq.write_table(pa.Table.from_pylist(hme_rows), hme / "data/train-00000-of-00001.parquet")
        pq.write_table(pa.Table.from_pylist([{**hme_rows[0], "label": "NEVER USE TEST"}]), hme / "data/test-00000-of-00001.parquet")
        train = chart / "ChartQA Dataset/train"
        (train / "png").mkdir(parents=True)
        chart_rows = []
        for index, color in enumerate(("yellow", "green", "navy", "brown", "lime", "silver")):
            path = train / "png" / f"chart-{index}.png"
            path.write_bytes(png_bytes(color))
            chart_rows.append({"imgname": path.name, "query": f"Chart question {index}?", "label": str(index)})
        (train / "train_human.json").write_text(json.dumps(chart_rows[:3]))
        (train / "train_augmented.json").write_text(json.dumps(chart_rows[3:]))
        for name, color in (("OCRBench", "blue"), ("ChartQA_TEST", "yellow")):
            payload = base64.b64encode(png_bytes(color)).decode()
            (benchmark / f"{name}.tsv").write_text(f"index\timage\n0\t{payload}\n")
        return base, hme, chart, benchmark, rows

    def test_replacement_preserves_size_other_tasks_and_validation_without_holdout_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, hme, chart, benchmark, original = self.fixture(root)
            before = {name: (base / name).read_bytes() for name in ("train.parquet", "val.parquet", "train.jsonl", "val.jsonl")}
            manifests = []
            selected = []
            for name in ("first", "second"):
                output = root / name
                manifest = prepare.prepare(base, hme, chart, benchmark, output, 0.2, 20261004)
                manifests.append(manifest)
                rows = pq.read_table(output / "train.parquet").to_pylist()
                self.assertEqual(len(rows), len(original))
                self.assertEqual(Counter(row["data_source"] for row in rows), {
                    "visual-agent-zwz-relation": 16, "visual-agent-tallyqa": 2,
                    "visual-agent-ocr": 2, "visual-agent-chartqa": 2,
                })
                self.assertTrue(all(row in rows for row in original[-2:]))
                self.assertEqual((output / "val.parquet").read_bytes(), before["val.parquet"])
                self.assertEqual((output / "val.jsonl").read_bytes(), before["val.jsonl"])
                additions = [row for row in rows if row["original_source"] in {"hme100k", "chartqa"}]
                selected.append(sorted((row["uid"], row["solution"]) for row in additions))
                self.assertTrue(all(Path(row["images"][0]).is_file() for row in additions))
                excluded = {prepare.image_digest(Image.new("RGB", (20, 12), color)) for color in ("blue", "green", "yellow")}
                self.assertTrue(all(row["image_digest"] not in excluded for row in additions))
                self.assertTrue(all("<" not in row["solution"] for row in additions))
                self.assertFalse((output / "resume").exists())
                self.assertEqual(manifest["replaced_relation_rows"], 4)
                self.assertEqual(manifest["experiment_mode"], "new_experiment")
                self.assertEqual(manifest["exclusion_counts"]["hme_excluded_answer_protocol"], 1)
            self.assertEqual(selected[0], selected[1])
            self.assertEqual(manifests[0]["removed_relation_indices"], manifests[1]["removed_relation_indices"])
            for name, content in before.items():
                self.assertEqual((base / name).read_bytes(), content)
            with self.assertRaises(FileExistsError):
                prepare.prepare(base, hme, chart, benchmark, root / "first")

    def test_formula_ocr_keeps_case_and_ignores_only_latex_whitespace(self):
        spec = importlib.util.spec_from_file_location("replacement_reward", ROOT / "reinforcement_learning/verl/utils/reward_score/visual_agent_thyme.py")
        reward = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward)
        info = {"data_source": "visual-agent-ocr", "original_source": "hme100k", "answer_aliases": []}
        for answer, gold, expected in (
            (r"\frac{A}{2}+x", r"\frac { A } { 2 } + x", 1),
            (r"\frac{a}{2}+x", r"\frac { A } { 2 } + x", 0),
            (r"\angleABC", r"\angle A B C", 0),
            (r"$\frac{A}{2}+x$", r"\frac { A } { 2 } + x", 1),
        ):
            with self.subTest(answer=answer), patch.object(reward, "judge_match", return_value=False) as judge:
                result = reward.compute_score(f"<answer>{answer}</answer>", gold, info)
                self.assertEqual(result["acc"], expected)
                if expected:
                    judge.assert_not_called()
                else:
                    judge.assert_called_once_with("", answer, gold)


if __name__ == "__main__":
    unittest.main()
