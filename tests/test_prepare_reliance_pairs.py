from collections import Counter
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_reliance_pairs as pairs  # noqa: E402


def record(source, image, question, calls):
    return {
        "source_metadata": {"data_source": source, "source_image": image, "question": question},
        "rollout_trace": {"tool_calls": [
            {"tool": tool, "arguments": arguments, "model_turn": turn, "status": status,
             "model_observation": json.dumps(observation)}
            for tool, arguments, turn, status, observation in calls
        ]},
    }


COUNT_CALL = ("object_count", {"query": "apples", "target_image": 0}, 1, "success", {"count": 7})


class ReliancePairTest(unittest.TestCase):
    def test_prefix_candidates_keep_only_faultable_first_calls(self):
        keys = {pairs.question_key("visual-agent-tallyqa", "a.jpg", "How many apples?")}
        records = [
            record("visual-agent-tallyqa", "a.jpg", "How many apples?", [COUNT_CALL]),
            record("visual-agent-tallyqa", "a.jpg", "How many apples?", [("crop_zoom", {}, 1, "success", {"target_image": 1})]),
            record("visual-agent-tallyqa", "a.jpg", "How many apples?", [("object_count", {}, 2, "success", {"count": 3})]),
            record("visual-agent-tallyqa", "a.jpg", "How many apples?", [("object_count", {}, 1, "error", {"count": 3})]),
            record("visual-agent-tallyqa", "b.jpg", "Unknown?", [COUNT_CALL]),
            {**record("visual-agent-tallyqa", "a.jpg", "How many apples?", [COUNT_CALL]),
             "source_metadata": {"data_source": "visual-agent-tallyqa", "source_image": "a.jpg",
                                 "question": "How many apples?", "reliance_branch": "factual"}},
        ]
        stats = Counter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            candidates = pairs.prefix_candidates([path], keys, stats)
        self.assertEqual(list(candidates.values()), [[{
            "tool": "object_count", "arguments": {"query": "apples", "target_image": 0}, "observation": {"count": 7},
        }]])
        self.assertEqual(stats, Counter(first_action_not_faultable=3, unmatched_question=1, skipped_prefixed_rollout=1))

    def test_pair_rows_share_the_call_and_differ_only_in_observation(self):
        row = {"data_source": "visual-agent-tallyqa", "solution": "5", "question": "How many apples?", "uid": "t1"}
        call = {"tool": "object_count", "arguments": {"query": "apples", "target_image": 0}, "observation": {"count": 7}}
        factual, counterfactual = pairs.build_pair(row, call, random.Random(0), "pair_000000")
        factual_prefix, counter_prefix = json.loads(factual["reliance_prefix"]), json.loads(counterfactual["reliance_prefix"])
        self.assertEqual(factual_prefix[0], counter_prefix[0])
        self.assertEqual(factual_prefix[0]["tool_calls"][0]["function"]["name"], "object_count")
        self.assertEqual(json.loads(factual_prefix[1]["content"]), {"count": 7})
        faulty_count = json.loads(counter_prefix[1]["content"])["count"]
        self.assertNotIn(faulty_count, {5, 7})
        self.assertEqual(json.loads(counterfactual["reliance_fault"]), {
            "tool": "object_count", "arguments": call["arguments"], "observation": counter_prefix[1]["content"],
        })
        self.assertEqual((factual["reliance_fault"], factual["reliance_branch"], counterfactual["reliance_branch"]),
                         ("", "factual", "counterfactual"))
        self.assertEqual({key: factual[key] for key in row}, row)
        self.assertEqual(factual["reliance_pair_id"], counterfactual["reliance_pair_id"])

    def test_profile_weight_zero_excludes_a_tool(self):
        rows = {
            pairs.question_key("visual-agent-tallyqa", "a.jpg", "q1"): {"data_source": "visual-agent-tallyqa", "solution": "5"},
            pairs.question_key("visual-agent-ocr", "b.jpg", "q2"): {"data_source": "visual-agent-ocr", "solution": "OPEN"},
        }
        candidates = {
            pairs.question_key("visual-agent-tallyqa", "a.jpg", "q1"): [
                {"tool": "object_count", "arguments": {}, "observation": {"count": 7}}],
            pairs.question_key("visual-agent-ocr", "b.jpg", "q2"): [
                {"tool": "ocr_read", "arguments": {}, "observation": {"text": "OPEN 24H"}}],
        }
        profile = {"default": 0.5, "by_source": {"visual-agent-ocr": {"ocr_read": 0.0}}}
        selected = pairs.select_pairs(rows, candidates, 1, profile, random.Random(0), Counter())
        self.assertEqual(selected[0][0]["data_source"], "visual-agent-tallyqa")
        with self.assertRaises(ValueError):
            pairs.select_pairs(rows, candidates, 2, profile, random.Random(0), Counter())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is required")
    def test_prepare_replaces_rows_and_keeps_schema(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        base_rows = [
            {"images": [f"{index}.jpg"], "question": f"q{index}", "solution": "5", "source_image": f"{index}.jpg",
             "data_source": "visual-agent-tallyqa", "uid": f"t{index}"}
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            base, rollouts, output = Path(directory) / "base", Path(directory) / "rollouts", Path(directory) / "out"
            base.mkdir()
            rollouts.mkdir()
            pq.write_table(pa.Table.from_pylist(base_rows), base / "train.parquet")
            pq.write_table(pa.Table.from_pylist(base_rows[:2]), base / "val.parquet")
            records = [record("visual-agent-tallyqa", f"{index}.jpg", f"q{index}", [COUNT_CALL]) for index in range(10)]
            (rollouts / "1.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            manifest = pairs.prepare(base, [rollouts], output, (1, 1), fraction=0.4)
            train = pq.read_table(output / "train.parquet").to_pylist()
            self.assertEqual((manifest["pairs"], manifest["replaced_rows"], len(train)), (2, 4, 10))
            self.assertEqual(Counter(row["reliance_branch"] for row in train), Counter({"": 6, "factual": 2, "counterfactual": 2}))
            self.assertEqual((output / "val.parquet").read_bytes(), (base / "val.parquet").read_bytes())

            factual_output = Path(directory) / "factual"
            pairs.prepare(base, [rollouts], factual_output, (1, 1), fraction=0.4, factual_only=True)
            train = pq.read_table(factual_output / "train.parquet").to_pylist()
            self.assertEqual(Counter(row["reliance_branch"] for row in train), Counter({"": 6, "factual": 4}))
            self.assertEqual(len({row["question"] for row in train if row["reliance_branch"]}), 4)


if __name__ == "__main__":
    unittest.main()
