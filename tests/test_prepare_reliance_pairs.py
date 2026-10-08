from collections import Counter
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
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
    def test_ocr_pair_is_rejected_when_normalized_answer_survives_injection(self):
        row = {"solution": "x = 5", "data_source": "visual-agent-ocr", "original_source": "hme100k"}
        call = {"tool": "ocr_read", "arguments": {"target_image": 0}, "observation": {"text": "x=5"}}
        with patch.object(pairs.tool_faults, "inject_fault", return_value={"text": r"\(\mathrm{x}=5\) or y=3"}) as inject:
            self.assertIsNone(pairs.build_pair(row, call, random.Random(0), "pair"))
        self.assertEqual(inject.call_args.kwargs["variant"], "hme")
        with patch.object(pairs.tool_faults, "inject_fault", return_value={"text": r"\(x=3\)"}):
            self.assertIsNotNone(pairs.build_pair(row, call, random.Random(0), "pair"))

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

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is required")
    def test_same_source_ablation_matches_questions_positions_and_untouched_sources(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        sources = ["visual-agent-ocr", "visual-agent-tallyqa", "visual-agent-depth-raw"]
        rows = [{"images": [f"{source}_{i}.jpg"], "source_image": f"{source}_{i}.jpg",
                 "question": f"{source}_{i}", "solution": "5", "data_source": source,
                 "original_source": "hme100k" if source == "visual-agent-ocr" and i == 0 else "other"}
                for source in sources for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, rollouts = root / "base", root / "rollouts"
            base.mkdir()
            rollouts.mkdir()
            pq.write_table(pa.Table.from_pylist(rows), base / "train.parquet")
            pq.write_table(pa.Table.from_pylist(rows[:1]), base / "val.parquet")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"default": 0, "by_source": {
                "visual-agent-ocr": {"ocr_read": 1}, "visual-agent-tallyqa": {"object_count": 1}}}))
            records = [record(row["data_source"], row["source_image"], row["question"], [
                ("ocr_read", {"target_image": 0}, 1, "success", {"text": "5"})
                if row["data_source"] == "visual-agent-ocr" else COUNT_CALL]) for row in rows]
            (rollouts / "1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            manifests, outputs = [], []
            for factual_only in [False, True]:
                output = root / str(factual_only)
                manifests.append(pairs.prepare(base, [rollouts], output, (1, 1), fraction=0.25,
                                               profile_path=profile, same_source=True, factual_only=factual_only))
                outputs.append(pq.read_table(output / "train.parquet").to_pylist())
                self.assertEqual((output / "val.parquet").read_bytes(), (base / "val.parquet").read_bytes())
            self.assertEqual(manifests[0]["replaced_indices_before_shuffle"],
                             manifests[1]["replaced_indices_before_shuffle"])
            for output in outputs:
                self.assertEqual(Counter(r["data_source"] for r in output), Counter(r["data_source"] for r in rows))
                self.assertEqual(Counter(pairs.replacement_source(r) for r in output),
                                 Counter(pairs.replacement_source(r) for r in rows))
                self.assertFalse(any(r["reliance_branch"] for r in output if r["original_source"] == "hme100k"))
                prefixed = Counter(pairs.replacement_source(r) for r in output if r["reliance_branch"])
                totals = Counter(pairs.replacement_source(r) for r in rows)
                self.assertTrue(all(count <= totals[source] // 2 for source, count in prefixed.items()))
                depth = [{k: v for k, v in r.items() if k not in pairs.RELIANCE_FIELDS}
                         for r in output if r["data_source"] == "visual-agent-depth-raw"]
                self.assertEqual(sorted(depth, key=lambda r: r["question"]), rows[16:])
            factual_prefixes = {r["reliance_pair_id"]: r["reliance_prefix"] for r in outputs[0]
                                if r["reliance_branch"] == "factual"}
            for paired, control in zip(*outputs, strict=True):
                if paired["reliance_branch"] == "counterfactual":
                    expected = {**paired, "reliance_branch": "factual", "reliance_fault": "",
                                "reliance_prefix": factual_prefixes[paired["reliance_pair_id"]]}
                    self.assertEqual(control, expected)
                else:
                    self.assertEqual(control, paired)
            self.assertEqual(Counter(r["reliance_branch"] for r in outputs[1]), {"": 18, "factual": 6})
            with self.assertRaisesRegex(ValueError, "Only 3 pairs"):
                pairs.prepare(base, [rollouts], root / "too_many", (1, 1), fraction=0.9,
                              profile_path=profile, same_source=True)
            self.assertFalse((root / "too_many").exists())


if __name__ == "__main__":
    unittest.main()
