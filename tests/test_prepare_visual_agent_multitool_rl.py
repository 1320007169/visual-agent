import importlib.util
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SPEC = importlib.util.spec_from_file_location(
    "prepare_visual_agent_multitool_rl",
    Path(__file__).parents[1] / "scripts/prepare_visual_agent_multitool_rl.py",
)
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


class PrepareVisualAgentMultitoolRLTest(unittest.TestCase):
    def test_depth_rows_exclude_conflicting_model_visible_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "scene.png"
            image.touch()
            rows = [
                {"images": [str(image)], "source": "ca_vqa_multichoice",
                 "uid": "ca_1", "question": "Which is closer: curtain [1,2,3,4] or light [5,6,7,8]? A. curtain B. light",
                 "solution": "A"},
                {"images": [str(image)], "source": "ca_vqa_multichoice",
                 "uid": "ca_2", "question": "Which is closer: curtain [1,2,3,4] or light [9,10,11,12]? A. curtain B. light",
                 "solution": "B"},
                {"images": [str(image)], "source": "gqa_depth", "uid": "gqa_1",
                 "question": "What is behind the hand?", "solution": "face"},
                {"images": [str(image)], "source": "gqa_depth", "uid": "gqa_2",
                 "question": "What is behind the hand?", "solution": "toothbrush"},
                {"images": [str(image)], "source": "gqa_depth", "uid": "valid_1",
                 "question": "What color is the curtain?", "solution": "white"},
                {"images": [str(image)], "source": "gqa_depth", "uid": "valid_2",
                 "question": "What color is the curtain?", "solution": "white"},
            ]
            path = root / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            original = path.read_bytes()
            result = prepare_module.depth_rows(path, "train")
            self.assertEqual([row["uid"] for row in result], ["valid_1", "valid_2"])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
