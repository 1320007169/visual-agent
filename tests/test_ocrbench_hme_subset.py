import ast
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / 'evaluation/VLMEvalKit/vlmeval/dataset/image_vqa.py'


class OcrBenchJsonEvaluationTest(unittest.TestCase):
    def setUp(self):
        # Load the dataset class without importing unrelated model dependencies.
        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OCRBench')
        self.dumps = {}
        namespace = {
            'ImageBaseDataset': object, 'pd': pd, 'tqdm': lambda x: x,
            'load': lambda path: json.loads(Path(path).read_text()),
            'dump': lambda value, path: self.dumps.update({path: value}),
            'get_intermediate_file_path': lambda path, suffix, ext: str(Path(path).with_suffix('')) + suffix + '.' + ext,
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.namespace = namespace
        self.dataset = namespace['OCRBench']()
        self.data = pd.DataFrame([
            {'index': i, 'category': ('Handwritten Mathematical Expression Recognition' if i >= 900
                                     else 'Regular Text Recognition'),
             'answer': "['x + y']" if i >= 900 else "['Red Sign']",
             'prediction': 'x+y' if i >= 900 else 'RED SIGN'} for i in range(1000)
        ])

    def test_full_json_evaluation_preserves_category_rules_and_long_traces(self):
        rows = self.data.to_dict('records')
        rows[0]['prediction'] = 'RED  SIGN'
        rows[1]['prediction'] = 'Failed to obtain answer via API.'
        rows[2]['raw_response'] = json.dumps({'messages': ['x' * 50000]})
        rows[900]['prediction'] = 'wrong'
        rows[901]['prediction'] = 'X+Y'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.json'
            path.write_text(json.dumps(rows))
            score = self.dataset.evaluate(str(path))
            self.assertEqual(json.loads(path.read_text())[2]['raw_response'], rows[2]['raw_response'])
        self.assertEqual(score['Handwritten Mathematical Expression Recognition'], 98)
        self.assertEqual(score['Text Recognition'], 898)
        self.assertEqual(score['Final Score'], 996)
        self.assertEqual(score['Final Score Norm'], 99.6)

    def test_existing_dataframe_predictions_still_score_all_1000_rows(self):
        self.namespace['load'] = lambda path: self.data
        score = self.dataset.evaluate('predictions.xlsx')
        self.assertEqual(score['Final Score'], 1000)
        self.assertEqual(score['Handwritten Mathematical Expression Recognition'], 100)
        self.assertEqual(score['Final Score Norm'], 100.0)


if __name__ == '__main__':
    unittest.main()
