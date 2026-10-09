import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summary', ROOT / 'scripts/summarize_fsc_count_ab.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
spec = importlib.util.spec_from_file_location('metrics', ROOT / 'evaluation/VLMEvalKit/vlmeval/dataset/utils/fsc147.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class FSCCountABTests(unittest.TestCase):
    def prepare(self, root):
        annotation = root / 'annotation_FSC147_384.json'
        annotation.write_text(json.dumps({f'{i}.jpg': {'points': [[0, 0]] * 10} for i in range(1190)}))
        (root / 'protocol.json').write_text(json.dumps({'sha256': {str(annotation): 'fixture'}, 'timing': 'End-to-end wall time'}))
        for arm in ('text_only', 'auto_exemplar'):
            output = root / arm
            output.mkdir()
            predictions = ['10'] * 1190
            if arm == 'text_only':
                predictions[:5] = ['12', '10', 'invalid', '11', 'invalid']
            else:
                predictions[:5] = ['10', '13', '10', 'invalid', 'invalid']
            scored, counts = metrics.counting_metrics([10] * 1190, predictions)
            pd.DataFrame({
                'index': range(1190), 'image_path': [f'/images/{i}.jpg' for i in range(1190)],
                'category': ['apple'] * 1190, 'prediction': predictions,
                'parsed_count': counts, 'ground_truth_count': [10] * 1190,
            }).to_excel(output / 'VisualAgent-vllm_FSC147_TEST_score.xlsx', index=False)
            pd.DataFrame([scored]).to_csv(output / 'VisualAgent-vllm_FSC147_TEST_acc.csv', index=False)
        (root / 'status.tsv').write_text('arm\texit_code\tseconds\ntext_only\t0\t120\nauto_exemplar\t0\t180\n')

    def test_pairing_reports_invalid_transitions_without_imputing_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            result = summary.summarize(root)
            self.assertEqual(result['paired'], {
                'both_valid': 1187, 'improved': 1, 'degraded': 1, 'equal_error': 1185,
                'valid_to_invalid': 1, 'invalid_to_valid': 1, 'both_invalid': 1,
            })
            self.assertIsNone(result['text_only']['MAE'])
            self.assertIsNone(result['auto_exemplar']['RMSE'])
            self.assertEqual(result['timing']['rows'][1]['seconds'], 180)
            self.assertEqual(len(pd.read_csv(root / 'paired_examples.csv')), 1190)

    def test_historical_plus_three_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            path = root / 'auto_exemplar/VisualAgent-vllm_FSC147_TEST_score.xlsx'
            frame = pd.read_excel(path)
            frame['ground_truth_count'] += 3
            frame.to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Non-official point labels'):
                summary.summarize(root)


if __name__ == '__main__':
    unittest.main()
