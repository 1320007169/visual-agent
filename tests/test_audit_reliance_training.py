import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_reliance_training import audit_step


class RelianceTrainingAuditTest(unittest.TestCase):
    def test_full_crop_bypass_and_distinct_k2_grpo_groups(self):
        sample = {'data_source': 'visual-agent-ocr', 'original_source': 'hme100k', 'source_image': 'image',
                  'images': ['image'], 'question': 'read', 'solution': '5', 'answer_aliases': [],
                  'original_correct': True, 'reliance_branch': 'counterfactual'}
        samples = [{**sample, 'reliance_fault': json.dumps({'tool': 'ocr_read', 'arguments': {'target_image': 0},
                    'observation': json.dumps({'text': text})})} for text in ('S', '8')]
        crop = {'tool': 'crop_zoom', 'arguments': {'bbox_2d': [50, 50, 950, 950]}, 'status': 'success',
                'model_observation': '{}', 'returned_image_count': 1,
                'raw_result': {'crop_zoom': {'bbox_2d': [0, 0, 1000, 1000]}}}
        records = []
        for index, results in enumerate(((1, 0), (0, 0))):
            for correct in results:
                call = {'tool': 'ocr_read', 'arguments': {'target_image': 1, 'bbox_2d': [0, 0, 1000, 1000]},
                        'status': 'success', 'replayed_fault': not correct,
                        'model_observation': json.dumps({'text': '5' if correct else ('S' if index == 0 else '8')})}
                records.append({'source_metadata': {**sample, 'source_index': 999, 'dataset_index': index}, 'score': correct, 'acc': correct,
                                'rollout_trace': {'tool_calls': [crop, call]}})
        path = Path(tempfile.mkdtemp(prefix='v3-audit-test-')) / '5.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        report = audit_step(path, samples, 2)
        counts = next(iter(report['by_source_branch_original_correct'].values()))
        self.assertEqual(counts['groups'], 2)
        self.assertEqual(counts['mixed_group_fraction'], 0.5)
        self.assertEqual(counts['bypass_rate'], 0.25)
        self.assertEqual(counts['replays'], 3)
        self.assertEqual(report['tools_per_trajectory'], 2)
        self.assertEqual(report['mean_reward'], 0.25)
        with self.assertRaisesRegex(ValueError, 'Incomplete GRPO group'):
            audit_step(path, samples, 16)


if __name__ == '__main__':
    unittest.main()
