import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_visual_agent_eval_ocr_hme_ab_8gpu_modelarts.sh'
SUMMARY = ROOT / 'scripts/summarize_ocr_hme_ab.py'
spec = importlib.util.spec_from_file_location('hme_summary', SUMMARY)
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


class HmeAbEvalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        repo = self.root / 'repo'
        scripts = repo / 'scripts'
        scripts.mkdir(parents=True)
        shutil.copyfile(SUMMARY, scripts / SUMMARY.name)
        model = repo / ('saves/visual_agent_zwz_rl/qwen3/'
                        'qwen3base_multitool_vlocr_ocr_chart1600_n16_8node_20261001T072342925318_021ea85c/'
                        'global_step_80/actor/huggingface')
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'weights'}}))
        (model / 'weights').write_bytes(b'fixture')
        self.tsv = self.root / 'DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData/OCRBench.tsv'
        self.tsv.parent.mkdir(parents=True)
        self.tsv.write_text('index\tcategory\n' + ''.join(
            f'{i}\t' + ('Handwritten Mathematical Expression Recognition' if i >= 900
                        else 'Regular Text Recognition') + '\n' for i in range(1000)
        ))
        self.output = self.root / 'results'
        self.env = dict(os.environ, BASE=str(self.root), REPO_ROOT=str(repo),
                        WORK_ROOT=str(self.output), RUN_ID='hme_test', NNODES='1',
                        SKIP_MODELARTS_BOOTSTRAP='1', CHECKPOINT_EVAL_CONFIG_ONLY='0',
                        VLOCR_REUSE_GROUP_ROOT='/old/results', VLOCR_RETRY_FAILED_ONLY='1')
        (scripts / 'run_visual_agent_eval_multitool_vlocr_8gpu.sh').write_text('''#!/usr/bin/env bash
python3 - <<'PY'
import json, os
from pathlib import Path
keys = ['VLOCR_MODEL_PATH', 'VLOCR_STEP', 'PRED_FORMAT',
        'EVAL_DATASETS', 'VISUAL_AGENT_OCR_RAW_BACKSLASH', 'VLMEVAL_EVAL_ID',
        'EVAL_TIMEOUT_SECONDS', 'VLOCR_REUSE_GROUP_ROOT', 'VLOCR_RETRY_FAILED_ONLY',
        'VISUAL_AGENT_FAILURE_TRACE_DIR']
with (Path(os.environ['BASE']) / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps({key: os.getenv(key) for key in keys}) + '\\n')
raw = os.environ['VISUAL_AGENT_OCR_RAW_BACKSLASH'] == '1'
if not raw and os.getenv('TEST_FIRST_EXIT'):
    raise SystemExit(int(os.environ['TEST_FIRST_EXIT']))
directory = Path(os.environ['WORK_ROOT']) / 'dino_latest/VisualAgent-vllm' / os.environ['VLMEVAL_EVAL_ID']
directory.mkdir(parents=True)
correct = 65 if raw else 45
rows = []
for i in range(1000):
    hme = i >= 900
    right = (i - 900 < correct) if hme else (i < 810 and not (raw and i == 0))
    rows.append({'index': i, 'answer': "['x+y']" if hme else "['Red Sign']",
                 'category': 'Handwritten Mathematical Expression Recognition' if hme else 'Regular Text Recognition',
                 'prediction': ('x+y' if hme else 'RED SIGN') if right else 'wrong',
                 'raw_response': json.dumps({'messages': [], 'tool_calls': []})})
(directory / 'VisualAgent-vllm_OCRBench.json').write_text(json.dumps(rows))
(directory / 'VisualAgent-vllm_OCRBench_score.json').write_text(json.dumps({
    'Handwritten Mathematical Expression Recognition': correct, 'Final Score': correct + (809 if raw else 810)}))
PY
''')

    def launch(self, **changes):
        return subprocess.run(['bash', str(LAUNCHER)], env=dict(self.env, **changes),
                              text=True, capture_output=True, timeout=30)

    def test_two_fresh_arms_share_weights_and_report_paired_changes(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['VLOCR_MODEL_PATH'], calls[1]['VLOCR_MODEL_PATH'])
        self.assertEqual([call['VISUAL_AGENT_OCR_RAW_BACKSLASH'] for call in calls], ['0', '1'])
        self.assertNotEqual(calls[0]['VLMEVAL_EVAL_ID'], calls[1]['VLMEVAL_EVAL_ID'])
        for call in calls:
            self.assertEqual(call['VLOCR_STEP'], '80')
            self.assertEqual(call['EVAL_DATASETS'], 'OCRBench')
            self.assertEqual(call['PRED_FORMAT'], 'json')
            self.assertEqual(call['EVAL_TIMEOUT_SECONDS'], '0')
            self.assertIsNone(call['VLOCR_REUSE_GROUP_ROOT'])
            self.assertIsNone(call['VLOCR_RETRY_FAILED_ONLY'])
        summary = json.loads((self.output / 'comparison.json').read_text())
        self.assertEqual(summary['json_observation']['correct'], 855)
        self.assertEqual(summary['raw_ocr_backslash']['correct'], 874)
        self.assertEqual(summary['score_delta'], 19)
        self.assertEqual(summary['wrong_to_right'], list(range(945, 965)))
        self.assertEqual(summary['right_to_wrong'], [0])
        hme = summary['by_category']['Handwritten Mathematical Expression Recognition']
        self.assertEqual((hme['total'], hme['json_correct'], hme['raw_backslash_correct']), (100, 45, 65))
        other = summary['by_category']['Other 900 (non-HME)']
        self.assertEqual((other['total'], other['score_delta'], other['right_to_wrong']), (900, -1, [0]))
        self.assertEqual(len((self.output / 'paired_examples.jsonl').read_text().splitlines()), 1000)

    def test_failed_first_arm_is_recorded_and_second_is_attempted(self):
        result = self.launch(TEST_FIRST_EXIT='17')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        status = (self.output / 'status.tsv').read_text().splitlines()
        self.assertEqual([row.split('\t')[:2] for row in status[1:]],
                         [['json_observation', '17'], ['raw_ocr_backslash', '0']])
        self.assertFalse((self.output / 'comparison.json').exists())

    def test_configuration_check_does_not_launch_or_create_results(self):
        result = self.launch(CHECKPOINT_EVAL_CONFIG_ONLY='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / 'calls.jsonl').exists())

    def test_wrong_sample_set_is_rejected_before_launch(self):
        self.tsv.write_text('index\tcategory\n900\tHandwritten Mathematical Expression Recognition\n')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Expected the full 1000 OCRBench samples', result.stderr)
        self.assertFalse((self.root / 'calls.jsonl').exists())

    def test_paired_summary_rejects_missing_samples(self):
        self.assertEqual(self.launch().returncode, 0)
        p = next((self.output / 'raw_ocr_backslash').glob('dino_latest/VisualAgent-vllm/*/VisualAgent-vllm_OCRBench.json'))
        p.write_text(json.dumps(json.loads(p.read_text())[:-1]))
        with self.assertRaisesRegex(ValueError, 'exactly the 1000 OCRBench samples'):
            summary_module.summarize(self.output)


if __name__ == '__main__':
    unittest.main()
