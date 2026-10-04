#!/usr/bin/env python3
"""Compare all OCRBench predictions, category scores, and OCR observations."""

import argparse
import ast
import json
from pathlib import Path

HME = 'Handwritten Mathematical Expression Recognition'


def summarize(root):
    arms = {}
    summary = {}
    for arm in ('json_observation', 'raw_ocr_backslash'):
        paths = list((root / arm / 'dino_latest' / 'VisualAgent-vllm').glob('*/VisualAgent-vllm_OCRBench.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one prediction JSON for {arm}, got {len(paths)}')
        rows = json.loads(paths[0].read_text())
        records = {}
        for row in rows:
            prediction = str(row['prediction'])
            failed = 'Failed to obtain answer via API' in prediction
            answers = ast.literal_eval(row['answer'])
            category = row['category']
            if category == HME:
                normalized = prediction.strip().replace('\n', ' ').replace(' ', '')
                normalized_answers = [answer.strip().replace('\n', ' ').replace(' ', '') for answer in answers]
            else:
                normalized = prediction.lower().strip().replace('\n', ' ')
                normalized_answers = [answer.lower().strip().replace('\n', ' ') for answer in answers]
            correct = not failed and any(
                answer in normalized for answer in normalized_answers
            )
            trace = json.loads(row['raw_response']) if row.get('raw_response') else {}
            calls = [call for call in trace.get('tool_calls', []) if call['name'] == 'ocr_read']
            messages = trace.get('messages', [])
            ocr_ids = {
                call['id'] for message in messages for call in message.get('tool_calls', [])
                if call['function']['name'] == 'ocr_read'
            }
            observations = [message['content'] for message in messages
                            if message.get('role') == 'tool' and message.get('tool_call_id') in ocr_ids]
            records[int(row['index'])] = {
                'prediction': prediction, 'correct': correct, 'api_failed': failed,
                'category': category, 'answers': answers, 'ocr_calls': calls, 'ocr_observations': observations,
            }
        if len(rows) != 1000 or set(records) != set(range(1000)):
            raise ValueError(f'{arm} must contain exactly the 1000 OCRBench samples, indices 0-999')
        arms[arm] = records
        summary[arm] = {
            'correct': sum(row['correct'] for row in records.values()),
            'total': len(records),
            'api_failed': sum(row['api_failed'] for row in records.values()),
            'prediction_file': str(paths[0]),
        }
        score_path = paths[0].with_name('VisualAgent-vllm_OCRBench_score.json')
        official = json.loads(score_path.read_text())
        summary[arm]['official_scores'] = official
        hme = [row for row in records.values() if row['category'] == HME]
        if len(hme) != 100:
            raise ValueError(f'{arm}: expected 100 HME samples within OCRBench')
        if (summary[arm]['correct'] != official['Final Score']
                or sum(row['correct'] for row in hme) != official[HME]):
            raise ValueError(f'{arm}: paired scoring disagrees with the OCRBench evaluator')
    baseline, raw = arms['json_observation'], arms['raw_ocr_backslash']
    for index in baseline:
        if (baseline[index]['category'], baseline[index]['answers']) != (raw[index]['category'], raw[index]['answers']):
            raise ValueError(f'Sample {index}: categories or answers differ between arms')
    summary['wrong_to_right'] = [i for i in sorted(baseline) if not baseline[i]['correct'] and raw[i]['correct']]
    summary['right_to_wrong'] = [i for i in sorted(baseline) if baseline[i]['correct'] and not raw[i]['correct']]
    summary['score_delta'] = summary['raw_ocr_backslash']['correct'] - summary['json_observation']['correct']
    groups = {category: [i for i in sorted(baseline) if baseline[i]['category'] == category]
              for category in sorted({row['category'] for row in baseline.values()})}
    groups['Other 900 (non-HME)'] = [i for i in sorted(baseline) if baseline[i]['category'] != HME]
    summary['by_category'] = {}
    for category, indices in groups.items():
        before = sum(baseline[i]['correct'] for i in indices)
        after = sum(raw[i]['correct'] for i in indices)
        summary['by_category'][category] = {
            'total': len(indices), 'json_correct': before, 'raw_backslash_correct': after,
            'score_delta': after - before,
            'json_api_failed': sum(baseline[i]['api_failed'] for i in indices),
            'raw_backslash_api_failed': sum(raw[i]['api_failed'] for i in indices),
            'wrong_to_right': [i for i in indices if not baseline[i]['correct'] and raw[i]['correct']],
            'right_to_wrong': [i for i in indices if baseline[i]['correct'] and not raw[i]['correct']],
        }
    (root / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (root / 'paired_examples.jsonl').open('w') as stream:
        for index in sorted(baseline):
            record = {'index': index, 'json_observation': baseline[index], 'raw_ocr_backslash': raw[index]}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.result_dir), ensure_ascii=False, indent=2))
