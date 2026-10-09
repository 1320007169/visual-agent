"""Analyze saved step80 counting errors and MME E answers before tool reruns."""

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def read_trace(raw):
    try:
        trace = json.loads(raw, strict=False)
        return trace, True, True
    except json.JSONDecodeError:
        decoder = json.JSONDecoder(strict=False)
        if '"tool_calls":' not in raw:
            return {'tool_calls': []}, False, False
        marker = raw.index('"tool_calls":') + len('"tool_calls":')
        remaining = raw[marker:].lstrip()
        assert remaining.startswith('[')
        remaining = remaining[1:].lstrip()
        calls = []
        while remaining and not remaining.startswith(']'):
            try:
                call, end = decoder.raw_decode(remaining)
            except json.JSONDecodeError:
                call = {}
                fields = remaining[1:].lstrip()
                while fields:
                    try:
                        key, end = decoder.raw_decode(fields)
                        fields = fields[end:].lstrip()[1:].lstrip()
                        if key == 'result':
                            result = {}
                            values = fields[1:].lstrip()
                            while values:
                                result_key, end = decoder.raw_decode(values)
                                values = values[end:].lstrip()[1:].lstrip()
                                value, end = decoder.raw_decode(values)
                                result[result_key] = value
                                call['result'] = result
                                values = values[end:].lstrip()
                                if not values.startswith(','):
                                    break
                                values = values[1:].lstrip()
                            break
                        value, end = decoder.raw_decode(fields)
                        call[key] = value
                        fields = fields[end:].lstrip()
                        if not fields.startswith(','):
                            break
                        fields = fields[1:].lstrip()
                    except json.JSONDecodeError:
                        break
                calls.append(call)
                return {'tool_calls': calls}, False, False
            calls.append(call)
            remaining = remaining[end:].lstrip()
            if remaining.startswith(','):
                remaining = remaining[1:].lstrip()
        return {'tool_calls': calls}, False, remaining.startswith(']')


output = Path(__file__).resolve().parent
repo = output.parents[2]
base = repo.parent
source = next((repo / 'outputs/vlmeval/vlocr_64gpu_step80_8gpu').glob('*/64gpu_step80/dino_latest/VisualAgent-vllm/T*'))
annotation = base / 'datasets/fsc147/annotation_FSC147_384.json'
official = json.loads(annotation.read_text())
image_root = base / 'visual-tools/datasets/fsc147/images_384_VarV2'
fsc_file = source / 'VisualAgent-vllm_FSC147_TEST_score.xlsx'
data = pd.read_excel(fsc_file).fillna('')
fsc_rows, inputs = [], []
for row in data.to_dict('records'):
    filename = Path(row['image_path']).name
    gt = len(official[filename]['points'])
    trace, trace_complete, calls_complete = read_trace(row['raw_response'])
    calls = [call for call in trace['tool_calls'] if call['name'] == 'object_count']
    counts = [call.get('result', {}).get('count') for call in calls]
    pred = int(row['parsed_count'])
    last_count = counts[-1] if counts else None
    fsc_rows.append({
        'index': int(row['index']), 'filename': filename, 'category': row['category'],
        'official_gt': gt, 'stored_gt': int(row['ground_truth_count']), 'prediction': pred,
        'absolute_error': abs(pred - gt), 'signed_error': pred - gt,
        'stored_absolute_error': abs(pred - int(row['ground_truth_count'])),
        'count_calls': len(calls), 'tool_counts': counts,
        'trace_complete': trace_complete, 'tool_calls_complete': calls_complete,
        'last_tool_count': last_count,
        'equals_last_tool': calls_complete and last_count is not None and pred == last_count,
        'equals_any_tool': pred in counts,
        'last_tool_absolute_error': abs(last_count - gt) if last_count is not None else None,
        'last_query': calls[-1].get('arguments', {}).get('query') if calls else None,
        'last_target_image': calls[-1].get('arguments', {}).get('target_image', 0) if calls else None,
        'tool_calls': [{'arguments': call.get('arguments'), 'count': call.get('result', {}).get('count')} for call in calls],
    })
    image_path = image_root / filename
    assert image_path.is_file()
    inputs.append({'dataset': 'FSC147_TEST', 'index': int(row['index']), 'image_path': str(image_path), 'query': row['category'], 'gt_count': gt, 'stored_gt': int(row['ground_truth_count'])})
assert len(fsc_rows) == 1190
assert Counter(row['stored_gt'] - row['official_gt'] for row in fsc_rows) == {3: 1190}
errors = np.array([row['absolute_error'] for row in fsc_rows])
top = sorted(fsc_rows, key=lambda row: (-row['absolute_error'], row['index']))[:math.ceil(.05 * len(fsc_rows))]
copy_top = [row for row in top if row['equals_last_tool']]
summary = {'FSC147': {
    'total': len(fsc_rows), 'mae_official_points': float(errors.mean()),
    'mae_stored_labels': float(data['parsed_count'].sub(data['ground_truth_count']).abs().mean()),
    'rmse_official_points': float(np.sqrt(np.mean(errors ** 2))),
    'label_difference': {'stored_minus_official': 3, 'affected_images': 1190},
    'absolute_error_quantiles': {str(q): float(np.quantile(errors, q)) for q in (0, .5, .9, .95, .99, 1)},
    'under_count': sum(row['signed_error'] < 0 for row in fsc_rows),
    'over_count': sum(row['signed_error'] > 0 for row in fsc_rows),
    'exact': sum(row['absolute_error'] == 0 for row in fsc_rows),
    'top_5pct': {
        'samples': len(top), 'absolute_error_sum': sum(row['absolute_error'] for row in top),
        'absolute_error_share': sum(row['absolute_error'] for row in top) / int(errors.sum()),
        'mae_contribution': sum(row['absolute_error'] for row in top) / len(fsc_rows),
        'remaining_mae': (int(errors.sum()) - sum(row['absolute_error'] for row in top)) / (len(fsc_rows) - len(top)),
        'equals_last_tool_samples': len(copy_top),
        'equals_last_tool_error_share_of_top': sum(row['absolute_error'] for row in copy_top) / sum(row['absolute_error'] for row in top),
        'equals_last_tool_error_share_of_total': sum(row['absolute_error'] for row in copy_top) / int(errors.sum()),
    },
    'overall_count_call_distribution': dict(Counter(row['count_calls'] for row in fsc_rows)),
    'overall_equals_last_tool': sum(row['equals_last_tool'] for row in fsc_rows),
    'complete_tool_call_lists': sum(row['tool_calls_complete'] for row in fsc_rows),
    'truncated_raw_response_cells': sum(not row['trace_complete'] for row in fsc_rows),
    'gt_over_900': [row['index'] for row in fsc_rows if row['official_gt'] > 900],
    'top_two_error_share': sum(row['absolute_error'] for row in top[:2]) / int(errors.sum()),
    'interpretation': 'Equal final/tool numbers are evidence of adoption, not proof of copying or causal attribution.',
}, 'MME': {}}
pd.DataFrame(fsc_rows).drop(columns=['tool_calls']).to_csv(output / 'fsc_agent_rows.csv', index=False)
pd.DataFrame(top).drop(columns=['tool_calls']).to_csv(output / 'fsc_agent_top5pct.csv', index=False)
for name, rows in [('fsc_agent_rows.jsonl', fsc_rows), ('fsc_inputs.jsonl', inputs)]:
    (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

parser_file = repo / 'evaluation/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py'
node = next(node for node in ast.parse(parser_file.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'extract_characters_regex')
namespace = {'re': re}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(parser_file), 'exec'), namespace)
extract = namespace['extract_characters_regex']
for dataset in ('MME-RealWorld-Lite', 'MME-RealWorld-CN'):
    records = []
    for row in pd.read_excel(source / f'VisualAgent-vllm_{dataset}_score.xlsx').fillna('').to_dict('records'):
        prediction = str(row['prediction'])
        answer_tag = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
        choice = extract(answer_tag.group(1).strip().upper() if answer_tag else prediction)
        assert int(choice == row['answer']) == int(row['score']), row['index']
        trace, trace_complete, calls_complete = read_trace(row['raw_response'])
        grounding = [call for call in trace.get('tool_calls', []) if call['name'] == 'grounding_detect']
        statuses = []
        for call in grounding:
            result = call.get('result') or {}
            if call.get('error') or result.get('error') or not isinstance(result.get('boxes'), list):
                statuses.append('unknown_or_error')
            else:
                statuses.append('empty' if not result['boxes'] else 'nonempty')
        group = ('incomplete_tool_trace' if not calls_complete else
                 'no_grounding' if not grounding else
                 'grounding_any_empty' if 'empty' in statuses else
                 'grounding_nonempty_only' if all(status == 'nonempty' for status in statuses) else
                 'grounding_unknown_or_error')
        records.append({
            'dataset': dataset, 'index': int(row['index']), 'question': row['question'],
            'prediction': prediction, 'parsed_choice': choice, 'answer': row['answer'],
            'correct': bool(row['score']), 'selected_E': choice == 'E',
            'E_option': row.get('E'), 'category': row['category'], 'l2_category': row['l2-category'],
            'grounding_group': group, 'grounding_calls': len(grounding),
            'trace_complete': trace_complete, 'tool_calls_complete': calls_complete,
            'grounding_statuses': statuses, 'last_grounding_status': statuses[-1] if statuses else 'no_grounding',
            'grounding_results': [{'arguments': call.get('arguments'), 'result': call.get('result'), 'error': call.get('error')} for call in grounding],
            'tool_sequence': [call['name'] for call in trace.get('tool_calls', [])],
        })
    groups = {}
    for label in ['Overall', *sorted({row['grounding_group'] for row in records})]:
        selected = records if label == 'Overall' else [row for row in records if row['grounding_group'] == label]
        e_rows = [row for row in selected if row['selected_E']]
        groups[label] = {
            'all_samples': len(selected), 'all_error_rate': sum(not row['correct'] for row in selected) / len(selected),
            'E_samples': len(e_rows), 'E_selection_rate': len(e_rows) / len(selected),
            'E_errors': sum(not row['correct'] for row in e_rows),
            'E_error_rate': sum(not row['correct'] for row in e_rows) / len(e_rows) if e_rows else None,
        }
    summary['MME'][dataset] = {'groups': groups, 'E_last_grounding_status': dict(Counter(row['last_grounding_status'] for row in records if row['selected_E']))}
    (output / f'{dataset}_rows.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    pd.DataFrame([row for row in records if row['selected_E']]).drop(columns=['grounding_results']).to_csv(output / f'{dataset}_E_rows.csv', index=False)
(output / 'trace_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
(output / 'trace_manifest.json').write_text(json.dumps({
    'prediction_root': str(source), 'official_fsc_annotation': str(annotation),
    'fsc_annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
    'fsc_prediction_sha256': hashlib.sha256(fsc_file.read_bytes()).hexdigest(),
    'choice_parser': str(parser_file),
    'scope': 'New-data 64-GPU step80; FSC uses official point count, MME uses saved official score and parser',
}, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
