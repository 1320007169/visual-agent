#!/usr/bin/env python3
"""Compare full FSC agent runs scored against official point counts."""

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    annotation = next(path for path in protocol['sha256'] if Path(path).name == 'annotation_FSC147_384.json')
    points = json.loads(Path(annotation).read_text())
    arms, summary = {}, {}
    for arm in ('text_only', 'auto_exemplar'):
        paths = [p for p in (root / arm).rglob('*_FSC147_TEST_score.xlsx') if not p.is_symlink()]
        if len(paths) != 1:
            raise ValueError(f'Expected one completed FSC score file for {arm}, got {len(paths)}')
        frame = pd.read_excel(paths[0])
        if len(frame) != 1190 or not frame['index'].is_unique:
            raise ValueError(f'Expected exactly 1190 unique predictions for {arm}')
        if any(int(row['ground_truth_count']) != len(points[Path(row['image_path']).name]['points'])
               for row in frame.to_dict('records')):
            raise ValueError(f'Non-official point labels in {arm}')
        frame = frame.set_index('index').sort_index()
        frame['absolute_error'] = (frame['parsed_count'] - frame['ground_truth_count']).abs()
        arms[arm] = frame
        metrics = pd.read_csv(str(paths[0]).replace('_score.xlsx', '_acc.csv')).iloc[0].to_dict()
        summary[arm] = {key: None if pd.isna(value) else value for key, value in metrics.items()}
    if not arms['text_only'].index.equals(arms['auto_exemplar'].index):
        raise ValueError('The two arms have different sample indices')
    paired = pd.DataFrame({
        'category': arms['text_only']['category'], 'ground_truth': arms['text_only']['ground_truth_count'],
        'text_prediction': arms['text_only']['prediction'], 'auto_prediction': arms['auto_exemplar']['prediction'],
        'text_count': arms['text_only']['parsed_count'], 'auto_count': arms['auto_exemplar']['parsed_count'],
        'text_error': arms['text_only']['absolute_error'], 'auto_error': arms['auto_exemplar']['absolute_error'],
    })
    paired['error_change'] = paired['auto_error'] - paired['text_error']
    paired.to_csv(root / 'paired_examples.csv')
    summary['paired'] = {
        'both_valid': int(paired['error_change'].notna().sum()),
        'improved': int((paired['error_change'] < 0).sum()),
        'degraded': int((paired['error_change'] > 0).sum()),
        'equal_error': int((paired['error_change'] == 0).sum()),
        'valid_to_invalid': int((paired['text_count'].notna() & paired['auto_count'].isna()).sum()),
        'invalid_to_valid': int((paired['text_count'].isna() & paired['auto_count'].notna()).sum()),
        'both_invalid': int((paired['text_count'].isna() & paired['auto_count'].isna()).sum()),
    }
    status = pd.read_csv(root / 'status.tsv', sep='\t')
    if len(status) != 2 or set(status['arm']) != set(arms) or (status['exit_code'] != 0).any():
        raise ValueError('Both arms must finish successfully before comparison')
    summary['timing'] = {'scope': protocol['timing'], 'rows': status.to_dict('records')}
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# FSC counting backend A/B', '',
             'Same step80 checkpoint, prompt, count-only observations and official point labels.', '',
             '| Arm | MAE | RMSE | MAE_valid | RMSE_valid | Invalid | End-to-end seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        result = summary[arm]
        seconds = int(status.set_index('arm').loc[arm, 'seconds'])
        values = ['undefined' if result[key] is None else f'{result[key]:.4f}'
                  for key in ('MAE', 'RMSE', 'MAE_valid', 'RMSE_valid')]
        lines.append(f"| {arm} | {' | '.join(values)} | {int(result['invalid_count'])} | {seconds} |")
    lines += ['', f"Paired outcomes: {json.dumps(summary['paired'])}", '',
              'Full-split MAE/RMSE remain undefined when invalid answers exist. Do not compare valid-only metrics without checking their sample coverage.', '',
              protocol['timing'], '', 'Per-example results: `paired_examples.csv`.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, allow_nan=False))
