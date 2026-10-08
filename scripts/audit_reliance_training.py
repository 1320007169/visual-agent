#!/usr/bin/env python3
"""Audit matched reliance runs by training step, source, branch and original correctness."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from prepare_reliance_pairs import normalized_ocr_text, tool_faults


def summarize(buckets):
    report = {}
    for key, counts in buckets.items():
        n = counts['trajectories']
        report[key] = {**counts, 'acc': counts['correct'] / n, 'mean_reward': counts['reward_sum'] / n,
                       'tools_per_trajectory': counts['tool_calls'] / n,
                       'mixed_group_fraction': counts['mixed_groups'] / counts['groups'],
                       'bypass_rate': counts['bypass_returned_truth_trajectories'] / n,
                       'replay_violation_rate': counts['replay_violation_trajectories'] / n}
    return report


def audit_step(path, data, rollout_n):
    groups = defaultdict(list)
    buckets = defaultdict(Counter)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            meta = row['source_metadata']
            index = int(meta['dataset_index'])
            sample = data[index]
            if (sample['data_source'], sample['source_image'], sample['question'].strip()) != (
                    meta['data_source'], meta['source_image'], meta['question'].strip()):
                raise ValueError(f'Training data does not match rollout dataset_index={index}')
            branch = sample.get('reliance_branch') or 'ordinary'
            if (meta.get('reliance_branch') or 'ordinary') != branch:
                raise ValueError(f'Wrong ablation arm at dataset_index={index}')
            source = sample['data_source'] + '|' + (sample.get('original_source') or '')
            original = sample.get('original_correct')
            key = f'{source}#{branch}#original_correct={original}'
            counts = buckets[key]
            correct = float(row['acc']) > 0
            groups[index].append((key, correct))
            calls = (row.get('rollout_trace') or {}).get('tool_calls') or []
            counts['trajectories'] += 1
            counts['correct'] += correct
            counts['reward_sum'] += float(row['score'])
            counts['tool_calls'] += len(calls)
            counts['tool_errors'] += sum(call.get('status') != 'success' for call in calls)
            if branch != 'counterfactual':
                continue
            fault = json.loads(sample['reliance_fault'])
            injected = json.loads(fault['observation'])
            images, aliases = list(sample['images']), {}
            answers = {normalized_ocr_text(answer) for answer in [sample['solution'], *(sample.get('answer_aliases') or [])]} - {''}
            violation, bypass = False, False
            for call_number, call in enumerate(calls):
                if call.get('status') != 'success':
                    continue
                tool, arguments = call['tool'], call['arguments']
                observed = json.loads(call['model_observation'])
                if tool == fault['tool'] and tool_faults.normalize_replay_arguments(tool, arguments, images, aliases) == (
                        tool_faults.normalize_replay_arguments(tool, fault['arguments'], images, aliases)):
                    counts['equivalent_recalls'] += 1
                    replay_ok = bool(call.get('replayed_fault')) and observed == injected
                    violation |= not replay_ok
                    returned_truth = (any(answer in normalized_ocr_text(observed.get('text', '')) for answer in answers)
                                      if tool == 'ocr_read' else str(observed.get('count')) == sample['solution'].strip())
                    bypass |= not replay_ok and returned_truth
                    counts['replays'] += replay_ok
                returned = [f'crop:{call_number}:{i}' for i in range(call.get('returned_image_count', 0))]
                if returned:
                    if tool == 'crop_zoom' and 'raw_result' not in call:
                        raise ValueError('Crop trace lacks raw_result; image equivalence cannot be audited')
                    tool_faults.record_image_aliases(tool, arguments, call.get('raw_result', {}), images, returned, aliases)
                    images.extend(returned)
            counts['replay_violation_trajectories'] += violation
            counts['bypass_returned_truth_trajectories'] += bypass
    for index, outcomes in groups.items():
        if len(outcomes) != rollout_n:
            raise ValueError(f'Incomplete GRPO group at {path.name}, dataset_index={index}: {len(outcomes)} != {rollout_n}')
        key = outcomes[0][0]
        correct = sum(value for _, value in outcomes)
        buckets[key]['groups'] += 1
        buckets[key]['all_correct_groups' if correct == rollout_n else 'all_wrong_groups' if correct == 0 else 'mixed_groups'] += 1
    sources = defaultdict(Counter)
    for key, counts in buckets.items():
        sources[key.rsplit('#original_correct=', 1)[0]].update(counts)
    total = sum(counts['trajectories'] for counts in buckets.values())
    return {'trajectories': total, 'mean_reward': sum(counts['reward_sum'] for counts in buckets.values()) / total,
            'tools_per_trajectory': sum(counts['tool_calls'] for counts in buckets.values()) / total,
            'by_source_branch': summarize(sources), 'by_source_branch_original_correct': summarize(buckets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--rollout-dir', type=Path, action='append', required=True)
    parser.add_argument('--validation-dir', type=Path, action='append', default=[])
    parser.add_argument('--steps', default='1-20')
    parser.add_argument('--rollout-n', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    data = pq.read_table(args.data_dir / 'train.parquet').to_pylist()
    low, high = map(int, args.steps.split('-'))
    report = {'data_dir': str(args.data_dir), 'training': {}, 'validation': {}}
    for directory in args.rollout_dir:
        for path in sorted(directory.glob('*.jsonl')):
            if not path.stem.isdigit() or not low <= int(path.stem) <= high:
                continue
            if path.stem in report['training']:
                raise ValueError(f'Duplicate training step across sessions: {path.stem}')
            report['training'][path.stem] = audit_step(path, data, args.rollout_n)
    for directory in args.validation_dir:
        for path in sorted(directory.glob('*.jsonl')):
            if not path.stem.isdigit() or not low <= int(path.stem) <= high:
                continue
            sources = defaultdict(list)
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    sources[row['source_metadata']['data_source']].append(float(row['acc']))
            means = {source: sum(values) / len(values) for source, values in sources.items()}
            report['validation'].setdefault(path.stem, []).append({
                'file': str(path), 'acc_by_source': means, 'macro_mean': sum(means.values()) / len(means)})
    if not report['training']:
        raise ValueError('No completed training steps found in the requested range')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Audited {len(report['training'])} training steps: {args.output}")


if __name__ == '__main__':
    main()
