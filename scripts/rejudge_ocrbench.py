#!/usr/bin/env python3
"""Apply the DeepEyesV2 judge template to rule-rejected OCRBench predictions."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


HME = 'Handwritten Mathematical Expression Recognition'
BASE = Path(__file__).resolve().parents[2]
TEMPLATE = BASE / 'DeepEyesV2/evaluation/VLMEvalKit/vlmeval/dataset/utils/judge_prompt/verify.md'


def rule_correct(prediction, answers, category):
    def normalize(text):
        if category == HME:
            return text.strip().replace('\n', ' ').replace(' ', '')
        return text.lower().strip().replace('\n', ' ')

    return any(normalize(answer) in normalize(prediction) for answer in answers)


def parse_verdict(content, finish_reason):
    verdicts = re.findall(r'\\boxed\s*\{\s*(Yes|No)\s*\}', content, flags=re.I)
    if finish_reason == 'length' or not verdicts:
        raise ValueError('Missing complete boxed verdict')
    return verdicts[-1].lower() == 'yes'


def judge(task, protocol, api_key):
    started = time.monotonic()
    record = dict(task, verdict=None)
    body = {key: protocol[key] for key in ('model', 'temperature', 'max_tokens', 'thinking')}
    body['messages'] = [{'role': 'user', 'content': task['prompt']}]
    request = urllib.request.Request(
        protocol['endpoint'], data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        choice = payload['choices'][0]
        content = choice['message']['content'] or ''
        record.update(response=content, usage=payload.get('usage'),
                      model_returned=payload.get('model'), finish_reason=choice.get('finish_reason'))
        record['verdict'] = parse_verdict(content, choice.get('finish_reason'))
    except Exception as error:
        # API failures remain unresolved; do not log credentials or response bodies from errors.
        record.update(error_type=type(error).__name__, http_status=getattr(error, 'code', None))
    record['seconds'] = round(time.monotonic() - started, 3)
    return record


def summarize(rows, cache):
    groups = {}
    for category in ['Overall', *sorted({row['category'] for row in rows})]:
        selected = rows if category == 'Overall' else [row for row in rows if row['category'] == category]
        recovered = [row['index'] for row in selected
                     if row.get('request_key') and cache.get(row['request_key'], {}).get('verdict') is True]
        unresolved = [row['index'] for row in selected
                      if row.get('request_key') and cache.get(row['request_key'], {}).get('verdict') is None]
        correct = sum(row['rule_correct'] for row in selected)
        groups[category] = {
            'total': len(selected), 'rule_correct': correct, 'judge_recovered': len(recovered),
            'combined_correct': correct + len(recovered), 'recovered_indices': recovered,
            'unresolved_indices': unresolved,
            'empty_predictions': sum(not row['prediction'].strip() for row in selected),
            'prediction_api_failed': sum('Failed to obtain answer via API' in row['prediction'] for row in selected),
        }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSON mapping run labels to prediction XLSX/JSON paths')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--api-base', default=os.getenv('LLM_AS_A_JUDGE_BASE', 'http://43.155.134.160:8080/v1'))
    parser.add_argument('--model', default=os.getenv('LLM_AS_A_JUDGE_MODEL', 'deepseek-v4-flash'))
    parser.add_argument('--key-file', type=Path, default=BASE / 'secrets/judge_api_43_155_134_160_key.txt')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    template = TEMPLATE.read_text()
    protocol = {
        'model': args.model, 'endpoint': args.api_base.rstrip('/') + '/chat/completions',
        'temperature': 0, 'max_tokens': 1024, 'thinking': {'type': 'disabled'},
        'template_source': str(TEMPLATE), 'template': template, 'system': None,
        'scope': 'All nonempty, non-API-failed OCRBench rule-rejected predictions',
        'reference_policy': 'List all reference alternatives without modifying the candidate',
    }
    manifest = json.loads(args.manifest.read_text())
    runs = {}
    tasks = {}
    for label, filename in manifest.items():
        path = Path(filename)
        if path.suffix == '.json':
            source = json.loads(path.read_text())
        else:
            import pandas as pd
            source = pd.read_excel(path).fillna('').to_dict(orient='records')
        rows = []
        for raw in source:
            answers = ast.literal_eval(raw['answer']) if isinstance(raw['answer'], str) else raw['answer']
            prediction = str(raw['prediction'])
            correct = rule_correct(prediction, answers, raw['category'])
            row = {'index': int(raw['index']), 'question': raw['question'], 'category': raw['category'],
                   'answers': answers, 'prediction': prediction, 'rule_correct': correct}
            if not correct and prediction.strip() and 'Failed to obtain answer via API' not in prediction:
                references = '\n'.join(f'Alternative {i + 1}: {answer.strip()}' for i, answer in enumerate(answers))
                prompt = template + f'【用户问题】:{raw["question"]}\n【参考答案】：{references}\n【模型回答】：{prediction}'
                identity = json.dumps([protocol, prompt], ensure_ascii=False, sort_keys=True)
                task_key = hashlib.sha256(identity.encode()).hexdigest()
                row['request_key'] = task_key
                tasks[task_key] = {'key': task_key, 'question': raw['question'], 'answers': answers,
                                   'prediction': prediction, 'prompt': prompt}
            rows.append(row)
        if len(rows) != 1000 or {row['index'] for row in rows} != set(range(1000)):
            raise ValueError(f'{label}: expected all 1000 OCRBench indices')
        official = json.loads(path.with_name(path.stem + '_score.json').read_text())
        if (sum(row['rule_correct'] for row in rows) != official['Final Score']
                or sum(row['rule_correct'] for row in rows if row['category'] == HME) != official[HME]):
            raise ValueError(f'{label}: rule scores disagree with saved OCRBench scores')
        runs[label] = rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    cache_path = args.output_dir / 'requests.jsonl'
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            record = json.loads(line)
            cache[record['key']] = record
    pending = [task for key, task in tasks.items() if cache.get(key, {}).get('verdict') is None]
    print(f'Runs: {len(runs)}; unique candidates: {len(tasks)}; pending: {len(pending)}', flush=True)
    api_key = args.key_file.read_text().strip() if pending else ''
    with cache_path.open('a') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge, task, protocol, api_key) for task in pending]
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            cache[record['key']] = record
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            if number % 10 == 0 or record['verdict'] is None or number == len(pending):
                print(f'Completed {number}/{len(pending)}; verdict={record["verdict"]}; error={record.get("error_type")}', flush=True)

    summary = {'protocol': protocol, 'runs': {}, 'unique_candidates': len(tasks)}
    with (args.output_dir / 'predictions.jsonl').open('w') as stream:
        for label, rows in runs.items():
            summary['runs'][label] = summarize(rows, cache)
            print(label, json.dumps(summary['runs'][label]['Overall']), flush=True)
            for row in rows:
                verdict = cache.get(row.get('request_key'), {}).get('verdict')
                combined = True if row['rule_correct'] else (verdict if row.get('request_key') else False)
                stream.write(json.dumps(dict(row, run=label, judge_verdict=verdict, combined_correct=combined), ensure_ascii=False) + '\n')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return 1 if any(run['Overall']['unresolved_indices'] for run in summary['runs'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
