"""Rescore all cached FSC agent predictions against official point counts."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import pandas as pd


output = Path(__file__).resolve().parent
repo = output.parents[2]
annotation = repo.parent / 'datasets/fsc147/annotation_FSC147_384.json'
points = json.loads(annotation.read_text())
spec = importlib.util.spec_from_file_location('fsc_metrics', repo / 'evaluation/VLMEvalKit/vlmeval/dataset/utils/fsc147.py')
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
summaries = []
for source in sorted((repo / 'outputs/vlmeval').rglob('VisualAgent-vllm_FSC147_TEST.xlsx')):
    if source.is_symlink():
        continue
    started = time.monotonic()
    data = pd.read_excel(source).fillna('')
    assert len(data) == 1190 and data['index'].is_unique, source
    answers = [len(points[Path(path).name]['points']) for path in data['image_path']]
    historical, extracted = metrics_module.counting_metrics(data['answer'].astype(int).tolist(), data['prediction'].tolist())
    corrected, corrected_extracted = metrics_module.counting_metrics(answers, data['prediction'].tolist())
    assert extracted == corrected_extracted
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    name = f'corrected_{source_hash[:12]}'
    destination = output / name
    destination.mkdir(exist_ok=True)
    rows = pd.DataFrame({
        'index': data['index'], 'image': [Path(p).name for p in data['image_path']],
        'category': data['category'], 'prediction': data['prediction'], 'parsed_count': extracted,
        'historical_answer': data['answer'], 'official_answer': answers,
    })
    rows['historical_absolute_error'] = (rows['parsed_count'] - rows['historical_answer'].astype(int)).abs()
    rows['official_absolute_error'] = (rows['parsed_count'] - rows['official_answer']).abs()
    rows.to_csv(destination / 'scores.csv', index=False)
    pd.DataFrame([corrected]).to_csv(destination / 'metrics.csv', index=False)
    summary = {
        'run': name, 'source': str(source), 'source_sha256': source_hash,
        'historical': historical, 'corrected': corrected, 'rescore_seconds': time.monotonic() - started,
    }
    summaries.append(summary)
    print(name, source.parent, 'MAE_valid:', historical['MAE_valid'], '->', corrected['MAE_valid'], flush=True)
(output / 'summary.json').write_text(json.dumps({
    'annotation': str(annotation), 'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
    'policy': 'Saved predictions unchanged; full-split invalid handling and answer extraction unchanged; only reference point counts corrected. Rescore time is CPU scoring time, not inference time.',
    'runs': summaries,
}, ensure_ascii=False, indent=2) + '\n')
pd.DataFrame([{
    'run': r['run'], 'source': r['source'],
    **{'old_' + k: v for k, v in r['historical'].items()},
    **{'new_' + k: v for k, v in r['corrected'].items()},
} for r in summaries]).to_csv(output / 'comparison.csv', index=False)
