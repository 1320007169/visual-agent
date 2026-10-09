"""Compare production text-only CountGD++ and automatic exemplars on fixed inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument('inputs', type=Path)
parser.add_argument('--limit', type=int, default=0)
args = parser.parse_args()
output = Path(__file__).resolve().parent
repo = output.parents[2]
pipeline = Path('/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/groundingdino_offline_pipeline')
helpers = runpy.run_path(str(repo / 'scripts/compare_count_backends.py'))
helpers['load_env_file'](pipeline / '.env')
sys.path.insert(0, str(pipeline / 'src'))
sys.path.insert(0, str(pipeline))
rows = list(map(json.loads, args.inputs.read_text().splitlines()))
if args.limit:
    rows = rows[:args.limit]
dataset = rows[0]['dataset']
path = output / f'{dataset}_tool_calls.jsonl'
completed = set()
if path.exists():
    for row in map(json.loads, path.open()):
        completed.add(row['index'])
rows = [row for row in rows if row['index'] not in completed]
manifest = {
    'inputs': str(args.inputs.resolve()), 'inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
    'dataset': dataset, 'threshold': .23, 'automatic_exemplars': 3,
    'ground_truth_boxes_used': False, 'device': 'CUDA_VISIBLE_DEVICES=1; cuda:0',
    'backend_source': str(pipeline / 'src/vts/tools/counting.py'),
    'backend_source_sha256': hashlib.sha256((pipeline / 'src/vts/tools/counting.py').read_bytes()).hexdigest(),
    'policy': 'Both production backends receive identical image and query; automatic boxes come only from model predictions',
}
(output / f'{dataset}_tool_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
text = helpers['build_backend'](pipeline, 'countgd_plusplus', 'cuda:0')
pseudo = helpers['build_backend'](pipeline, 'countgd_plusplus_pseudo', 'cuda:0')
text._load()
for attribute in ('_model', '_transform', '_torch', '_nested_tensor'):
    setattr(pseudo, attribute, getattr(text, attribute))
started = time.monotonic()
print(f'Dataset: {dataset}; pending: {len(rows)}; existing: {len(completed)}', flush=True)
with path.open('a') as stream:
    for number, row in enumerate(rows, 1):
        record = dict(row)
        for name, backend in [('text_only', text), ('auto_exemplar', pseudo)]:
            tick = time.monotonic()
            result = backend.count(row['image_path'], row['query'])
            record[name] = {key: result.get(key) for key in (
                'count', 'confidence', 'boxes', 'image_size', 'count_mode', 'first_pass_count',
                'pseudo_exemplar_count', 'pseudo_exemplar_boxes',
            )}
            record[name]['seconds'] = time.monotonic() - tick
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        if number % 20 == 0 or number == len(rows):
            print(f'Completed {number}/{len(rows)}; elapsed={time.monotonic() - started:.1f}s', flush=True)
