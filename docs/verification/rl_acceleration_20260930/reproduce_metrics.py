import ast
import __future__
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

root = Path(__file__).resolve().parents[3]
source = root / 'reinforcement_learning/verl/trainer/ppo/ray_trainer.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'compute_entropy_metrics', 'compute_truncation_metrics', 'compute_rollout_consistency_metrics'}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
ns = {'np': np, 'torch': torch, 'agg_loss': lambda loss_mat, loss_mask, loss_agg_mode: (loss_mat * loss_mask).sum() / loss_mask.sum()}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec', flags=__future__.annotations.compiler_flag), ns)

# An unclosed tool-call quote in an observation must not classify the next assistant answer.
data = SimpleNamespace(batch={'responses': torch.tensor([[10, 2, 30, 40]]),
                             'loss_mask': torch.tensor([[1, 0, 0, 1]])})
entropy = ns['compute_entropy_metrics'](data, torch.tensor([[1., 2., 3., 4.]]), 'token-mean', (2, 3))
print(json.dumps({'case': 'observation_tag_contaminates_entropy_category', 'actual': entropy,
                  'expected_tool_call_tokens': 0}))

data = SimpleNamespace(batch={'truncated': torch.tensor([True, False]), 'loss_mask': torch.tensor([[0, 0], [1, 1]])},
                      non_tensor_batch={'__num_turns__': np.array([1, 1]), 'stream_id': np.array(['agent', 'native'])})
print(json.dumps({'case': 'three_executed_turns_trimmed_to_one', 'actual': ns['compute_truncation_metrics'](data),
                  'missing_fields': ['executed_tool_use', 'truncated_loss_token_share', 'think_entropy', 'answer_entropy']}))

path = root / 'reinforcement_learning/verl/workers/rollout/rollout_consistency.py'
spec = importlib.util.spec_from_file_location('consistency', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
_, _, stats = module.align_sampled_turns([1, 2, 9], ['assistant'], 9, [7, 8], [None])
print(json.dumps({'case': 'missing_logprobs_reported_as_mismatch', 'actual': stats,
                  'reported_exact_match_rate': stats['matched_turns'] / stats['turns']}))
