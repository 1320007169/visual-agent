import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.online_faults import OnlineFaultController
from verl.utils.reward_score import visual_agent_thyme as reward
from verl.workers.reward_manager.naive_async import AsyncNaiveRewardManager


ROOT = Path(__file__).resolve().parents[1]


class RewardFailureRecoveryTest(unittest.TestCase):
    def test_reward_errors_are_not_retried_without_metadata(self):
        manager = Mock(side_effect=[RuntimeError('reward failure'), torch.zeros(1)])
        with self.assertRaisesRegex(RuntimeError, 'reward failure'):
            compute_reward(object(), manager)
        self.assertEqual(manager.call_count, 1)

    def test_unresolved_judge_keeps_metadata_and_is_excluded_from_grpo_and_crt(self):
        source = 'visual-agent-ocr'
        controller = OnlineFaultController(str(ROOT / 'configs/online_crt_v1.json'))
        key, _ = controller._bucket(source, {})
        import json
        batch = DataProto.from_dict(tensors={
            'prompts': torch.zeros(4, 1, dtype=torch.long),
            'responses': torch.tensor([[1], [2], [3], [1]]),
            'attention_mask': torch.ones(4, 2, dtype=torch.long),
        }, non_tensors={
            'data_source': np.array([source] * 4, dtype=object),
            'extra_info': np.array([{'data_source': source}] * 4, dtype=object),
            'reward_model': np.array([{'ground_truth': 'cat'}] * 4, dtype=object),
            'uid': np.array(['group'] * 4, dtype=object),
            'online_fault': np.array([json.dumps({'bucket_key': key})] * 4, dtype=object),
            'online_fault_injected': np.ones(4, dtype=bool),
            'online_fault_level': np.zeros(4, dtype=int),
        })
        answers = ['question', '<answer>cat</answer>', '<answer>unknown</answer>', '<answer>dog</answer>']
        tokenizer = SimpleNamespace(decode=lambda ids: answers[int(ids[0])])
        manager = AsyncNaiveRewardManager(tokenizer, 0)
        with patch.object(reward, 'judge_match', side_effect=lambda q, a, g, **kw: None if a == 'unknown' else False):
            scores, extra = compute_reward(batch, manager)
        self.assertEqual(extra['reward_valid'], [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(extra['acc'], [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(scores[1].item(), 0.0)
        advantages, _ = compute_grpo_outcome_advantage(
            scores, torch.ones_like(scores), batch.non_tensor_batch['uid'], reward_valid_mask=extra['reward_valid'],
        )
        self.assertEqual(advantages[1].item(), 0.0)
        expected, _ = compute_grpo_outcome_advantage(
            scores[[0, 2, 3]], torch.ones(3, 1), np.array(['group'] * 3),
        )
        torch.testing.assert_close(advantages[[0, 2, 3]], expected)
        batch.non_tensor_batch.update({k: np.array(v) for k, v in extra.items()})
        metrics = controller.update(batch, 4)
        self.assertEqual(metrics[f'online_faults/{source}/ocr_read/qualified_groups'], 0)
        self.assertEqual(controller.buckets[key]['pending']['groups'], 0)


    def test_validation_excludes_unresolved_answers_and_does_not_select_a_best_checkpoint(self):
        path = ROOT / 'reinforcement_learning/verl/trainer/ppo/ray_trainer.py'
        tree = ast.parse(path.read_text())
        names = {'_compute_visual_tool_metrics', '_visual_accuracy_macro_mean'}
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_validate')
        start = next(i for i, node in enumerate(method.body) if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Tuple)
                     and getattr(node.target.elts[0], 'id', None) == 'key_info')
        method.body = method.body[start:]
        method.args.args.extend([ast.arg(arg='reward_extra_infos_dict'), ast.arg(arg='sample_rollout_traces')])
        ast.fix_missing_locations(method)
        source = 'visual-agent-ocr'
        for valid in ([1, 0], [1, 1], [0, 0]):
            with self.subTest(valid=valid):
                namespace = {'np': np, 'defaultdict': defaultdict, 'Optional': Optional,
                             'process_validation_metrics': process_validation_metrics,
                             'reward_extra_infos_dict': {'acc': [1, 0], 'reward_valid': valid},
                             'sample_scores': [1, 0], 'data_source_lst': [np.array([source] * 2)],
                             'sample_inputs': ['q1', 'q2'], 'sample_source_metadata': [{}, {}],
                             'sample_rollout_traces': [{'tool_calls': []}] * 2}
                exec(compile(ast.Module(body=[*helpers, method], type_ignores=[]), str(path), 'exec'), namespace)
                metrics = namespace['_validate'](None, namespace['reward_extra_infos_dict'], namespace['sample_rollout_traces'])
                self.assertEqual(metrics['val-aux/reward_valid_fraction'], sum(valid) / 2)
                if any(valid):
                    self.assertEqual(metrics[f'val-core/{source}/acc/mean@1'], 1 if valid == [1, 0] else 0.5)
                else:
                    self.assertNotIn(f'val-core/{source}/acc/mean@1', metrics)
                self.assertEqual('val-core/visual-agent/acc/macro_mean' in metrics, all(valid))
                tool_metrics = namespace['_compute_visual_tool_metrics'](
                    [source] * 2, [{'tool_calls': []}] * 2, [1, 0], reward_valid=valid)
                self.assertEqual(tool_metrics[f'train-tools/{source}/reward_valid_fraction'], sum(valid) / 2)
                if valid == [1, 0]:
                    self.assertEqual(tool_metrics[f'train-tools/{source}/acc_mean'], 1)


if __name__ == '__main__':
    unittest.main()
