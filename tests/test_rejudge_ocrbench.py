import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('rejudge_ocrbench', Path(__file__).resolve().parents[1] / 'scripts/rejudge_ocrbench.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RejudgeOcrBenchTest(unittest.TestCase):
    def test_hme_keeps_case_and_backslashes_but_ignores_spaces_and_newlines(self):
        answer = '\\frac { 1 } { x }\n'
        self.assertTrue(module.rule_correct('Result: \\frac{1}{x}', [answer], module.HME))
        self.assertFalse(module.rule_correct('\\frac{1}{X}', [answer], module.HME))
        self.assertFalse(module.rule_correct('\\\\frac{1}{x} + \\\\frac{1}{y}',
                                             ['\\frac{1}{x}+\\frac{1}{y}'], module.HME))
        self.assertTrue(module.rule_correct('THE RED SIGN', ['blue', 'red sign'], 'Regular Text Recognition'))

    def test_verdict_requires_complete_final_box_and_handles_closing_tag_typo(self):
        self.assertTrue(module.parse_verdict('<最终结果>\\boxed{Yes}<最终结果>', 'stop'))
        self.assertFalse(module.parse_verdict('Yes in discussion. \\boxed{No}', 'stop'))
        for text, finish in [('Yes', 'stop'), ('\\boxed{Yes}', 'length')]:
            with self.assertRaises(ValueError):
                module.parse_verdict(text, finish)

    def test_unresolved_requests_are_separate_from_rejections_and_empty_predictions(self):
        rows = [
            {'index': 0, 'category': module.HME, 'prediction': 'a', 'rule_correct': True},
            {'index': 1, 'category': module.HME, 'prediction': 'b', 'rule_correct': False, 'request_key': 'yes'},
            {'index': 2, 'category': module.HME, 'prediction': 'c', 'rule_correct': False, 'request_key': 'no'},
            {'index': 3, 'category': module.HME, 'prediction': 'd', 'rule_correct': False, 'request_key': 'timeout'},
            {'index': 4, 'category': module.HME, 'prediction': '', 'rule_correct': False},
        ]
        result = module.summarize(rows, {'yes': {'verdict': True}, 'no': {'verdict': False}})['Overall']
        self.assertEqual(result['combined_correct'], 2)
        self.assertEqual(result['recovered_indices'], [1])
        self.assertEqual(result['unresolved_indices'], [3])
        self.assertEqual(result['empty_predictions'], 1)


if __name__ == '__main__':
    unittest.main()
