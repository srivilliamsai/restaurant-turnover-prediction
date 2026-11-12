"""Small deterministic tests of target isolation, selection and failure paths."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import features
import modeling
import workflow
from core import Budget, Cache, BudgetExceeded, CONFIRM, save


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class Tests(unittest.TestCase):
    def test_cap_retains_rows_originals_and_uses_training_only(self):
        training = np.array([1., 2., 3., 4., 100.]) * 1e6
        original = training.copy()
        for name, threshold in [('Q99', 96.16e6), ('Q995', 98.08e6)]:
            result, record = features.training_target(training, name)
            self.assertEqual(len(result), len(training))
            self.assertAlmostEqual(record['threshold_INR'], threshold, places=5)
            self.assertEqual(record['capped_rows'], 1)
            np.testing.assert_array_equal(result[:-1], original[:-1] / 1e6)
            self.assertAlmostEqual(result[-1], threshold / 1e6)
        np.testing.assert_array_equal(training, original)

    def test_square_root_decodes_second_moment_in_correct_units(self):
        class UniformTwoToFour:
            def mean_of_square(self, logits):
                return Tensor([28 / 3])
            def mean(self, logits):
                return Tensor([3.])
        result = modeling.decode_full({'criterion': UniformTwoToFour(), 'logits': None})
        self.assertAlmostEqual(result['prediction'][0], 28 / 3 * 1e6)
        self.assertAlmostEqual(result['naive_squared_mean'][0], 9e6)
        target, _ = features.training_target(np.array([4e6, 9e6]), 'SQ2')
        np.testing.assert_array_equal(target, [2., 3.])

    def test_selection_requires_both_splits_and_practical_gain(self):
        def row(name, gain, weight=.25, positive=True):
            return {'name': name, 'mean_relative_gain': gain, 'weight': weight,
                    'gain_42': 1., 'gain_71': 1. if positive else -1.}
        rows = [row('Q99', .00199), row('SQ2', .01, positive=False), row('Q995', .003, .5),
                row('SQ2', .003, .25), row('Q99', .003, .25)]
        self.assertEqual(workflow.selection(rows)[0]['name'], 'Q99')
        self.assertEqual(workflow.selection(rows[:2]), [])

    def test_confirmation_gates_each_required_condition(self):
        metrics = {str(s): {'gain': 100., 'relative_gain': .003} for s in CONFIRM}
        self.assertTrue(workflow.qualifies(metrics, {'ci95_INR': [1., 200.]}))
        self.assertFalse(workflow.qualifies(metrics, {'ci95_INR': [-1., 200.]}))
        metrics['city']['relative_gain'] = -.01001
        self.assertFalse(workflow.qualifies(metrics, {'ci95_INR': [1., 200.]}))
        metrics['city']['relative_gain'] = -.01
        metrics['48211']['gain'] = -1.
        self.assertFalse(workflow.qualifies(metrics, {'ci95_INR': [1., 200.]}))

    def test_delivery_reserve_and_persistent_stage_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(tmp)
            start = budget.authorization['started_epoch']
            with patch('core.time.time', return_value=start + 6.1 * 3600):
                self.assertEqual(budget.remaining('screen'), 0.)
                self.assertGreater(budget.remaining('export'), 0.)
            save(budget.path, {'seconds': 0., 'by_stage': {'pilot': 3600.}})
            self.assertEqual(budget.remaining('pilot'), 0.)
            with self.assertRaises(BudgetExceeded):
                with budget.charge('pilot'):
                    pass

    def test_cache_rejects_changed_treatment_corruption_and_partial_write(self):
        class Unlimited:
            def require(self, estimate=0):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            cache = Cache(tmp, Unlimited())
            identity = {'quantile': .99, 'training_targets': 'original'}
            compute = lambda: ({'prediction': np.array([1., 2.])}, {'threshold_INR': 3.})
            cache.get('fold', identity, compute)
            with self.assertRaises(AssertionError):
                cache.get('fold', {**identity, 'quantile': .995}, compute)
            (cache.root / 'fold.npz').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                cache.get('fold', identity, compute)
            np.savez_compressed(cache.root / 'partial.npz', prediction=[1.])
            with self.assertRaises(ValueError):
                cache.get('partial', identity, compute)


if __name__ == '__main__':
    unittest.main(verbosity=2)
