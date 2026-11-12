"""Exercise consequential V20 isolation, cache, selection and budget behavior."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import features
import new_models
import workflow
from core import ROOT, Budget, Cache, CONFIRM, Engine, save


class V20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = Engine(out=ROOT / 'preflight_results')
        cls.raw = cls.e.train.iloc[:240].copy()

    def test_target_and_id_cannot_change_features(self):
        other = self.raw.copy()
        other['Annual Turnover'] = -1e20
        other['Registration Number'] = -9
        a, b = new_models.Codec().fit(self.raw), new_models.Codec().fit(other)
        self.assertEqual(a.vocabulary, b.vocabulary)
        np.testing.assert_array_equal(a.transform(self.raw), b.transform(other))
        self.assertEqual(len(a.vocabulary), 8)

    def test_query_categories_do_not_refit_codec(self):
        codec = new_models.Codec().fit(self.raw)
        query = self.raw.iloc[:2].copy()
        query['City'] = 'UNSEEN_V20'
        before = json.dumps(codec.vocabulary, sort_keys=True)
        self.assertTrue(np.isfinite(codec.transform(query)).all())
        self.assertTrue(codec.typed(query)['City'].isna().all())
        self.assertEqual(before, json.dumps(codec.vocabulary, sort_keys=True))

    def test_fixed_query_layout_preserves_real_rows(self):
        raw = self.e.train.iloc[:699].copy()
        blocks = list(new_models.query_blocks(raw, self.raw.iloc[:1], 256))
        self.assertEqual([len(block) for block, _ in blocks], [256, 256, 256])
        self.assertEqual([count for _, count in blocks], [256, 256, 187])
        reconstructed = __import__('pandas').concat([block.iloc[:count] for block, count in blocks], ignore_index=True)
        __import__('pandas').testing.assert_frame_equal(reconstructed, raw.reset_index(drop=True))
        self.assertEqual(len(list(new_models.query_blocks(raw.iloc[:1], self.raw.iloc[:1]))[0][0]), 256)

    def test_time_limit_reserves_delivery_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(directory)
            start = budget.authorization['started_epoch']
            with patch('core.time.time', return_value=start + 3 * 3600):
                self.assertEqual(budget.remaining('screen'), 1.5 * 3600)
            with patch('core.time.time', return_value=start + 4.5 * 3600):
                self.assertEqual(budget.remaining('screen'), 0)
                self.assertEqual(budget.remaining('export'), 1.5 * 3600)
            with patch('core.time.time', return_value=start + 6 * 3600):
                self.assertEqual(budget.remaining('export'), 0)

    def test_engineering_ceiling_keeps_global_delivery_reserve_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=Budget(directory)
            start=budget.authorization['started_epoch']
            with patch('core.time.time',return_value=start+900):
                self.assertEqual(budget.remaining('audit'),2700)
                self.assertEqual(budget.remaining('pilot'),2700)

    def test_routes_follow_engineering_membership(self):
        routes = features.routes(['I8', 'KS8-CLEAN', 'I16'])
        self.assertEqual(routes['CORE_EQUAL'], {'I8': .5, 'KS8-CLEAN': .5})
        self.assertNotIn('I16', routes['CORE_EQUAL'])
        self.assertNotIn('CORE_EQUAL', features.routes(['KS8-CLEAN']))
        self.assertEqual(len(features.routes(list(features.SPECS))) * len(features.WEIGHTS), 20)
        self.assertEqual(CONFIRM[:3], [920201, 920202, 920203])

    def test_selection_rejects_tiny_or_one_split_losses(self):
        def row(route, weight, gain, relative, count=1):
            return dict(route=route, name=route, weight=weight, gain_42=gain,
                        gain_71=gain, mean_relative_gain=relative, components=count)
        self.assertEqual(workflow.selection([row('C8', .1, 1, .0019), row('D8', .1, -1, .005)]), [])
        self.assertEqual(workflow.selection([row('CORE_EQUAL', .5, 1, .002, 3), row('C8', .5, 1, .002)])[0]['route'], 'C8')

    def test_stale_corrupt_and_incomplete_cache_rejected(self):
        class Unlimited:
            def require(self, estimate=0):
                pass
        with tempfile.TemporaryDirectory() as directory:
            cache = Cache(directory, Unlimited())
            value, context = {'prediction': np.array([1., 2.])}, {'targets': 'original'}
            cache.get('x', context, lambda: (value, {}))
            cache.get('x', context, lambda: self.fail('Valid cache recomputed'))
            with self.assertRaises(AssertionError):
                cache.get('x', {'targets': 'changed'}, lambda: (value, {}))
            (cache.root / 'x.npz').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                cache.get('x', context, lambda: (value, {}))
            np.savez_compressed(cache.root / 'partial.npz', **value)
            with self.assertRaises(ValueError):
                cache.get('partial', context, lambda: (value, {}))

    def test_confirmation_requires_every_random_gain(self):
        metrics = {str(s): {'gain': 1., 'relative_gain': .003} for s in CONFIRM}
        self.assertTrue(workflow.qualifies(metrics, {'ci95_INR': [.1, 2.]}))
        metrics[str(CONFIRM[1])]['gain'] = -1
        self.assertFalse(workflow.qualifies(metrics, {'ci95_INR': [.1, 2.]}))

    def test_no_finalist_never_trains_or_exports_candidate(self):
        e = self.e
        with tempfile.TemporaryDirectory() as directory:
            original = e.out
            e.out = Path(directory)
            try:
                save(e.out / 'candidate_lock.json', {'sources': e.sources, 'inputs': e.inputs,
                     'model_contract_sha256': 'x', 'choices': []})
                with patch('workflow.sha', return_value='x'), patch('modeling.fit', side_effect=AssertionError('Unexpected fit')):
                    workflow.confirm(e)
                decision = json.loads((e.out / 'decision.json').read_text())
                self.assertFalse(decision['qualified'])
                self.assertEqual(decision['portal_uploads'], 0)
            finally:
                e.out = original


if __name__ == '__main__':
    unittest.main(verbosity=2)
