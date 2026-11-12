"""Leakage, split, lifecycle and decision checks for the bounded V16 experiment."""
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import features
import modeling
import tabm_model
import workflow
from core import ROOT, Engine, Budget


class V16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.raw = cls.engine.train.iloc[:240].copy()

    def test_training_and_query_target_id_isolation(self):
        altered = self.raw.copy()
        altered['Annual Turnover'] = -1e20
        altered['Registration Number'] = -8
        original = tabm_model.Preprocessor().fit(self.raw)
        changed = tabm_model.Preprocessor().fit(altered)
        np.testing.assert_array_equal(original.medians, changed.medians)
        np.testing.assert_array_equal(original.quantiles.quantiles_, changed.quantiles.quantiles_)
        self.assertEqual(original.vocabularies, changed.vocabularies)
        for key, expected in original.transform(self.raw).items():
            np.testing.assert_array_equal(expected, changed.transform(altered)[key])

    def test_validation_inputs_do_not_refit_preprocessing(self):
        processor = tabm_model.Preprocessor().fit(self.raw)
        before = processor.quantiles.quantiles_.copy()
        outlier = self.engine.train.iloc[-20:].copy()
        outlier['Facebook Popularity Quotient'] = 1e25
        transformed = processor.transform(outlier)
        np.testing.assert_array_equal(before, processor.quantiles.quantiles_)
        self.assertTrue(np.isfinite(transformed['numeric']).all())

    def test_derived_categories_and_missing_are_supported(self):
        processor = tabm_model.Preprocessor().fit(self.raw)
        self.assertIn('Cuisine 1', processor.categorical)
        self.assertIn('Cuisine 2', processor.categorical)
        unseen = self.raw.iloc[:3].copy()
        unseen['City'] = 'NEVER_SEEN_V16'
        unseen['Cuisine'] = pd.NA
        result = processor.transform(unseen)
        self.assertTrue((result['categorical'][:, processor.categorical.index('City')] == 1).all())
        self.assertTrue((result['categorical'][:, processor.categorical.index('Cuisine')] == 0).all())
        self.assertTrue(np.isfinite(result['numeric']).all())

    def test_city_epoch_selection_has_disjoint_groups(self):
        training, validation = tabm_model.inner_indices(self.raw, fold=3, scheme='city')
        groups = features.canonical_city(self.raw['City']).fillna('__V10_UNKNOWN_CITY__').to_numpy()
        self.assertFalse(set(groups[training]) & set(groups[validation]))
        altered = self.raw.copy()
        altered['Annual Turnover'] = 3.
        other = tabm_model.inner_indices(altered, fold=3, scheme='city')
        np.testing.assert_array_equal(training, other[0])
        np.testing.assert_array_equal(validation, other[1])

    def test_blend_endpoints_and_v14_control(self):
        v9, p35, new = np.array([10., 20.]), np.array([30., 40.]), np.array([80., 90.])
        v14 = .25 * v9 + .75 * p35
        np.testing.assert_array_equal(modeling.combine(v9, new, 1., p35), new)
        np.testing.assert_allclose(modeling.combine(v9, new, .1, p35), .9 * v14 + .1 * new)
        with self.assertRaises(AssertionError):
            modeling.combine(v9, new, .2, p35)

    def test_development_gate_and_tie_break(self):
        def candidate(name, weight, gain, relative):
            return {'name': name, 'route': 'V14-blend', 'weight': weight,
                    'effective_new_share': weight, 'gain_42': gain, 'gain_71': gain,
                    'mean_relative_gain': relative}
        small = candidate('small', .1, 10., .0003)
        bad = candidate('one-split-loss', .1, -1., .005)
        first = candidate('TM32-PLE', .25, 1., .002)
        second = candidate('MITRA2-FT', .5, 1., .002)
        self.assertEqual(workflow.selection([small, bad]), [])
        self.assertEqual(workflow.selection([second, first])[0]['name'], 'TM32-PLE')

    def test_budget_preserves_delivery_and_combined_setup_cap(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            budget = Budget(directory)
            start = budget.authorization['started_epoch']
            with patch('core.time.time', return_value=start + 3601):
                self.assertEqual(budget.remaining('pilot'), 0.)
            with patch('core.time.time', return_value=start + 5 * 3600):
                self.assertEqual(budget.remaining('screen'), 0.)
                self.assertGreater(budget.remaining('export'), 3500.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
