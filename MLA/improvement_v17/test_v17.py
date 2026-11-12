"""Leakage, grouped validation and decision guards for the V17 implementation."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import features
import new_models
import workflow
from core import ROOT, Engine, Budget


class V17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=Engine(out=ROOT/'preflight_results')
        cls.raw=cls.e.train.iloc[:240].copy()

    def test_training_and_query_labels_ids_cannot_change_features(self):
        other=self.raw.copy()
        other['Annual Turnover']=-1e20
        other['Registration Number']=-9
        a,b=new_models.Codec().fit(self.raw),new_models.Codec().fit(other)
        self.assertEqual(a.vocabulary,b.vocabulary)
        np.testing.assert_array_equal(a.transform(self.raw),b.transform(other))

    def test_unseen_query_categories_do_not_refit_training_maps(self):
        codec=new_models.Codec().fit(self.raw)
        original={c:dict(v) for c,v in codec.vocabulary.items()}
        query=self.raw.iloc[:3].copy()
        query['City']='NEVER_SEEN_IN_V17'
        query['Cuisine']=None
        output=codec.transform(query)
        self.assertTrue(np.isnan(output[:,codec.columns.index('City')]).all())
        self.assertEqual(codec.vocabulary,original)
        self.assertIn('Cuisine 1',codec.vocabulary)

    def test_city_epoch_selection_holds_out_whole_groups(self):
        a,b=new_models.inner_indices(self.raw,fold=3,scheme='city')
        groups=features.canonical_city(self.raw['City']).fillna('__V10_UNKNOWN_CITY__')
        self.assertFalse(set(groups.iloc[a]) & set(groups.iloc[b]))
        other=self.raw.copy();other['Annual Turnover']=0.
        c,d=new_models.inner_indices(other,fold=3,scheme='city')
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)

    def test_reject_small_gains_and_one_assignment_loss(self):
        def row(name,weight,gain,relative):
            return dict(name=name,route='V14-blend',weight=weight,effective_new_share=weight,
                        gain_42=gain,gain_71=gain,mean_relative_gain=relative)
        small=row('XIA32',.1,1.,.0004)
        bad=row('NOR15',.1,-1.,.004)
        a=row('XIA32',.25,1.,.002)
        b=row('NOR15',.5,1.,.002)
        self.assertEqual(workflow.selection([small,bad]),[])
        self.assertEqual(workflow.selection([b,a])[0]['name'],'XIA32')

    def test_setup_cap_and_delivery_reserve_use_same_clock(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            budget=Budget(directory)
            start=budget.authorization['started_epoch']
            with patch('core.time.time',return_value=start+3601):
                self.assertEqual(budget.remaining('pilot'),0.)
            with patch('core.time.time',return_value=start+5*3600):
                self.assertEqual(budget.remaining('screen'),0.)
                self.assertGreater(budget.remaining('export'),3500.)


if __name__=='__main__':
    unittest.main(verbosity=2)
