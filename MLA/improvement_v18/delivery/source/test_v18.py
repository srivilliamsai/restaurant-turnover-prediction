"""Meaningful leakage, budget, route and provenance checks before GPU allocation."""
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
import features,new_models,workflow
from core import ROOT,Engine,Budget,Cache,CONFIRM
class V18Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.e=Engine(out=ROOT/'preflight_results');cls.raw=cls.e.train.iloc[:240].copy()
 def test_target_id_exclusion_and_frozen_vocab(self):
  raw=self.raw;other=raw.copy();other['Annual Turnover']=-1e20;other['Registration Number']=-9
  a,b=new_models.Codec().fit(raw),new_models.Codec().fit(other)
  self.assertEqual(a.vocabulary,b.vocabulary);np.testing.assert_array_equal(a.transform(raw),b.transform(other))
  query=raw.iloc[:2].copy();query['City']='UNSEEN_V18';old=json.dumps(a.vocabulary,sort_keys=True)
  out=a.transform(query);self.assertTrue(np.isnan(out[:,a.columns.index('City')]).all());self.assertEqual(old,json.dumps(a.vocabulary,sort_keys=True));self.assertEqual(len(a.vocabulary),8)
 def test_city_holdout_is_disjoint(self):
  a,b=new_models.inner_indices(self.raw,3,'city');cities=features.canonical_city(self.raw['City']).fillna('missing')
  self.assertFalse(set(cities.iloc[a])&set(cities.iloc[b]))
 def test_shared_pool_and_delivery_reserve(self):
  with tempfile.TemporaryDirectory() as folder:
   budget=Budget(folder);start=budget.authorization['started_epoch']
   with patch('core.time.time',return_value=start+5*3600):self.assertGreater(budget.remaining('pilot'),9.9*3600)
   with patch('core.time.time',return_value=start+15*3600):self.assertEqual(budget.remaining('screen'),0);self.assertEqual(budget.remaining('export'),3*3600)
   with patch('core.time.time',return_value=start+18*3600):self.assertEqual(budget.remaining('export'),0)
 def test_admission_and_route_selection(self):
  def row(route,w,g,r):return dict(route=route,name=route,weight=w,effective_new_share=w,gain_42=g,gain_71=g,mean_relative_gain=r)
  self.assertEqual(workflow.selection([row('FT512',.1,1,.0019),row('EXA8',.1,-1,.005)]),[])
  self.assertEqual(workflow.selection([row('FT512_EXA8',.5,1,.002),row('FT512',.5,1,.002)])[0]['route'],'FT512')
  self.assertEqual(len(features.ROUTES)*len(features.WEIGHTS),25)
  self.assertEqual(CONFIRM[:3],[918601,918602,918603])
 def test_cache_requires_matching_targets_and_hash(self):
  class Unlimited:
   def require(self,estimate=0):pass
  with tempfile.TemporaryDirectory() as folder:
   c=Cache(folder,Unlimited());v={'prediction':np.array([1.,2.])};ctx={'targets':'original'}
   c.get('x',ctx,lambda:(v,{}));c.get('x',ctx,lambda:(_ for _ in ()).throw(AssertionError('miss')))
   with self.assertRaises(AssertionError):c.get('x',{'targets':'changed'},lambda:(v,{}))
   (c.root/'x.npz').write_bytes(b'corrupt')
   with self.assertRaises(AssertionError):c.get('x',ctx,lambda:(v,{}))
 def test_fine_tuning_recipe_comparison_is_controlled(self):
  a,b=dict(features.SPECS['FT512']),dict(features.SPECS['FT1024']);a.pop('context_plus_query_rows');b.pop('context_plus_query_rows')
  self.assertEqual(a,b);self.assertFalse(a['use_lr_scheduler']);self.assertEqual(a['learning_rate'],3e-6)
 def test_failed_confirmation_not_qualified(self):
  m={str(s):{'gain':1.,'relative_gain':.003} for s in CONFIRM}
  self.assertTrue(workflow.qualifies(m,{'ci95_INR':[.1,2.]}));m[str(CONFIRM[1])]['gain']=-1
  self.assertFalse(workflow.qualifies(m,{'ci95_INR':[.1,2.]}))
 def test_no_finalist_path_does_not_fit(self):
  e=self.e
  with tempfile.TemporaryDirectory() as folder:
   from pathlib import Path
   old=e.out;e.out=Path(folder)
   try:
    from core import save
    save(e.out/'candidate_lock.json',{'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':'x','choices':[]})
    with patch('workflow.sha',return_value='x'),patch('modeling.fit',side_effect=AssertionError('Unnecessary training')):workflow.confirm(e)
    d=json.loads((e.out/'decision.json').read_text());self.assertFalse(d['qualified']);self.assertEqual(d['portal_uploads'],0)
   finally:e.out=old
if __name__=='__main__':unittest.main(verbosity=2)
