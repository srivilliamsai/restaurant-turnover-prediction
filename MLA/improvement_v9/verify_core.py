"""Meaningful cache, feature and fold-isolation checks for the V9 models."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import modeling as m
from engine import Engine, Cache, save


def run(data_dir=None, out_dir=None):
    e = Engine(data_dir, out_dir)
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        c = Cache(tmp); ctx = {'config': 'frozen', 'train': [1, 2]}
        p = c.get('ok', ctx, lambda: np.array([1., 2.]))
        assert np.array_equal(c.get('ok', ctx, lambda: (_ for _ in ()).throw(AssertionError())), p)
        try: c.get('ok', {'config': 'changed'}, lambda: p)
        except ValueError: pass
        else: raise AssertionError('Stale checkpoint accepted')
        (Path(tmp)/'cache/ok.npz').write_bytes(b'corrupt')
        try: c.get('ok', ctx, lambda: p)
        except ValueError: pass
        else: raise AssertionError('Corrupt checkpoint accepted')
        np.savez_compressed(Path(tmp)/'cache/partial.npz', prediction=[99.])
        assert np.array_equal(c.get('partial',ctx,lambda:p),p)
        checks['valid_stale_corrupt_interrupted_checkpoints'] = True
    raw=e.train.iloc[:500];valid=e.test.iloc[:20].copy();y=e.y[:500]
    for representation in ['compact','v7']:
        X=m.features(raw,representation)
        changed=raw.copy();changed['Annual Turnover']=1e30;changed['Registration Number']=-1
        pd.testing.assert_frame_equal(X,m.features(changed,representation))
        assert m.features(valid,representation).columns.equals(X.columns)
    assert m.features(raw,'compact').shape[1]==33 and m.features(raw).shape[1]==40
    X=pd.DataFrame({'category':pd.Series(['a','b',None],dtype='string'),'number':[1.,np.nan,3.]})
    frame,state=m.native_frame(X)
    unknown,_=m.native_frame(pd.DataFrame({'category':['new'],'number':[np.nan]}),state)
    assert unknown.isna().all().all() and set(state['category'])=={'a','b'}
    checks['fold_local_vocabularies_and_unseen_missing']=True
    checks['feature_schemas_and_target_id_exclusion']=True
    original_fit=m.ExplainableBoostingRegressor.fit
    seen=[]
    def spy(self,X,target,*args,**kwargs):
        assert len(X)==len(raw)
        assert 'Registration Number' not in X and 'Annual Turnover' not in X
        assert np.allclose(target,y/1e6) or np.allclose(target,np.sqrt(y/1e6))
        seen.append(len(X))
        return original_fit(self,X,target,*args,**kwargs)
    with patch.object(m.ExplainableBoostingRegressor,'fit',new=spy):
        for name,spec in m.SPECS.items():
            if spec['family']!='ebm':continue
            fitted=m.fit_model(raw,y,spec)
            p=m.predict_model(fitted,valid)
            changed=valid.copy();changed['Annual Turnover']=1e30;changed['Registration Number']=-7
            assert np.max(np.abs(m.predict_model(fitted,changed)-p))<.01
            perm=np.random.default_rng(91).permutation(len(valid))
            assert np.max(np.abs(m.predict_model(fitted,valid.iloc[perm])-p[perm]))<.01
            edge=valid.iloc[:3].copy();edge['City']='unseen';edge['Cuisine']=pd.NA
            edge['Opening Day of Restaurant']='invalid'
            for col in ['Facebook Popularity Quotient','Instagram Popularity Quotient','Resturant Tier']:edge[col]=np.nan
            q=m.predict_model(fitted,edge)
            assert np.isfinite(q).all() and (q>=0).all()
    assert len(seen)==4
    checks.update(ebm_all_settings_behavior=True,outer_validation_labels_excluded_from_model_fit=True,
                  all_original_fit_targets_retained=True,status='PASS',gpu_model_fit_tested=False)
    save(e.out/'core_verification.json',checks)
    print(json.dumps(checks,indent=2),flush=True)
    return checks


if __name__=='__main__':run()
