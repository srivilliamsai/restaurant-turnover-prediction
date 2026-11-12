"""Meaningful leakage, cache, feature-semantics and gate checks; no model fitting."""
import json
import numpy as np
import pandas as pd
import features,modeling,workflow,verify
from core import ROOT,Engine,save

def main():
    e=Engine()
    checks=[]
    with e.budget.charge('audit'):
        verify.reference_checks(e)
        verify.feature_checks(e)
    checks.append('All 35 V9 and 35 V14 control-fold witnesses, original targets and source/runtime identities verified.')
    raw=e.train.iloc[:30].copy()
    f=features.frame(raw,'F40');n=features.frame(raw,'N32')
    assert len(f.columns)==40 and len(n.columns)==32
    assert pd.api.types.is_datetime64_any_dtype(n['Opening Day of Restaurant'])
    expected=pd.to_datetime(raw['Opening Day of Restaurant'],format='%d-%m-%Y',errors='coerce').reset_index(drop=True)
    pd.testing.assert_series_equal(n['Opening Day of Restaurant'],expected)
    for c in features.CATEGORICAL:pd.testing.assert_series_equal(f[c],n[c])
    assert not {'Annual Turnover','Registration Number'} & set(n.columns)
    for c in ['Service','Comfortablility','Facebook Popularity Quotient']:
        np.testing.assert_allclose(n[c],raw[c].reset_index(drop=True),equal_nan=True)
    checks.append('N32 retains native dates and original numeric missingness with identical V14 categorical normalization.')
    a=np.array([20.,40.]);old=np.array([50.,100.]);new=np.array([70.,140.])
    np.testing.assert_array_equal(modeling.combine(a,new,.5,old),[50.,100.])
    np.testing.assert_array_equal(modeling.combine(a,new,1.,old),.25*a+.75*new)
    np.testing.assert_array_equal(modeling.combine(a,old,.25,old),.25*a+.75*old)
    checks.append('V14 control and V15 replacement algebra preserve the quarter-share V9 anchor.')
    def row(name,w,g1=1,g2=1,rel=.003):return dict(name=name,route='P35-replacement',weight=w,effective_new_share=.75*w,gain_42=g1,gain_71=g2,mean_relative_gain=rel)
    assert not workflow.selection([row('F40-T080',.25,rel=.00199)])
    assert not workflow.selection([row('F40-T080',.25,g2=-1)])
    assert workflow.selection([row('N32-T120',.5),row('F40-T080',.25)])[0]['name']=='F40-T080'
    assert workflow.selection([row('N32-T120',.25),row('F40-T080',.25)])[0]['name']=='F40-T080'
    good={str(s):{'gain':1.,'relative_gain':.003} for s in [48211,48212,48213,'strat:27421','city']}
    assert workflow.qualifies(good,{'ci95_INR':[1.,2.]})
    assert not workflow.qualifies(good,{'ci95_INR':[0.,2.]})
    for key in ['city','strat:27421']:
        bad={k:dict(v) for k,v in good.items()};bad[key]['relative_gain']=-.01001
        assert not workflow.qualifies(bad,{'ci95_INR':[1.,2.]})
    checks.append('Both-split development, minimum gain, deterministic ranking, bootstrap and stress gates enforced.')
    def predictor(x):return pd.to_numeric(x['Hygiene Rating'],errors='coerce').fillna(0).to_numpy(float)*1e6+20e6
    verify.inference_checks(predictor,raw,e.y[:30])
    try:verify.inference_checks(lambda x:predictor(x)+np.arange(len(x))*100.,raw,e.y[:30])
    except AssertionError:pass
    else:raise AssertionError('Order dependence accepted')
    verify.csv_checks(e,np.full(500,2e7))
    checks.append('Behavior tests reject substantive order dependence; twelve malformed CSV cases rejected.')
    save(ROOT/'local_tests.json',{'status':'PASS','checks':checks,'new_models_fitted':0})
    print(json.dumps({'status':'PASS','checks':len(checks)}))
if __name__=='__main__':main()
