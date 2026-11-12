"""Decision, inference-contract and provenance checks without model training."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import features
import modeling
import workflow
from core import ROOT, Engine, save


def main():
    checks = []
    a = np.array([20.,40.]); v7 = np.array([10.,20.]); t9 = np.array([30.,60.]); m = np.array([50.,100.])
    np.testing.assert_array_equal(modeling.combine(a,m,.5,'A',v7,t9),[35.,70.])
    np.testing.assert_array_equal(modeling.combine(a,m,1.,'B',v7,t9),[30.,60.])
    checks.append('Both routes preserve the declared original component weights.')
    rows = [dict(name='I8',route='A',weight=.25,effective_new_share=.25,gain_42=10,gain_71=2,mean_relative_gain=.0019),
            dict(name='I8',route='B',weight=.5,effective_new_share=.25,gain_42=10,gain_71=-1,mean_relative_gain=.004),
            dict(name='P35_8',route='B',weight=.25,effective_new_share=.125,gain_42=10,gain_71=2,mean_relative_gain=.003)]
    assert workflow.selection(rows) == rows[2:]
    tied = [{**rows[2], 'effective_new_share':.5, 'route':'A'},rows[2]]
    assert workflow.selection(tied)[0]['route'] == 'B'
    checks.append('Minimum gain, both-split improvement and effective-share ranking enforced.')
    good = {str(s):{'gain':1.,'relative_gain':.003} for s in [48211,48212,48213,'strat:27421','city']}
    assert workflow.qualifies(good,{'ci95_INR':[1.,2.]})
    assert not workflow.qualifies(good,{'ci95_INR':[-1.,2.]})
    bad = {k:dict(v) for k,v in good.items()};bad['city']['relative_gain']=-.011
    assert not workflow.qualifies(bad,{'ci95_INR':[1.,2.]})
    checks.append('Frozen confirmation requires a positive interval and stress limits.')
    e = Engine()
    import verify
    with e.budget.charge('audit'):
        verify.reference_checks(e)
        verify.feature_checks(e)
    checks.append('All 35 baseline fold witnesses, training-only features and invalid-cache rejections passed.')
    raw=e.train.iloc[:20].copy(); target=e.y[:20]
    values={str(x):float(i+20)*1e6 for i,x in enumerate(raw['Registration Number'])}
    # Feature-based deterministic stand-in; changing IDs must not change it.
    def predictor(x):
        return pd.to_numeric(x['Hygiene Rating'],errors='coerce').fillna(0).to_numpy(float)*1e6+20e6
    result=verify.inference_checks(predictor,raw,target)
    assert result['rows_checked']==20 and result['individual_probes']==12
    def dependent(x):
        return predictor(x)+np.arange(len(x))*100.
    try:verify.inference_checks(dependent,raw,target)
    except AssertionError:pass
    else:raise AssertionError('Substantive order dependence was accepted')
    checks.append('Full-batch behavior probes accept invariant outputs and reject substantive row-order dependence.')
    save(ROOT/'local_tests.json',{'status':'PASS','checks':checks,'new_models_fitted':0})
    print(json.dumps({'status':'PASS','checks':len(checks)},indent=2))


if __name__=='__main__':
    main()
