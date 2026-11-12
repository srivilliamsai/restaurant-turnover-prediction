"""Provenance, pair-removal isolation, selection and inference contract checks."""
import json
import time
import numpy as np
import pandas as pd
import features
import modeling
import verify
import workflow
from core import ROOT, Engine, save


def main():
    engine = Engine()
    checks = []
    with engine.budget.charge('audit'):
        verify.reference_checks(engine)
        verify.feature_checks(engine)
    checks.append('All 35 V9 and 35 V14 witnesses, seven assignments and scored artifacts verified.')
    training = engine.train.iloc[:30].copy()
    builder = features.FeatureBuilder('P35-SC38').fit(training)
    expected = features.frame(training).drop(columns=features.DROP)
    pd.testing.assert_frame_equal(builder.transform(training), expected)
    assert expected.shape == (30, 38)
    changed = training.copy()
    changed['Service'] = np.arange(30)*1e20
    changed['Comfortablility'] = np.nan
    pd.testing.assert_frame_equal(builder.transform(changed), expected)
    assert features.FeatureBuilder('P35-SC38').fit(changed).state() == builder.state()
    checks.append('Exact 38-column projection and invariance to arbitrary removed-field mutations verified.')
    original = np.array([20., 40.])
    old_component = np.array([50., 100.])
    new_component = np.array([70., 140.])
    np.testing.assert_array_equal(modeling.combine(original, new_component, .5, old_component), [50., 100.])
    np.testing.assert_array_equal(modeling.combine(original, new_component, 1., old_component), .25*original+.75*new_component)
    assert len(features.SPECS)*len(features.WEIGHTS) == 3
    def comparison(weight, first=1., second=1., relative=.003):
        return dict(name='P35-SC38', route='P35-replacement', weight=weight,
                    effective_new_share=.75*weight, gain_42=first, gain_71=second, mean_relative_gain=relative)
    assert not workflow.selection([comparison(.25,relative=.00199)])
    assert not workflow.selection([comparison(.25,second=-1.)])
    assert workflow.selection([comparison(.5),comparison(.25)])[0]['weight'] == .25
    metrics = {str(s): {'gain':1.,'relative_gain':.003} for s in [48211,48212,48213,'strat:27421','city']}
    assert workflow.qualifies(metrics, {'ci95_INR':[1.,2.]})
    assert not workflow.qualifies(metrics, {'ci95_INR':[0.,2.]})
    for s in ['city','strat:27421','48211']:
        failed={key:dict(value) for key,value in metrics.items()}
        failed[s].update(gain=-1.,relative_gain=-.01001)
        assert not workflow.qualifies(failed, {'ci95_INR':[1.,2.]})
    checks.append('Three comparison limit, exact blend algebra, qualification floor and confirmation gates enforced.')
    def predictor(raw):
        return pd.to_numeric(raw['Hygiene Rating'],errors='coerce').fillna(0).to_numpy(float)*1e6+20e6
    verify.inference_checks(predictor, training, engine.y[:30])
    try:
        verify.inference_checks(lambda raw: predictor(raw)+np.arange(len(raw))*100., training,engine.y[:30])
    except AssertionError: pass
    else: raise AssertionError('Order dependence accepted')
    verify.csv_checks(engine,np.full(500,2e7))
    checks.append('Substantive order dependence and malformed CSV outputs rejected.')
    assert engine.budget.authorization['maximum_hours']==4
    assert engine.budget.remaining('pilot') <= engine.budget.authorization['engineering_stop_epoch']-time.time()+1
    assert engine.budget.remaining('screen') <= 1800
    checks.append('Four-hour persisted clock, 45-minute engineering cutoff and delivery reserve enforced.')
    save(ROOT/'local_tests.json',{'status':'PASS','checks':checks,'new_models_fitted':0})
    print(json.dumps({'status':'PASS','checks':len(checks)}))


if __name__ == '__main__': main()
