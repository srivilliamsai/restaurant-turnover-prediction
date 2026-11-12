"""Leakage, group-context semantics, validation gates and malformed CSV checks."""
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
    checks.append('All 35 V9 and 35 V14 fold witnesses and original inputs verified.')
    training = engine.train.iloc[:30].copy()
    training['City'] = ['Mumbai'] * 10 + ['Pune'] * 20
    builder = features.FeatureBuilder('C+S+G').fit(training)
    transformed = builder.transform(training)
    expected_social = transformed['V10 Social mean']
    global_mean = float(expected_social.mean())
    observed = expected_social.iloc[:10].dropna()
    expected_group_mean = (float(observed.sum()) + 20 * global_mean) / (len(observed) + 20)
    np.testing.assert_allclose(builder.maps['city']['means']['social']['mumbai'], expected_group_mean)
    assert builder.maps['city']['counts']['mumbai'] == 10
    query = training.iloc[:2].copy()
    query['City'] = 'V21_UNSEEN'
    output = builder.transform(query)
    assert output['V10 Log city count'].eq(0).all()
    np.testing.assert_allclose(output['V10 social relative city'], output['V10 Social mean'] - global_mean, equal_nan=True)
    checks.append('Group counts, strength-20 shrinkage and unseen-group fallback reconstructed independently.')
    for name, count in [('S', 56), ('C+S+G', 66)]:
        recipe = features.FeatureBuilder(name).fit(training)
        assert recipe.transform(query).shape == (2, count)
    original = np.array([20., 40.])
    old_component = np.array([50., 100.])
    new_component = np.array([70., 140.])
    np.testing.assert_array_equal(modeling.combine(original, new_component, .5, old_component), [50., 100.])
    np.testing.assert_array_equal(modeling.combine(original, new_component, 1., old_component), .25 * original + .75 * new_component)
    np.testing.assert_array_equal(modeling.combine(original, old_component, .25, old_component), .25 * original + .75 * old_component)
    assert len(features.SPECS) * len(features.WEIGHTS) == 6
    checks.append('Exact feature counts, six comparisons and fixed-quarter V9 replacement algebra verified.')
    def comparison(name, weight, first=1., second=1., relative=.003):
        return dict(name=name, route='P35-replacement', weight=weight, effective_new_share=.75 * weight,
                    gain_42=first, gain_71=second, mean_relative_gain=relative)
    assert not workflow.selection([comparison('S', .25, relative=.00199)])
    assert not workflow.selection([comparison('S', .25, second=-1.)])
    assert workflow.selection([comparison('S', .5), comparison('C+S+G', .25)])[0]['name'] == 'C+S+G'
    metrics = {str(scheme): {'gain': 1., 'relative_gain': .003}
               for scheme in [48211, 48212, 48213, 'strat:27421', 'city']}
    assert workflow.qualifies(metrics, {'ci95_INR': [1., 2.]})
    assert not workflow.qualifies(metrics, {'ci95_INR': [0., 2.]})
    for scheme in ['city', 'strat:27421', '48211']:
        failed = {key: dict(value) for key, value in metrics.items()}
        failed[scheme].update(gain=-1., relative_gain=-.01001)
        assert not workflow.qualifies(failed, {'ci95_INR': [1., 2.]})
    checks.append('Development floor, both-assignment gains, ranking, bootstrap and stress checks enforced.')
    def predictor(raw):
        return pd.to_numeric(raw['Hygiene Rating'], errors='coerce').fillna(0).to_numpy(float) * 1e6 + 20e6
    verify.inference_checks(predictor, training, engine.y[:30])
    try:
        verify.inference_checks(lambda raw: predictor(raw) + np.arange(len(raw)) * 100., training, engine.y[:30])
    except AssertionError:
        pass
    else:
        raise AssertionError('Order dependence accepted')
    verify.csv_checks(engine, np.full(500, 2e7))
    checks.append('Substantive query-order dependence and twelve malformed CSV cases rejected.')
    assert engine.budget.remaining('pilot') > 0
    assert engine.budget.remaining('screen') <= engine.budget.authorization['hard_stop_epoch'] - time.time() - 5399
    assert engine.budget.remaining('pilot') <= engine.budget.authorization['engineering_stop_epoch'] - time.time() + 1
    save(ROOT / 'local_tests.json', {'status': 'PASS', 'checks': checks, 'new_models_fitted': 0})
    print(json.dumps({'status': 'PASS', 'checks': len(checks)}))


if __name__ == '__main__':
    main()
