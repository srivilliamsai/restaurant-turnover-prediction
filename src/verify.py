"""Inference invariance and missing-value checks for the final blend."""

import numpy as np
import pandas as pd

import features


def inference_checks(predict, raw):
    sample = raw.copy()
    baseline = predict(sample)
    assert baseline.shape == (len(sample),) and np.isfinite(baseline).all() and (baseline >= 0).all()
    permutation = np.random.default_rng(91).permutation(len(sample))
    permuted = np.empty(len(sample))
    permuted[permutation] = predict(sample.iloc[permutation])
    changed = sample.copy()
    changed['Annual Turnover'] = 1e30
    changed['Registration Number'] = -7
    isolated = predict(changed)
    mid = len(sample) // 2
    split = np.concatenate([predict(sample.iloc[:mid]), predict(sample.iloc[mid:])])
    individual = np.array([predict(sample.iloc[i:i+1])[0] for i in range(min(12, len(sample)))])
    checks = {'permutation_difference_INR': float(np.max(np.abs(permuted - baseline))),
              'label_id_difference_INR': float(np.max(np.abs(isolated - baseline))),
              'batch_difference_INR': float(np.max(np.abs(split - baseline))),
              'individual_difference_INR': float(np.max(np.abs(individual - baseline[:len(individual)])))}
    assert checks['label_id_difference_INR'] < .01, checks
    for name in ['permutation_difference_INR', 'batch_difference_INR', 'individual_difference_INR']:
        assert checks[name] <= 10., f'{name}: INR {checks[name]}'
    edge = sample.iloc[:3].copy()
    edge['City'] = 'V14_NEVER_SEEN'
    edge['Cuisine'] = pd.NA
    edge['Restaurant Theme'] = 'V14_NEVER_SEEN'
    edge['Opening Day of Restaurant'] = 'invalid'
    for c in features.SOCIAL + features.EVENTS + features.FACILITIES + ['Resturant Tier']:
        edge[c] = np.nan
    result = predict(edge)
    assert result.shape == (3,) and np.isfinite(result).all() and (result >= 0).all()
    return {**checks, 'rows_checked': len(sample), 'individual_probes': len(individual),
            'missing_unseen_inputs': True, 'behavior_tolerance_INR': 10.}
