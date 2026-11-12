"""Behavioral verification before accuracy screening; no acceptance scores tuned here."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
import modeling as m
from engine import Engine, Cache, ROOT, save


def run(data_dir=None):
    e = Engine(data_dir)
    checks = {}
    with tempfile.TemporaryDirectory() as temporary:
        cache = Cache(temporary)
        context = {'training': [1, 2], 'config': 'fixed'}
        one = cache.get('valid', context, lambda: np.array([1., 2.]))
        two = cache.get('valid', context, lambda: (_ for _ in ()).throw(AssertionError('Recomputed valid checkpoint')))
        assert np.array_equal(one, two) and cache.reads == 1
        try:
            cache.get('valid', {'changed': True}, lambda: one)
        except ValueError:
            checks['stale_checkpoint_rejected'] = True
        else:
            raise AssertionError('Stale cache accepted')
        (Path(temporary) / 'cache/valid.npz').write_bytes(b'corruption')
        try:
            cache.get('valid', context, lambda: one)
        except ValueError:
            checks['corrupt_checkpoint_rejected'] = True
        else:
            raise AssertionError('Corrupt cache accepted')
        np.savez_compressed(Path(temporary) / 'cache/interrupted.npz', prediction=[9., 9.])
        p = cache.get('interrupted', context, lambda: one)
        assert np.array_equal(p, one)
        checks['incomplete_checkpoint_recomputed'] = True
        checks['valid_checkpoint_reused'] = True
    X = pd.DataFrame({'group': pd.Series(['a', 'b', 'c'] * 20, dtype='string')})
    y = np.arange(60, dtype=float)
    args = dict(target_type='continuous', smooth=50, cv=KFold(5, shuffle=True, random_state=831))
    a = TargetEncoder(**args)
    b = TargetEncoder(**args)
    za = a.fit_transform(X, y)
    changed = y.copy()
    changed[0] += 1e6
    zb = b.fit_transform(X, changed)
    assert np.array_equal(za[0], zb[0])
    assert np.isclose(a.transform(pd.DataFrame({'group': ['never_seen']}))[0, 0], y.mean())
    checks['own_label_excluded_from_target_encoding'] = True
    checks['unseen_target_encoding_uses_training_mean'] = True
    levels = pd.DataFrame({'city': np.repeat([f'city_{i:03}' for i in range(300)], 10)})
    native, state = m.native_frame(levels)
    assert len(native.city.cat.categories) == 254
    unseen, _ = m.native_frame(pd.DataFrame({'city': ['never_seen']}), state)
    assert unseen.city.isna().all()
    checks['native_cardinality_bounded_and_unseen_missing'] = True
    train, valid = e.train.iloc[:500], e.test.iloc[:40].copy()
    labels = e.y[:500]
    names = ['native_raw_7', 'encoded_sqrt_200', 'ridge_sqrt_100_structured', 'ridge_raw_10_base']
    for name in names:
        fitted = m.fit_model(train, labels, m.SPECS[name])
        p = m.predict_model(fitted, valid)
        perm = np.random.default_rng(91).permutation(len(valid))
        assert np.max(np.abs(m.predict_model(fitted, valid.iloc[perm]) - p[perm])) < .01
        changed = valid.copy()
        changed['Annual Turnover'] = 1e30
        changed['Registration Number'] = -77
        assert np.max(np.abs(m.predict_model(fitted, changed) - p)) < .01
        edge = valid.iloc[:3].copy()
        edge['City'] = 'new_city'
        edge['Cuisine'] = pd.NA
        edge['Opening Day of Restaurant'] = 'invalid'
        for c in ['Facebook Popularity Quotient', 'Instagram Popularity Quotient', 'Resturant Tier']:
            edge[c] = np.nan
        prediction = m.predict_model(fitted, edge)
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
    checks['all_model_families_missing_unseen_permutation_target_id_invariance'] = True
    # Exported frequency maps cannot learn support from prediction rows.
    maps = m.structured_frame(train)[1]
    absent = valid.iloc[:2].copy()
    absent['City'] = 'new_city'
    absent['Cuisine'] = 'new_cuisine'
    frame, _ = m.structured_frame(absent, maps)
    assert (frame[['V8 Log City count', 'V8 Log Cuisine count']] == 0).all().all()
    checks['frequency_maps_training_only_unseen_zero'] = True
    checks['status'] = 'PASS'
    save(e.out / 'core_verification.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    run()
