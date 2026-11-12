"""Independent provenance, target-isolation, inference and submission checks."""
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import ROOT, Cache, digest, sha, save, validate_csv


def reference_checks(e):
    for name, expected in e.manifest['reference_files'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in e.registry['generating_sources'].items():
        assert sha(ROOT / 'provenance' / name) == expected, name
    checked = []
    for scheme in [42, 71, 48211, 48212, 48213, 'strat:27421', 'city']:
        record = e.registry['schemes'][str(scheme)]
        assert record['inputs'] == e.inputs and record['row_ids_sha256'] == e.row_hash
        with np.load(ROOT / record['file'], allow_pickle=False) as data:
            v7, t9, v9 = (data[k].copy() for k in ['v7', 't9', 'v9'])
            if 'ids' in data:
                np.testing.assert_array_equal(data['ids'], e.train['Registration Number'])
        reconstructed = np.zeros(len(e.y))
        mac = np.zeros(len(e.y))
        fold_hash = digest([b.tolist() for _, b in e.splits(scheme)])
        if 'fold_sha256' in record:
            assert record['fold_sha256'] == fold_hash
        for fold, (a, b) in enumerate(e.splits(scheme)):
            path = ROOT / record['tabpfn_witness_paths'][fold]
            witness = json.loads(path.with_suffix('.json').read_text())
            context = witness['context']
            assert witness['sha256'] == sha(path)
            assert context['sources'] == e.registry['generating_sources']
            assert context['inputs'] == e.inputs
            assert context['train_indices'] == digest(a.tolist())
            assert context['valid_indices'] == digest(b.tolist())
            assert context['training_targets'] == hashlib.sha256(e.y[a].tobytes()).hexdigest()
            assert context['execution_contract'] == e.manifest['execution_contract']
            assert context['seed'] == 10242 and context['estimators'] == 16
            with np.load(path, allow_pickle=False) as data:
                reconstructed[b] = data['prediction']
            if record['v7_witness_kind'] == 'original_mac_folds':
                path = ROOT / record['v7_witness_paths'][fold]
                witness = json.loads(path.with_suffix('.json').read_text())
                context = witness['context']
                assert witness['sha256'] == sha(path) and context['inputs'] == e.inputs
                assert context['train_ids'] == digest(e.train.iloc[a]['Registration Number'].tolist())
                assert context['valid_ids'] == digest(e.train.iloc[b]['Registration Number'].tolist())
                assert context['training_targets'] == hashlib.sha256(e.y[a].tobytes()).hexdigest()
                assert context['extra'] == {'scheme': str(scheme), 'fold': fold}
                for name in ['baseline.py', 'legacy_models.py', 'legacy_reference.py', 'v5_contract.json']:
                    assert context['sources'][name] == sha(ROOT / name)
                with np.load(path, allow_pickle=False) as data:
                    mac[b] = data['prediction']
        np.testing.assert_array_equal(reconstructed, t9)
        np.testing.assert_array_equal(.5 * v7 + .5 * t9, v9)
        if record['v7_witness_kind'] == 'original_mac_folds':
            np.testing.assert_array_equal(mac, v7)
        else:
            assert sha(ROOT / record['v7_witness_file']) == record['v7_witness_sha256']
            with np.load(ROOT / record['v7_witness_file'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['prediction'] if str(scheme) == 'city' else data['v7'], v7)
        checked.append({'scheme': str(scheme), 'fold_sha256': fold_hash,
                        'baseline_sha256': record['sha256'], 'inference': record['control_behavior']})
    save(e.out / 'reference_verification.json', {'status': 'PASS', 'assignments': checked,
                                               'validated_tabpfn_fold_witnesses': 35})


def feature_checks(e):
    a, b = e.splits(42)[0]
    train, valid = e.train.iloc[a], e.train.iloc[b].iloc[:20].copy()
    altered = valid.copy()
    altered['Registration Number'] = -99
    altered['Annual Turnover'] = 1e30
    pd.testing.assert_frame_equal(features.frame(valid), features.frame(altered))
    _, vocabulary = features.native_frame(features.frame(train))
    edge = valid.copy()
    for c in ['City', 'Cuisine', 'Restaurant Theme']:
        edge[c] = 'V13_NEVER_SEEN'
    native, _ = features.native_frame(features.frame(edge), vocabulary)
    for c in ['City', 'Cuisine', 'Restaurant Theme']:
        assert native[c].isna().all()
    original = e.y.copy()
    targets = {}
    for name in features.SPECS:
        values, metadata = features.training_target(e.y[a], name)
        changed = e.y.copy()
        changed[b] = 1e30
        other, other_metadata = features.training_target(changed[a], name)
        np.testing.assert_array_equal(values, other)
        assert metadata == other_metadata and len(values) == len(a)
        targets[name] = metadata
    np.testing.assert_array_equal(e.y, original)
    with tempfile.TemporaryDirectory(dir=e.out) as tmp:
        cache = Cache(tmp, e.budget)
        context = {'training_targets': 'original', 'quantile': .99}
        value = {'prediction': np.array([1., 2.])}
        cache.get('good', context, lambda: (value, {}))
        cache.get('good', context, lambda: (_ for _ in ()).throw(AssertionError('cache miss')))
        for kind in ['stale', 'corrupt', 'incomplete']:
            if kind != 'incomplete':
                cache.get(kind, context, lambda: (value, {}))
            if kind == 'corrupt':
                (cache.root / (kind + '.npz')).write_bytes(b'corrupt')
            if kind == 'incomplete':
                np.savez_compressed(cache.root / (kind + '.npz'), **value)
            try:
                cache.get(kind, {**context, 'quantile': .995} if kind == 'stale' else context,
                          lambda: (value, {}))
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('Bad checkpoint accepted: ' + kind)
    result = {'status': 'PASS', 'features': 40, 'target_id_isolation': True,
              'unknown_categories': True, 'all_training_rows_retained': True,
              'validation_target_changes_do_not_change_training_treatment': True,
              'training_treatments_first_fold': targets, 'invalid_checkpoints_rejected': True}
    save(e.out / 'feature_verification.json', result)
    return result


def inference_checks(predict, raw):
    sample = raw.iloc[:8].copy()
    baseline = predict(sample)
    permutation = np.random.default_rng(91).permutation(len(sample))
    permuted = predict(sample.iloc[permutation])
    changed = sample.copy()
    changed['Annual Turnover'] = 1e30
    changed['Registration Number'] = -7
    isolated = predict(changed)
    split = np.concatenate([predict(sample.iloc[:3]), predict(sample.iloc[3:])])
    individual = predict(sample.iloc[:1])
    checks = {'permutation_difference_INR': float(np.max(np.abs(permuted - baseline[permutation]))),
              'label_id_difference_INR': float(np.max(np.abs(isolated - baseline))),
              'batch_difference_INR': float(np.max(np.abs(split - baseline))),
              'individual_difference_INR': float(abs(individual[0] - baseline[0]))}
    for name, value in checks.items():
        assert value < .01, f'{name}: INR {value}'
    edge = sample.iloc[:3].copy()
    edge['City'] = 'V13_NEVER_SEEN'
    edge['Cuisine'] = pd.NA
    edge['Opening Day of Restaurant'] = 'invalid'
    for c in features.SOCIAL + features.EVENTS + features.FACILITIES + ['Resturant Tier']:
        edge[c] = np.nan
    result = predict(edge)
    assert result.shape == (3,) and np.isfinite(result).all() and (result >= 0).all()
    return {**checks, 'missing_unseen_inputs': True}


def moment_checks():
    """Compare the actual pinned criterion's moment with independent integration."""
    import torch
    from scipy.integrate import quad
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    criterion = FullSupportBarDistribution(torch.tensor([1., 2., 4., 7.], dtype=torch.float64))
    logits = torch.tensor([-.4, .7, -.2], dtype=torch.float64)
    moment = float(criterion.mean_of_square(logits))
    probabilities = torch.softmax(logits, -1).numpy()
    cuts = np.r_[0., np.cumsum(probabilities)]
    def squared_quantile(q):
        return float(criterion.icdf(logits, float(q))) ** 2
    integrated = sum(quad(squared_quantile, float(a), float(b), epsabs=1e-7)[0]
                     for a, b in zip(cuts[:-1], cuts[1:]))
    assert abs(moment - integrated) < 1e-5, (moment, integrated)
    import modeling
    output = modeling.decode_full({'criterion': criterion, 'logits': logits})
    assert abs(float(output['prediction']) / 1e6 - integrated) < 1e-5
    assert float(output['prediction']) > float(output['naive_squared_mean'])
    return {'status': 'PASS', 'analytical_second_moment': moment,
            'independent_quantile_integral': integrated, 'difference': abs(moment - integrated)}


def csv_checks(e, prediction):
    frame = pd.DataFrame({'Registration Number': e.test['Registration Number'], 'Annual Turnover': prediction})
    validate_csv(frame, e.test)
    bads = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]], ignore_index=True),
            frame.iloc[::-1].reset_index(drop=True), frame.iloc[:, ::-1],
            frame.rename(columns={'Annual Turnover': 'Turnover'}), frame.assign(extra=0)]
    for column, value in [('Registration Number', 'wrong'),
                          ('Registration Number', frame['Registration Number'].iloc[1]),
                          ('Annual Turnover', np.nan), ('Annual Turnover', np.inf),
                          ('Annual Turnover', -np.inf), ('Annual Turnover', -1.)]:
        bad = frame.copy()
        if isinstance(value, str):
            bad[column] = bad[column].astype(object)
        bad.loc[0, column] = value
        bads.append(bad)
    for bad in bads:
        try:
            validate_csv(bad, e.test)
        except (AssertionError, ValueError, TypeError):
            continue
        raise AssertionError('Malformed submission accepted')
    return {'rows': 500, 'malformed_cases_rejected': len(bads)}
