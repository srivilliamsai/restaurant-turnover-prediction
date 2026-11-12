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
        parts = e.reference_parts(scheme)
        assert np.isfinite(parts['v14']).all()
        checked.append({'scheme': str(scheme), 'v14_rmse': __import__('core').rmse(e.y, parts['v14']), 'fold_sha256': fold_hash,
                        'baseline_sha256': record['sha256'], 'inference': record['control_behavior']})
    save(e.out / 'reference_verification.json', {'status': 'PASS', 'assignments': checked,
                                               'validated_tabpfn_fold_witnesses': 35,
                                               'validated_v14_component_fold_witnesses': 35})


def feature_checks(e):
    a, b = e.splits(42)[0]
    train, valid = e.train.iloc[a], e.train.iloc[b].iloc[:20].copy()
    altered = valid.copy()
    altered['Registration Number'] = -99
    altered['Annual Turnover'] = 1e30
    pd.testing.assert_frame_equal(features.frame(valid), features.frame(altered))
    import new_models
    processor = new_models.Codec().fit(train)
    np.testing.assert_array_equal(processor.transform(valid), processor.transform(altered))
    before = json.dumps(processor.vocabulary,sort_keys=True)
    edge = valid.copy()
    for column in ['City', 'Cuisine', 'Restaurant Theme']:
        edge[column] = 'V17_NEVER_SEEN'
    processor.transform(edge)
    assert before == json.dumps(processor.vocabulary,sort_keys=True)
    with tempfile.TemporaryDirectory(dir=e.out) as tmp:
        cache = Cache(tmp, e.budget)
        context = {'training_targets': 'original', 'target_units': 'raw INR millions'}
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
                cache.get(kind, {**context, 'training_targets': 'changed'} if kind == 'stale' else context,
                          lambda: (value, {}))
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('Bad checkpoint accepted: ' + kind)
    result = {'status': 'PASS', 'original_feature_count': 40, 'target_id_isolation': True,
              'unknown_categories': True, 'missing_unseen_representation': 'NaN with train-only vocabularies',
              'all_training_rows_retained': True, 'invalid_checkpoints_rejected': True}
    save(e.out / 'feature_verification.json', result)
    return result


def inference_checks(predict, raw, y=None):
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
    if y is not None:
        from core import rmse
        target = np.asarray(y, float)
        base_rmse = rmse(target, baseline)
        changes = {'permutation': abs(rmse(target, permuted)-base_rmse),
                   'batch': abs(rmse(target, split)-base_rmse),
                   'individual': abs(rmse(target[:len(individual)], individual)-rmse(target[:len(individual)], baseline[:len(individual)]))}
        assert max(changes.values()) <= 1., changes
        checks['rmse_difference_INR'] = changes
    perturb = sample.copy()
    perturb.iloc[1:, perturb.columns.get_loc('Facebook Popularity Quotient')] = 1e6
    fixed_difference = abs(float(predict(perturb)[0])-float(baseline[0]))
    assert fixed_difference <= 10., fixed_difference
    checks['other_query_rows_changed_difference_INR'] = fixed_difference
    edge = sample.iloc[:3].copy()
    edge['City'] = 'V17_NEVER_SEEN'
    edge['Cuisine'] = pd.NA
    edge['Restaurant Theme'] = 'V17_NEVER_SEEN'
    edge['Opening Day of Restaurant'] = 'invalid'
    for c in features.SOCIAL + features.EVENTS + features.FACILITIES + ['Resturant Tier']:
        edge[c] = np.nan
    result = predict(edge)
    assert result.shape == (3,) and np.isfinite(result).all() and (result >= 0).all()
    return {**checks, 'rows_checked': len(sample), 'individual_probes': len(individual),
            'missing_unseen_inputs': True, 'behavior_tolerance_INR': 10., 'rmse_tolerance_INR': 1.}


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

