"""Verify historical folds and reconstruct V14–V18 metrics without fitting models."""
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V18 = PROJECT / 'improvement_v18'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def main():
    with (PROJECT / 'Train_dataset.csv').open(newline='') as stream:
        train = list(csv.DictReader(stream))
    with (PROJECT / 'Test_dataset.csv').open(newline='') as stream:
        test = list(csv.DictReader(stream))
    y = np.array([float(row['Annual Turnover']) for row in train])
    ids = [int(row['Registration Number']) for row in train]
    registry = read(V18 / 'baseline_registry.json')
    inputs = {name: sha(PROJECT / name) for name in registry['inputs']}
    assert inputs == registry['inputs']
    assert digest(ids) == registry['row_ids_sha256']
    assert np.isfinite(y).all() and len(set(ids)) == len(ids)
    test_ids = [int(row['Registration Number']) for row in test]
    assert len(set(test_ids)) == len(test_ids) and not set(ids) & set(test_ids)
    portal = read(V18 / 'reference/v14_portal_trial.json')
    references = {}
    (ROOT / 'reference').mkdir(exist_ok=True)
    for extension, key in [('.csv', 'csv_sha256'), ('.ipynb', 'notebook_sha256')]:
        source = V18 / 'reference' / ('scored_v14' + extension)
        assert sha(source) == portal[key]
        destination = ROOT / 'reference' / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
        assert sha(destination) == portal[key]
        references[destination.name] = sha(destination)
    with (ROOT / 'reference/scored_v14.csv').open(newline='') as stream:
        submission = list(csv.DictReader(stream))
    assert [int(row['Registration Number']) for row in submission] == test_ids
    old_sources = read(V18 / 'provenance/v14_source_frozen.json')['sources']
    verified = []

    def load_fold(root, directory, name, seed, fold, a, b, sources):
        path = root / directory / f'{name}_{seed}_fold{fold}.npz'
        metadata = read(path.with_suffix('.json'))
        context = metadata['context']
        assert sha(path) == metadata['sha256']
        assert context['inputs'] == inputs and context['sources'] == sources
        assert context['name'] == name
        assert context['train_indices'] == digest(a.tolist())
        assert context['valid_indices'] == digest(b.tolist())
        assert context['training_targets'] == hashlib.sha256(y[a].tobytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            prediction = data['prediction'].copy()
        assert prediction.shape == b.shape and np.isfinite(prediction).all()
        verified.append({'file': str(path.relative_to(PROJECT)), 'sha256': metadata['sha256']})
        return prediction

    controls = {}
    baseline_metrics = {}
    for seed in [42, 71]:
        item = registry['schemes'][str(seed)]
        path = V18 / item['file']
        assert sha(path) == item['sha256']
        with np.load(path, allow_pickle=False) as data:
            v9 = data['v9'].copy()
            assert np.max(np.abs(v9 - .5 * data['v7'] - .5 * data['t9'])) < .01
        p35 = np.zeros(len(y))
        seen = np.zeros(len(y), dtype=int)
        for fold, group in enumerate(np.array_split(np.random.RandomState(seed).permutation(len(y)), 5)):
            b = np.sort(group)
            a = np.setdiff1d(np.arange(len(y)), b)
            seen[b] += 1
            p35[b] = load_fold(V18, 'provenance/v14_cache', 'P35_8', seed, fold, a, b, old_sources)
        assert (seen == 1).all()
        baseline = .25 * v9 + .75 * p35
        controls[seed] = (v9, p35, baseline)
        squared = (y - baseline) ** 2
        top_five = np.argsort(squared)[-5:]
        baseline_metrics[str(seed)] = {
            'rmse_INR': rmse(y, baseline),
            'tail_gt80M_rows': int((y > 80e6).sum()),
            'tail_gt80M_squared_error_share': float(squared[y > 80e6].sum() / squared.sum()),
            'largest_five_squared_error_share': float(squared[top_five].sum() / squared.sum()),
        }

    history = {}
    maximum_difference = 0.
    for version in ['v15_retry1', 'v16', 'v17', 'v18']:
        root = PROJECT / ('improvement_' + version)
        frozen = read(root / 'source_frozen.json')
        assert frozen['inputs'] == inputs
        with (root / 'results/screening.csv').open(newline='') as stream:
            comparisons = list(csv.DictReader(stream))
        names = sorted({row['name'] for row in comparisons})
        if version == 'v18':
            names.append('UNADAPTED')
        predictions = {}
        correlations = {}
        for name in names:
            for seed in [42, 71]:
                prediction = np.zeros(len(y))
                for fold, group in enumerate(np.array_split(np.random.RandomState(seed).permutation(len(y)), 5)):
                    b = np.sort(group)
                    a = np.setdiff1d(np.arange(len(y)), b)
                    prediction[b] = load_fold(root, 'results/cache', name, seed, fold, a, b, frozen['sources'])
                predictions[name, seed] = prediction
                baseline = controls[seed][2]
                correlations.setdefault(name, {})[str(seed)] = float(np.corrcoef(y - baseline, y - prediction)[0, 1])
        reconstructed = []
        for row in comparisons:
            weight = float(row['weight'])
            gains = []
            metrics = {}
            for seed in [42, 71]:
                v9, p35, baseline = controls[seed]
                component = predictions[row['name'], seed]
                if version == 'v15_retry1':
                    candidate = .25 * v9 + .75 * ((1 - weight) * p35 + weight * component)
                else:
                    candidate = (1 - weight) * baseline + weight * component
                value = rmse(y, candidate)
                gain = rmse(y, baseline) - value
                relative = gain / rmse(y, baseline)
                for key, actual in [('rmse', value), ('gain', gain)]:
                    difference = abs(actual - float(row[f'{key}_{seed}']))
                    maximum_difference = max(maximum_difference, difference)
                    assert difference < 1e-5
                assert abs(relative - float(row[f'relative_gain_{seed}'])) < 1e-12
                gains.append(relative)
                metrics[str(seed)] = {'rmse_INR': value, 'gain_INR': gain, 'relative_gain': relative}
            mean = float(np.mean(gains))
            assert abs(mean - float(row['mean_relative_gain'])) < 1e-12
            reconstructed.append({'name': row['name'], 'weight': weight, 'mean_relative_gain': mean, 'assignments': metrics})
        best = max(reconstructed, key=lambda item: item['mean_relative_gain'])
        decision = read(root / 'results/decision.json')
        assert not decision['qualified'] and best['mean_relative_gain'] < .002
        history[version] = {
            'comparisons_verified': len(reconstructed), 'best': best,
            'candidate_residual_correlations': correlations, 'qualified': False,
        }
    report = {
        'status': 'PASS', 'checked_at_IST': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        'scope': 'Read-only historical prediction reconstruction and reference copying; no training or GPU allocation.',
        'inputs': inputs, 'training_rows': len(train), 'test_rows': len(test),
        'all_original_labels_retained': True,
        'v14_portal_rmse_INR': portal['actual_portal_rmse_INR'],
        'preserved_references': references, 'v14_development': baseline_metrics,
        'history': history, 'verified_fold_count': len(verified), 'verified_folds': verified,
        'maximum_metric_difference_INR': maximum_difference,
        'script_sha256': sha(__file__),
    }
    (ROOT / 'history_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'verified_fold_count': len(verified),
                      'maximum_metric_difference_INR': maximum_difference,
                      'v14_development': baseline_metrics,
                      'history_best_gain_percent': {k: 100 * v['best']['mean_relative_gain'] for k, v in history.items()}}, indent=2))


if __name__ == '__main__':
    main()
