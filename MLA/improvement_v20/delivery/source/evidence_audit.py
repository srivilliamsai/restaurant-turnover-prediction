"""Read-only analysis of the original inputs and saved V19 predictions; no fits."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V19 = PROJECT / 'improvement_v19'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def ks_statistic(a, b):
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.unique(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side='right') / len(a)
                              - np.searchsorted(b, grid, side='right') / len(b))))


def main():
    registry = read(V19 / 'baseline_registry.json')
    inputs = {name: sha(PROJECT / name) for name in registry['inputs']}
    assert inputs == registry['inputs']
    train = pd.read_csv(PROJECT / 'Train_dataset.csv').rename(columns={'Endorsed By': 'Endoresed By'})
    test = pd.read_csv(PROJECT / 'Test_dataset.csv')
    y = train['Annual Turnover'].to_numpy(float)
    ids = train['Registration Number'].tolist()
    assert digest(ids) == registry['row_ids_sha256']
    assert train['Registration Number'].is_unique and test['Registration Number'].is_unique
    assert set(ids).isdisjoint(test['Registration Number'])
    assert len(train) == 3493 and len(test) == 500 and np.isfinite(y).all()
    X = train.drop(columns=['Registration Number', 'Annual Turnover'])
    Z = test.drop(columns='Registration Number')
    assert list(X) == list(Z)
    frozen = read(V19 / 'source_frozen.json')['sources']
    control_sources = read(V19 / 'provenance/v14_source_frozen.json')['sources']
    verified = []

    def load_fold(directory, name, seed, fold, a, b, sources):
        path = V19 / directory / f'{name}_{seed}_fold{fold}.npz'
        meta = read(path.with_suffix('.json'))
        context = meta['context']
        assert sha(path) == meta['sha256']
        assert context['inputs'] == inputs and context['sources'] == sources
        assert context['name'] == name
        assert context['train_indices'] == digest(a.tolist())
        assert context['valid_indices'] == digest(b.tolist())
        assert context['training_targets'] == hashlib.sha256(y[a].tobytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            prediction = data['prediction'].copy()
        assert prediction.shape == b.shape and np.isfinite(prediction).all()
        verified.append({'file': str(path.relative_to(PROJECT)), 'sha256': meta['sha256']})
        return prediction

    controls, components, diagnostics = {}, {}, {}
    for seed in [42, 71]:
        reference = registry['schemes'][str(seed)]
        path = V19 / reference['file']
        assert sha(path) == reference['sha256']
        with np.load(path, allow_pickle=False) as data:
            v9 = data['v9'].copy()
            assert np.max(np.abs(v9 - .5 * data['v7'] - .5 * data['t9'])) < .01
        predictions = {name: np.zeros(len(y)) for name in ['P35_8', 'C8', 'D8']}
        seen = np.zeros(len(y), int)
        for fold, group in enumerate(np.array_split(np.random.RandomState(seed).permutation(len(y)), 5)):
            b = np.sort(group)
            a = np.setdiff1d(np.arange(len(y)), b)
            seen[b] += 1
            for name in predictions:
                directory = 'provenance/v14_cache' if name == 'P35_8' else 'results/cache'
                sources = control_sources if name == 'P35_8' else frozen
                predictions[name][b] = load_fold(directory, name, seed, fold, a, b, sources)
        assert (seen == 1).all()
        controls[seed] = .25 * v9 + .75 * predictions['P35_8']
        components[seed] = {name: predictions[name] for name in ['C8', 'D8']}
        components[seed]['CORE_EQUAL'] = .5 * (predictions['C8'] + predictions['D8'])
        error = y - controls[seed]
        squared = error ** 2
        diagnostics[str(seed)] = {
            'v14_rmse_INR': rmse(y, controls[seed]),
            'tail_gt80M_squared_error_share': float(squared[y > 80e6].sum() / squared.sum()),
            'largest_five_squared_error_share': float(np.sort(squared)[-5:].sum() / squared.sum()),
            'components': {},
        }
        for name, pred in components[seed].items():
            diagnostics[str(seed)]['components'][name] = {
                'rmse_INR': rmse(y, pred),
                'residual_correlation': float(np.corrcoef(error, y - pred)[0, 1]),
                'residual_correlation_target_le80M': float(np.corrcoef(error[y <= 80e6], (y - pred)[y <= 80e6])[0, 1]),
            }
    with (V19 / 'results/screening.csv').open(newline='') as stream:
        saved_rows = list(csv.DictReader(stream))
    reconstructed = []
    max_difference = 0.
    for row in saved_rows:
        weight = float(row['weight'])
        metrics, gains = {}, []
        for seed in [42, 71]:
            pred = (1 - weight) * controls[seed] + weight * components[seed][row['name']]
            score = rmse(y, pred)
            gain = rmse(y, controls[seed]) - score
            relative = gain / rmse(y, controls[seed])
            difference = abs(score - float(row[f'rmse_{seed}']))
            max_difference = max(max_difference, difference)
            assert difference < 1e-5
            assert abs(gain - float(row[f'gain_{seed}'])) < 1e-5
            metrics[str(seed)] = {'rmse_INR': score, 'gain_INR': gain}
            gains.append(relative)
        mean = float(np.mean(gains))
        assert abs(mean - float(row['mean_relative_gain'])) < 1e-12
        reconstructed.append({'name': row['name'], 'weight': weight, 'mean_relative_gain': mean, 'assignments': metrics})
    best = max(reconstructed, key=lambda row: row['mean_relative_gain'])
    assert len(saved_rows) == 15 and best['mean_relative_gain'] < .002
    shift = {'numeric_KS_statistics': {}, 'categorical_total_variation': {}}
    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]):
            a, b = X[column].dropna(), Z[column].dropna()
            shift['numeric_KS_statistics'][column] = ks_statistic(a, b)
        elif column != 'Opening Day of Restaurant':
            a = X[column].fillna('<missing>').astype(str).value_counts(normalize=True)
            b = Z[column].fillna('<missing>').astype(str).value_counts(normalize=True)
            values = a.index.union(b.index)
            shift['categorical_total_variation'][column] = float(.5 * np.abs(a.reindex(values, fill_value=0) - b.reindex(values, fill_value=0)).sum())
    xhash = pd.util.hash_pandas_object(X, index=False)
    zhash = pd.util.hash_pandas_object(Z, index=False)
    portal = read(V19 / 'reference/v14_portal_trial.json')
    references = {}
    for suffix, key in [('csv', 'csv_sha256'), ('ipynb', 'notebook_sha256')]:
        path = V19 / 'reference' / ('scored_v14.' + suffix)
        assert sha(path) == portal[key]
        references[str(path.relative_to(PROJECT))] = sha(path)
    pilot_i = read(PROJECT / 'improvement_v11/results/pilot.json')['families']['I']
    pilot_k = read(V19 / 'results/pilot.json')['settings']['KS8']
    report = {
        'status': 'PASS', 'checked_at_IST': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        'scope': 'Original input checks and saved-prediction reconstruction only; no model fits, installations or runtime allocation.',
        'script_sha256': sha(Path(__file__)), 'inputs': inputs, 'references': references,
        'training_rows': len(y), 'test_rows': len(test), 'source_predictors': len(X.columns),
        'target_quantiles_INR': dict(zip(['min', 'p50', 'p95', 'p99', 'max'], np.quantile(y, [0, .5, .95, .99, 1]).tolist())),
        'tail_gt80M_rows': int((y > 80e6).sum()),
        'exact_duplicate_predictor_rows_train': int(X.duplicated(keep=False).sum()),
        'exact_duplicate_predictor_rows_test': int(Z.duplicated(keep=False).sum()),
        'exact_train_test_predictor_matches': int(zhash.isin(xhash).sum()),
        'missing_train': {c: int(n) for c, n in X.isna().sum().items() if n},
        'missing_test': {c: int(n) for c, n in Z.isna().sum().items() if n},
        'covariate_diagnostics': shift,
        'covariate_diagnostics_limitation': 'Raw marginal distributions only; these do not identify the hidden target distribution or justify label removal/reweighting.',
        'verified_fold_outputs': verified, 'candidate_fold_outputs': 20, 'control_fold_outputs': 10,
        'comparisons_reconstructed': len(reconstructed), 'max_metric_difference_INR': max_difference,
        'v19_best': best, 'v14_portal_rmse_INR': portal['actual_portal_rmse_INR'],
        'diagnostics': diagnostics,
        'untested_accuracy_families': {'TabICLv2_V11': pilot_i, 'Kumo_Small_V19': {k: pilot_k[k] for k in ['status', 'error_type', 'reason', 'accuracy_tested']}},
        'comparisons': reconstructed,
    }
    (ROOT / 'evidence_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'training_rows', 'test_rows', 'target_quantiles_INR', 'tail_gt80M_rows', 'exact_duplicate_predictor_rows_train', 'exact_train_test_predictor_matches', 'candidate_fold_outputs', 'comparisons_reconstructed', 'max_metric_difference_INR', 'v19_best', 'diagnostics']}, indent=2))


if __name__ == '__main__':
    main()
