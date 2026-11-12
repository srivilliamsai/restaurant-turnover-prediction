"""Recompute saved V13 evidence without fitting models or allocating compute."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def review(root):
    v13 = root / 'improvement_v13'
    registry = json.loads((v13 / 'baseline_registry.json').read_text())
    inputs = {name: sha(root / name) for name in registry['inputs']}
    assert inputs == registry['inputs']
    train = pd.read_csv(root / 'Train_dataset.csv')
    y = train['Annual Turnover'].to_numpy(float)
    assert digest(train['Registration Number'].tolist()) == registry['row_ids_sha256']
    controls, concentration, witnesses = {}, [], {}
    for scheme, record in registry['schemes'].items():
        path = v13 / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as data:
            p = data['v9'].copy()
            np.testing.assert_allclose(p, .5 * data['v7'] + .5 * data['t9'], atol=1e-7, rtol=0)
        assert p.shape == y.shape and np.isfinite(p).all()
        controls[scheme] = p
        sq = (y - p) ** 2
        concentration.append({
            'scheme': scheme, 'rmse_INR': rmse(y, p),
            'gt80M_count': int((y > 80e6).sum()),
            'gt80M_squared_error_share': float(sq[y > 80e6].sum() / sq.sum()),
            'top_five_squared_error_share': float(np.sort(sq)[-5:].sum() / sq.sum()),
        })
    table = pd.read_csv(v13 / 'delivery/screening.csv')
    maximum_difference, component_count = 0., 0
    components = {}
    runtime = json.loads((v13 / 'resolved_models.json').read_text())
    for name in ['Q99', 'Q995', 'SQ2']:
        for scheme in [42, 71]:
            prediction, covered = np.zeros(len(y)), np.zeros(len(y), int)
            for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=scheme).split(train)):
                path = v13 / f'results/cache/{name}_{scheme}_fold{fold}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                context = meta['context']
                assert sha(path) == meta['sha256']
                assert context['inputs'] == inputs and context['runtime_contract'] == runtime
                assert context['name'] == name and context['scheme'] == str(scheme)
                assert context['fold'] == fold
                assert context['train_indices'] == digest(a.tolist())
                assert context['valid_indices'] == digest(b.tolist())
                assert context['training_targets'] == hashlib.sha256(y[a].tobytes()).hexdigest()
                assert context['row_ids'] == registry['row_ids_sha256']
                with np.load(path, allow_pickle=False) as data:
                    p = data['prediction'].copy()
                assert p.shape == b.shape and np.isfinite(p).all()
                prediction[b] = p
                covered[b] += 1
                witnesses[str(path.relative_to(root))] = meta['sha256']
                component_count += 1
            assert (covered == 1).all()
            components[name, scheme] = prediction
    results = []
    for _, row in table.iterrows():
        gains, relative = [], []
        for scheme in [42, 71]:
            base = controls[str(scheme)]
            p = (1 - row['weight']) * base + row['weight'] * components[row['name'], scheme]
            score = rmse(y, p)
            maximum_difference = max(maximum_difference, abs(score - row[f'rmse_{scheme}']))
            gain = rmse(y, base) - score
            gains.append(gain)
            relative.append(gain / rmse(y, base))
        results.append({'name': row['name'], 'weight': float(row['weight']),
                        'gain_42_INR': gains[0], 'gain_71_INR': gains[1],
                        'mean_relative_gain': float(np.mean(relative)),
                        'qualified': min(gains) > 0 and np.mean(relative) >= .002})
    assert maximum_difference < .01
    assert not any(row['qualified'] for row in results)
    csv = root / 'improvement_v9/scored_v9_backup/submission.csv'
    assert sha(csv) == sha(v13 / 'reference/scored_v9.csv')
    n = json.loads((root / 'improvement_v11/delivery/resolved_models.json').read_text())
    pilot = json.loads((root / 'improvement_v11/delivery/pilot.json').read_text())
    return {
        'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved-data review only; no V14 model fitted, GPU allocated or portal submission performed.',
        'inputs': inputs, 'review_script_sha256': sha(Path(__file__)),
        'scored_v9_csv_sha256': sha(csv), 'scored_v9_portal_rmse_INR': 11904515.9864996,
        'v13_complete_candidate_folds_recomputed': component_count,
        'v13_blends_recomputed': len(results),
        'maximum_metric_difference_INR': maximum_difference,
        'v13_qualified_candidates': 0, 'v13_screening': results,
        'baseline_concentration': concentration,
        'v11_actually_evaluated_new_tabpfn_version': n['models']['N']['checkpoint']['version'],
        'v11_tabpfn_3_5_engineering_failure': n['engineering_failures']['N_pilot'],
        'v11_tabicl_engineering_result': pilot['families']['I'],
        'baseline_files': {record['file']: record['sha256'] for record in registry['schemes'].values()},
        'candidate_fold_files': witnesses,
        'limits': [
            'Conditional on historical saved fitted predictions; not a new accuracy experiment.',
            'All labels have been examined historically; concentration does not identify erroneous labels.',
            'Exact cause of the TabPFN 3.5 loading failure remains unconfirmed.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('review_evidence.json'))
    args = parser.parse_args()
    result = review(args.data_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'v13_complete_candidate_folds_recomputed',
                                           'v13_blends_recomputed', 'maximum_metric_difference_INR',
                                           'v13_qualified_candidates']}, indent=2))
