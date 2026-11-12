"""Resumable bounded v7 research; exports experimental trials without claiming a portal gain."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import catboost, sklearn, scipy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import reference as ref
from models import (SPECS, SEEDS, REQUIRED, load_data, features, split_indices,
                    fit_component, predict_component, v5_fold_prediction,
                    fit_submission, predict_submission)

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OUT = ROOT / 'results'
STEM = 'submission'
CONFIRM = [18421, 18422, 'strat:27421', 'city']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log(value):
    print(value if isinstance(value, str) else json.dumps(value), flush=True)


def safe(value):
    return str(value).replace(':', '_')


class Research:
    def __init__(self, data_dir=BASE):
        self.train, self.test, self.y = load_data(data_dir)
        self.inputs = {name: sha(Path(data_dir) / name) for name in REQUIRED}
        self.sources = {name: sha(ROOT / name) for name in ['models.py', 'workflow.py', 'reference.py', 'scored_reference.json']}
        self.contract = json.loads((ROOT / 'v5_contract.json').read_text())
        OUT.mkdir(exist_ok=True)
        (OUT / 'cache').mkdir(exist_ok=True)

    def context(self, name, raw=None, extra=None):
        raw = self.train if raw is None else raw
        return {'name': name, 'inputs': self.inputs, 'sources': self.sources,
                'contract': digest(self.contract), 'spec': SPECS.get(name), 'seeds': SEEDS,
                'training_rows': digest(raw['Registration Number'].tolist()), 'extra': extra}

    def cached(self, name, context, compute):
        path = OUT / 'cache' / (safe(name) + '.npz')
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            assert record['context'] == context, f'Stale checkpoint: {path}'
            assert record['sha256'] == sha(path), f'Corrupt checkpoint: {path}'
            p = np.load(path, allow_pickle=False)['prediction'].copy()
            return p, record.get('diagnostics', {})
        p, diagnostics = compute()
        p = np.asarray(p, float)
        assert np.isfinite(p).all()
        # Rename temporary files only after a complete write, to survive interruption.
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, prediction=p)
        temporary.replace(path)
        temporary_meta = meta.with_suffix('.partial.json')
        save(temporary_meta, {'context': context, 'sha256': sha(path), 'diagnostics': diagnostics})
        temporary_meta.replace(meta)
        return p, diagnostics

    def audit(self):
        old = json.loads((ROOT / 'scored_reference.json').read_text())
        assert old['inputs'] == self.inputs
        for ext in ['csv', 'ipynb']:
            name = STEM + '.' + ext
            assert sha(ROOT / 'scored_v5_backup' / name) == old['preserved_v5'][name]
            original = BASE / 'improvement_v5' / name
            if original.exists():
                assert sha(original) == old['preserved_v5'][name]
        X, T = features(self.train), features(self.test)
        changed = self.train.copy()
        changed['Annual Turnover'] = 1e30
        changed['Registration Number'] = -77
        pd.testing.assert_frame_equal(features(changed), X)
        assert X.columns.equals(T.columns)
        assert np.isclose(ref.rmse([0, 3], [0, 0]), np.sqrt(4.5))
        # Local similarity is a diagnostic; it does not establish that a label is wrong.
        cats = X.select_dtypes(include=['object', 'string']).columns.tolist()
        nums = [c for c in X if c not in cats]
        pre = ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), nums),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
        transformed = pre.fit_transform(X)
        nn = NearestNeighbors(n_neighbors=2, n_jobs=2).fit(transformed)
        distances, indices = nn.kneighbors(transformed)
        neighbor = self.y[indices[:, 1]]
        diagnostic = self.train[['Registration Number', 'Annual Turnover', 'City', 'Resturant Tier',
                                 'Hygiene Rating', 'Facebook Popularity Quotient',
                                 'Instagram Popularity Quotient']].copy()
        diagnostic['nearest_training_id'] = self.train['Registration Number'].to_numpy()[indices[:, 1]]
        diagnostic['nearest_training_turnover'] = neighbor
        diagnostic['distance'] = distances[:, 1]
        diagnostic.loc[self.y > 80e6].sort_values('Annual Turnover', ascending=False).to_csv(OUT / 'tail_audit.csv', index=False)
        record = {'inputs': self.inputs, 'sources': self.sources, 'preserved_v5': old['preserved_v5'],
                  'target_rows': len(self.y), 'tail_above_80M': int((self.y > 80e6).sum()),
                  'nearest_neighbor_tail_above80M': int((neighbor[self.y > 80e6] > 80e6).sum()),
                  'median_nearest_target_gap_INR': float(np.median(np.abs(self.y - neighbor))),
                  'train_feature_duplicates': int(X.duplicated().sum()),
                  'cross_train_test_duplicates': int(pd.concat([X, T]).duplicated().sum() - X.duplicated().sum() - T.duplicated().sum()),
                  'labels_changed': 0, 'verified_unit_errors': 0,
                  'label_audit_conclusion': 'No independently verified unit error; retain all original labels. Dissimilarity is not evidence of an incorrect label.',
                  'portal_v5_rmse': 12046722.3691139, 'leader_screenshot_rmse': 4796740.5600053,
                  'rules_source': 'full competition instructions supplied by user on 26 October 2026',
                  'stated_metric': 'ordinary RMSE across the 500 test predictions',
                  'csv_submission_limit': None, 'final_notebook_submissions': 1,
                  'portal_uploads': 0, 'external_data_used': False,
                  'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                              'catboost': catboost.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__},
                  'utc': datetime.now(timezone.utc).isoformat()}
        save(OUT / 'audit.json', record)
        plan = {'specs': SPECS, 'seeds': SEEDS, 'development': [42, 71], 'weights': [.25, .5, .75, 1.],
                'shortlist': 'best eligible configuration from each family, development mean <=1% worse, each assignment <=2% worse',
                'confirmation': CONFIRM, 'nested_residual_training': 'three-fold v5 OOF within each outer training fold; checkpoints shared only for identical rows/configurations',
                'local_champion': 'positive gain on both random confirmation assignments; city relative increase <=1%',
                'experimental_trial': 'random confirmation mean <=1% worse; city <=3% worse; stratified <=2% worse',
                'export_limit': 2, 'export_order': 'local champions first, then mean random confirmation RMSE; distinct families',
                'scope': 'original supplied data and original code; no leaderboard-fitted row adjustments',
                'longer_time': 'per user authorization; bounded settings, no search expansion after confirmation'}
        save(ROOT / 'experiment_plan.json', plan)
        log({'audit': 'PASS', 'tail': record['tail_above_80M'], 'labels_changed': 0, 'settings': len(SPECS)})

    def baseline(self, scheme):
        name = f'v5_{safe(scheme)}'
        context = self.context('v5', extra={'scheme': str(scheme), 'folds': digest([b.tolist() for _, b in split_indices(self.train, scheme)])})

        def compute():
            if str(scheme) in ['42', '71', '18421'] and (BASE / 'improvement_v6/results' / f'v5_{scheme}.npz').exists():
                path = BASE / 'improvement_v6/results' / f'v5_{scheme}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert meta['context']['inputs'] == self.inputs
                assert meta['context']['v5_contract'] == digest(self.contract)
                assert meta['context']['fold_sha256'] == context['extra']['folds']
                assert meta['sha256'] == sha(path)
                for filename, h in meta['context']['sources'].items():
                    assert sha(BASE / 'improvement_v6' / filename) == h
                return np.load(path)['oof'], {'origin': 'verified actual v6 frozen-v5 predictions', 'source_sha256': sha(path)}
            if str(scheme) == '18422' and (BASE / 'improvement_v5/results/comparison_18422.npz').exists():
                directory = BASE / 'improvement_v5/results'
                manifest = json.loads((directory / 'cache_manifest.json').read_text())
                path = directory / 'comparison_18422.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                provenance = json.loads((ROOT / 'reference_provenance.json').read_text())
                assert meta['context']['inputs'] == self.inputs == manifest['inputs']
                assert meta['context']['contract'] == self.contract
                assert meta['context']['sources'] == provenance['sources']
                assert sha(path) == meta['sha256'] == manifest['files'][path.name]
                return np.load(path)['candidate'], {'origin': 'verified actual v5 OOF', 'source_sha256': sha(path)}
            p = np.zeros(len(self.y))
            for f, (a, b) in enumerate(split_indices(self.train, scheme)):
                fold_context = self.context('v5', self.train.iloc[a], {'scheme': str(scheme), 'fold': f, 'valid_ids': self.train.iloc[b]['Registration Number'].tolist()})
                part, _ = self.cached(f'v5_{safe(scheme)}_fold{f}', fold_context,
                    lambda: (v5_fold_prediction(self.train.iloc[a], self.y[a], self.train.iloc[b], self.contract, f), {}))
                p[b] = part
                log(f'v5 / {scheme}: fold {f + 1}/5 complete')
            return p, {'origin': 'fresh frozen-v5 folds on v7 split assignment'}

        prediction, _ = self.cached(name, context, compute)
        assert prediction.shape == self.y.shape and (prediction >= 0).all()
        return prediction

    def inner(self, raw, y, tag, seed=731):
        p = np.zeros(len(raw))
        for f, (a, b) in enumerate(split_indices(raw, f'inner:{seed}')):
            context = self.context('inner_v5', raw.iloc[a], {'tag': tag, 'fold': f,
                'valid_ids': raw.iloc[b]['Registration Number'].tolist(), 'seed': seed,
                'training_target_sha256': hashlib.sha256(np.asarray(y)[a].tobytes()).hexdigest()})
            part, _ = self.cached(f'inner_{safe(tag)}_fold{f}', context,
                lambda: (v5_fold_prediction(raw.iloc[a], np.asarray(y)[a], raw.iloc[b], self.contract, f), {}))
            p[b] = part
            log(f'Nested v5 / {tag}: inner fold {f + 1}/3 complete')
        return p

    def candidate(self, name, scheme):
        spec = SPECS[name]
        baseline = self.baseline(scheme)
        parts = np.zeros((len(self.y), len(SEEDS)))
        started = time.monotonic()
        for f, (a, b) in enumerate(split_indices(self.train, scheme)):
            # Cached fold predictions are checked before starting expensive nested fits.
            expected = [OUT / 'cache' / f'{name}_{safe(scheme)}_fold{f}_seed{seed}.npz' for seed in SEEDS]
            need_fit = not all(path.exists() for path in expected)
            inner = self.inner(self.train.iloc[a], self.y[a], f'{safe(scheme)}_outer{f}', 731 + f) if spec['family'] == 'residual' and need_fit else None
            for j, seed in enumerate(SEEDS):
                context = self.context(name, self.train.iloc[a], {'scheme': str(scheme), 'fold': f,
                    'seed': seed + f, 'valid_ids': self.train.iloc[b]['Registration Number'].tolist()})

                def compute():
                    fitted = fit_component(self.train.iloc[a], self.y[a], spec, seed + f, inner)
                    return predict_component(fitted, self.train.iloc[b], baseline[b]), fitted['info']

                part, _ = self.cached(f'{name}_{safe(scheme)}_fold{f}_seed{seed}', context, compute)
                parts[b, j] = part
            log(f'{name} / {scheme}: fold {f + 1}/5 complete')
        p = parts.mean(axis=1)
        path = OUT / f'{name}_{safe(scheme)}.npz'
        np.savez_compressed(path, oof=p, individual=parts)
        save(path.with_suffix('.json'), {'context': self.context(name, extra={'scheme': str(scheme)}), 'sha256': sha(path)})
        log({'name': name, 'scheme': str(scheme), 'rmse': ref.rmse(self.y, p), 'seconds': time.monotonic() - started})
        return p

    def screen(self):
        baseline = {s: self.baseline(s) for s in [42, 71]}
        rows = []
        for name, spec in SPECS.items():
            predictions = {s: self.candidate(name, s) for s in [42, 71]}
            for weight in [.25, .5, .75, 1.]:
                row = {'name': name, 'family': spec['family'], 'weight': weight}
                for s in [42, 71]:
                    score = ref.rmse(self.y, (1 - weight) * baseline[s] + weight * predictions[s])
                    row[f'rmse_{s}'] = score
                    row[f'gain_{s}'] = ref.rmse(self.y, baseline[s]) - score
                row['mean_rmse'] = np.mean([row['rmse_42'], row['rmse_71']])
                rows.append(row)
            pd.DataFrame(rows).sort_values('mean_rmse').to_csv(OUT / 'screening.csv', index=False)
        table = pd.DataFrame(rows).sort_values(['mean_rmse', 'name', 'weight'])
        b42, b71 = [ref.rmse(self.y, baseline[s]) for s in [42, 71]]
        eligible = table[(table.mean_rmse <= 1.01 * np.mean([b42, b71])) &
                         (table.rmse_42 <= 1.02 * b42) & (table.rmse_71 <= 1.02 * b71)]
        shortlist = eligible.groupby('family', sort=False).head(1).to_dict(orient='records')
        lock = {'shortlist': shortlist, 'inputs': self.inputs, 'sources': self.sources,
                'specs': SPECS, 'frozen_before_new_confirmation': True,
                'utc': datetime.now(timezone.utc).isoformat()}
        path = OUT / 'candidate_lock.json'
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['shortlist'] == shortlist and saved['sources'] == self.sources and saved['inputs'] == self.inputs
        else:
            save(path, lock)
        log({'FROZEN_SHORTLIST': shortlist, 'sha256': sha(path)})

    def confirm(self):
        lock = json.loads((OUT / 'candidate_lock.json').read_text())
        assert lock['inputs'] == self.inputs and lock['sources'] == self.sources
        rows, segments, folds, decisions = [], [], [], []
        for choice in lock['shortlist']:
            metrics = {}
            for scheme in CONFIRM:
                b = self.baseline(scheme)
                component = self.candidate(choice['name'], scheme)
                p = (1 - choice['weight']) * b + choice['weight'] * component
                old, new = ref.rmse(self.y, b), ref.rmse(self.y, p)
                metrics[str(scheme)] = {'baseline': old, 'candidate': new, 'gain': old - new, 'relative_change': new / old - 1}
                rows.append({'name': choice['name'], 'family': choice['family'], 'weight': choice['weight'],
                             'scheme': str(scheme), 'v5_rmse': old, 'candidate_rmse': new, 'gain': old - new})
                for label, mask in [('<=40M', self.y <= 40e6), ('40-80M', (self.y > 40e6) & (self.y <= 80e6)), ('>80M', self.y > 80e6)]:
                    segments.append({'name': choice['name'], 'scheme': str(scheme), 'segment': label,
                        'rows': int(mask.sum()), 'v5_rmse': ref.rmse(self.y[mask], b[mask]),
                        'candidate_rmse': ref.rmse(self.y[mask], p[mask]),
                        'candidate_sse_share': float(((self.y[mask] - p[mask]) ** 2).sum() / ((self.y - p) ** 2).sum())})
                for f, (_, indices) in enumerate(split_indices(self.train, scheme)):
                    folds.append({'name': choice['name'], 'scheme': str(scheme), 'fold': f,
                        'rows': len(indices), 'tail_rows': int((self.y[indices] > 80e6).sum()),
                        'v5_rmse': ref.rmse(self.y[indices], b[indices]),
                        'candidate_rmse': ref.rmse(self.y[indices], p[indices])})
                pd.DataFrame(rows).to_csv(OUT / 'confirmation.csv', index=False)
                pd.DataFrame(segments).to_csv(OUT / 'segments.csv', index=False)
                pd.DataFrame(folds).to_csv(OUT / 'folds.csv', index=False)
            gain = np.mean([metrics[s]['gain'] for s in ['18421', '18422']])
            baseline_mean = np.mean([metrics[s]['baseline'] for s in ['18421', '18422']])
            champion = all(metrics[s]['gain'] > 0 for s in ['18421', '18422']) and metrics['city']['relative_change'] <= .01
            exploratory = (-gain / baseline_mean <= .01 and metrics['city']['relative_change'] <= .03 and
                           metrics['strat:27421']['relative_change'] <= .02)
            decisions.append({**choice, 'metrics': metrics, 'local_champion': bool(champion),
                'experimental_eligible': bool(exploratory), 'mean_confirmation_gain': float(gain)})
        qualified = sorted([d for d in decisions if d['local_champion'] or d['experimental_eligible']],
                           key=lambda d: (not d['local_champion'], -d['mean_confirmation_gain']))[:2]
        save(OUT / 'decision.json', {'candidates': decisions, 'trials': qualified,
            'baseline_portal_rmse': 12046722.3691139, 'lock_sha256': sha(OUT / 'candidate_lock.json'),
            'decision': 'experimental CSVs for portal evaluation; keep best observed score' if qualified else 'no stable experimental candidate; preserve v5',
            'limitation': 'All labels have been examined previously. No portal improvement is claimed.'})
        log({'DECISION': json.loads((OUT / 'decision.json').read_text())})

    def fit(self):
        decision = json.loads((OUT / 'decision.json').read_text())
        checks = []
        for index, choice in enumerate(decision['trials'], 1):
            inner = self.inner(self.train, self.y, 'full_training', 731) if choice['family'] == 'residual' else None
            log({'fresh_full_fit': index, 'choice': choice['name'], 'weight': choice['weight']})
            p, b, fitted = fit_submission(self.train, self.test, self.y, self.contract, choice, inner)
            scored = pd.read_csv(ROOT / 'scored_v5_backup' / f'{STEM}.csv')['Annual Turnover'].to_numpy()
            assert np.max(np.abs(b - scored)) < .01
            verified = check_behavior(fitted, self.test, p)
            directory = ROOT / 'candidates' / f'trial_{index}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{STEM}.csv'
            pd.DataFrame({'Registration Number': self.test['Registration Number'], 'Annual Turnover': p}).to_csv(path, index=False)
            exported = pd.read_csv(path)
            ref.check_submission(exported, self.test)
            assert np.max(np.abs(exported['Annual Turnover'].to_numpy() - p)) < .01
            np.savez_compressed(OUT / f'full_fit_trial_{index}.npz', prediction=p, baseline=b)
            difference = float(np.sqrt(np.mean((p - scored) ** 2)))
            verified.update(trial=f'trial_{index}', choice=choice, csv_sha256=sha(path),
                prediction_rms_difference_from_v5=difference,
                conditional_portal_rmse_lower_bound=max(0., 12046722.3691139 - difference),
                can_mathematically_beat_shown_leader=bool(difference > 12046722.3691139 - 4796740.5600053),
                bound_assumptions='same 500 rows, given v5 score, ordinary unweighted RMSE as stated in rules',
                model_diagnostics=[m['info'] for m in fitted['models']])
            checks.append(verified)
            save(OUT / 'submission_checks.json', checks)
            log({'exported': str(path), 'rms_difference': difference, 'lower_bound': verified['conditional_portal_rmse_lower_bound']})


def check_behavior(fitted, test, prediction):
    permutation = np.random.default_rng(91).permutation(len(test))
    assert np.max(np.abs(predict_submission(fitted, test.iloc[permutation]) - prediction[permutation])) < .01
    edge = test.iloc[:3].copy()
    edge['City'] = 'unseen_city_v7'
    edge['Cuisine'] = pd.NA
    edge['Facebook Popularity Quotient'] = np.nan
    edge['Resturant Tier'] = np.nan
    assert np.isfinite(predict_submission(fitted, edge)).all()
    changed = test.copy()
    changed['Registration Number'] = -777
    changed['Annual Turnover'] = 1e30
    assert np.max(np.abs(predict_submission(fitted, changed) - prediction)) < .01
    frame = pd.DataFrame({'Registration Number': test['Registration Number'], 'Annual Turnover': prediction})
    ref.check_submission(frame, test)
    invalid = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]], ignore_index=True),
               frame.iloc[::-1].reset_index(drop=True), frame.iloc[:, ::-1],
               frame.rename(columns={'Annual Turnover': 'Turnover'}), frame.assign(index=0)]
    for column, value in [('Registration Number', 'submission'), ('Registration Number', test['Registration Number'].iloc[1]),
                           ('Annual Turnover', np.nan), ('Annual Turnover', np.inf), ('Annual Turnover', -np.inf), ('Annual Turnover', -1.)]:
        bad = frame.copy()
        if isinstance(value, str):
            bad[column] = bad[column].astype(object)
        bad.loc[0, column] = value
        invalid.append(bad)
    for bad in invalid:
        try:
            ref.check_submission(bad, test)
        except (AssertionError, ValueError, TypeError):
            continue
        raise AssertionError('Malformed submission accepted')
    return {'rows': len(test), 'columns': frame.columns.tolist(), 'finite_nonnegative': True,
            'row_order': True, 'unique_ids': True, 'permutation_invariance': True,
            'target_id_invariance': True, 'missing_unseen_values': True,
            'malformed_cases_rejected': len(invalid)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'screen', 'confirm', 'fit', 'all'])
    parser.add_argument('--data-dir', type=Path, default=BASE)
    args = parser.parse_args()
    run = Research(args.data_dir)
    for stage in (['audit', 'screen', 'confirm', 'fit'] if args.stage == 'all' else [args.stage]):
        getattr(run, stage)()
