"""Audited, resumable development -> locked confirmation -> final export."""
import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import catboost, lightgbm, sklearn, scipy
import reference as ref
from modeling import (REQUIRED, SEEDS, SPECS, load_data, splits, feature_frame,
                      component_oof, fit_final, predict_final)

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RESULTS = ROOT / 'results'
STEM = 'submission'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log(message):
    print(message if isinstance(message, str) else json.dumps(message), flush=True)


class Run:
    def __init__(self, data_dir):
        RESULTS.mkdir(exist_ok=True)
        self.data_dir = Path(data_dir).resolve()
        self.train, self.test, self.y = load_data(self.data_dir)
        self.inputs = {n: sha(self.data_dir / n) for n in REQUIRED}
        self.contract = json.loads((ROOT / 'v5_contract.json').read_text())
        self.sources = {n: sha(ROOT / n) for n in ['reference.py', 'modeling.py', 'workflow.py']}
        self.started = time.monotonic()

    def context(self, name, scheme):
        fold_hash = digest([b.tolist() for _, b in splits(self.train, scheme)])
        return {'inputs': self.inputs, 'sources': self.sources, 'name': name,
                'spec': SPECS.get(name), 'seeds': SEEDS, 'scheme': str(scheme),
                'fold_sha256': fold_hash, 'v5_contract': digest(self.contract)}

    def read_cache(self, name, scheme):
        path = RESULTS / f'{name}_{scheme}.npz'
        if not path.exists():
            return None
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['context'] == self.context(name, scheme), f'Stale cache: {path}'
        assert meta['sha256'] == sha(path), f'Corrupt cache: {path}'
        with np.load(path, allow_pickle=False) as saved:
            p = saved['oof'].copy()
        assert p.shape == self.y.shape and np.isfinite(p).all() and (p >= 0).all()
        return p

    def write_cache(self, name, scheme, p, individual=None, diagnostics=None, origin=None):
        path = RESULTS / f'{name}_{scheme}.npz'
        np.savez_compressed(path, oof=p, **({'individual': individual} if individual is not None else {}))
        save_json(path.with_suffix('.json'), {'context': self.context(name, scheme),
                  'sha256': sha(path), 'origin': origin or 'fresh fold fits',
                  'diagnostics': diagnostics or []})

    def audit(self):
        provenance = json.loads((ROOT / 'reference_provenance.json').read_text())
        assert provenance['reference_sha256'] == sha(ROOT / 'reference.py')
        assert all(sha(BASE / n) == h for n, h in provenance['sources'].items())
        manifest = json.loads((BASE / 'improvement_v5/results/cache_manifest.json').read_text())
        assert manifest['inputs'] == self.inputs and manifest['contract'] == digest(self.contract)
        source = BASE / 'improvement_v5' / f'{STEM}.csv'
        assert sha(source) == '66a673efc5f31bc9f4010de033999fd3712444b8f6dd0478c97e77bb3934630d'
        ref.check_submission(pd.read_csv(source), self.test)
        backup = ROOT / 'scored_v5_backup'
        backup.mkdir(exist_ok=True)
        preserved = {}
        for extension in ['csv', 'ipynb']:
            original = BASE / 'improvement_v5' / f'{STEM}.{extension}'
            destination = backup / original.name
            if not destination.exists():
                shutil.copy2(original, destination)
            assert sha(destination) == sha(original)
            preserved[original.name] = sha(original)
        changed = self.train.copy()
        changed['Annual Turnover'] = 0
        changed['Registration Number'] = -99
        pd.testing.assert_frame_equal(feature_frame(changed), feature_frame(self.train))
        assert feature_frame(self.train).columns.equals(feature_frame(self.test).columns)
        assert np.isclose(ref.rmse([0, 3], [0, 0]), np.sqrt(4.5))
        report = {'inputs': self.inputs, 'sources': self.sources, 'preserved_v5': preserved,
                  'source_provenance': provenance, 'train_shape': list(self.train.shape),
                  'test_shape': list(self.test.shape), 'target_above_80m': int((self.y > 80e6).sum()),
                  'portal_v5_rmse': 12046722.3691139, 'leader_screenshot_rmse': 4796740.5600053,
                  'portal_scoring_subset': 'unverified', 'remaining_portal_allowance': 'unverified',
                  'portal_deadline_timezone': 'unverified', 'automatic_uploads': 0,
                  'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__, 'catboost': catboost.__version__,
                              'lightgbm': lightgbm.__version__, 'sklearn': sklearn.__version__,
                              'scipy': scipy.__version__},
                  'created_utc': datetime.now(timezone.utc).isoformat()}
        save_json(RESULTS / 'audit.json', report)
        save_json(ROOT / 'experiment_plan.json', {
            'specs': SPECS, 'model_seeds': SEEDS, 'development': [42, 71],
            'confirmation': [18421, 18422, 'city'], 'candidate_weights': [.25, .5, .75, 1.],
            'maximum_shortlist': 2, 'different_families': True,
            'acceptance': 'gain > 0 on both random confirmation assignments; city relative RMSE increase <= 0.01',
            'city_groups': 'existing v5 clean city normalization; baseline refitted on identical groups',
            'maximum_portal_trials': 2, 'search_deadline_seconds': 70 * 60})
        log({'audit': 'PASS', 'inputs': self.inputs, 'preserved_v5': preserved})

    def baseline(self, scheme):
        cached = self.read_cache('v5', scheme)
        if cached is not None:
            return cached
        if str(scheme) in ['18421', '18422']:
            old = BASE / 'improvement_v5/results'
            manifest = json.loads((old / 'cache_manifest.json').read_text())
            name = f'comparison_{scheme}.npz'
            meta = json.loads((old / name).with_suffix('.json').read_text())
            assert manifest['inputs'] == self.inputs
            assert meta['context']['inputs'] == self.inputs
            assert meta['context']['contract'] == self.contract
            assert meta['context']['sources'] == json.loads((ROOT / 'reference_provenance.json').read_text())['sources']
            assert sha(old / name) == manifest['files'][name] == meta['sha256']
            p = np.load(old / name, allow_pickle=False)['candidate']
            origin = {'method': 'preserved actual v5 OOF, input/config/source/file hashes verified',
                      'path': str(old / name), 'sha256': sha(old / name)}
        else:
            log(f'Fresh frozen-v5 reference: assignment {scheme}')
            previous_splitter = ref.get_splits
            ref.get_splits = splits
            try:
                p = ref.evaluate(self.train, self.y, scheme, self.contract)['candidate']
            finally:
                ref.get_splits = previous_splitter
            origin = 'fresh frozen-v5 fits; same v6 splits, including canonical city groups'
            if str(scheme) in ['42', '71']:
                # Independently reconstruct the actual earlier development arrays.
                old = BASE / 'improvement_v5/results'
                if str(scheme) == '42':
                    prior = .5 * np.load(BASE / 'improvement_v3/results/v2_42.npz')['oof']
                    prior += .5 * np.load(BASE / 'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
                else:
                    prior = np.load(BASE / 'improvement_v3/results/comparison_71.npz')['candidate']
                reconstructed = self.contract['v3_weight'] * prior
                for name, component in self.contract['components'].items():
                    reconstructed += component['weight'] * np.load(old / f'{name}_{scheme}.npz')['oof']
                delta = float(np.max(np.abs(p - reconstructed)))
                log({'baseline_reconstruction': str(scheme), 'max_difference_INR': delta})
                assert delta < .01, 'Frozen reference differs from historical v5 development prediction'
        self.write_cache('v5', scheme, p, origin=origin)
        log({'baseline': str(scheme), 'rmse': ref.rmse(self.y, p)})
        return p

    def candidate(self, name, scheme):
        p = self.read_cache(name, scheme)
        if p is not None:
            return p
        start = time.monotonic()
        log(f'Training {name}: assignment {scheme}')
        p, individual, diagnostics = component_oof(
            self.train, self.test, self.y, SPECS[name], scheme,
            lambda s: log(f'{name} / {scheme}: {s}'))
        self.write_cache(name, scheme, p, individual, diagnostics)
        log({'name': name, 'scheme': str(scheme), 'rmse': ref.rmse(self.y, p),
             'seconds': time.monotonic() - start})
        return p

    def screen(self):
        baseline = {s: self.baseline(s) for s in [42, 71]}
        started = time.monotonic()
        rows = []
        for name, spec in SPECS.items():
            if time.monotonic() - started > 70 * 60:
                log('Predeclared screening time limit reached; no additional settings started.')
                break
            predictions = {s: self.candidate(name, s) for s in [42, 71]}
            for weight in [.25, .5, .75, 1.]:
                row = {'name': name, 'family': spec['family'], 'weight': weight}
                for s in [42, 71]:
                    p = (1 - weight) * baseline[s] + weight * predictions[s]
                    row[f'rmse_{s}'] = ref.rmse(self.y, p)
                    row[f'gain_{s}'] = ref.rmse(self.y, baseline[s]) - row[f'rmse_{s}']
                row['mean_rmse'] = np.mean([row['rmse_42'], row['rmse_71']])
                rows.append(row)
            pd.DataFrame(rows).sort_values('mean_rmse').to_csv(RESULTS / 'screening.csv', index=False)
        table = pd.DataFrame(rows).sort_values(['mean_rmse', 'name', 'weight'])
        eligible = table[(table.gain_42 > 0) & (table.gain_71 > 0)]
        chosen = []
        for row in eligible.to_dict(orient='records'):
            if row['family'] not in {c['family'] for c in chosen}:
                chosen.append(row)
            if len(chosen) == 2:
                break
        lock = {'shortlist': chosen, 'specs': SPECS, 'sources': self.sources, 'inputs': self.inputs,
                'v5_contract': digest(self.contract), 'locked_utc': datetime.now(timezone.utc).isoformat(),
                'selection': 'development only; positive gain on both assignments; different families',
                'confirmation_not_yet_examined': True}
        path = RESULTS / 'candidate_lock.json'
        if path.exists():
            previous = json.loads(path.read_text())
            assert previous['shortlist'] == chosen and previous['sources'] == self.sources
        else:
            save_json(path, lock)
        log({'FROZEN_SHORTLIST': chosen, 'lock_sha256': sha(path)})

    def confirm(self):
        lock = json.loads((RESULTS / 'candidate_lock.json').read_text())
        assert lock['sources'] == self.sources and lock['inputs'] == self.inputs
        rows, segments, folds = [], [], []
        decisions = []
        for choice in lock['shortlist']:
            gains = {}
            city_relative = None
            for scheme in [18421, 18422, 'city']:
                b = self.baseline(scheme)
                component = self.candidate(choice['name'], scheme)
                p = (1 - choice['weight']) * b + choice['weight'] * component
                old_score, score = ref.rmse(self.y, b), ref.rmse(self.y, p)
                gain = old_score - score
                gains[str(scheme)] = gain
                if scheme == 'city':
                    city_relative = score / old_score - 1
                rows.append({'name': choice['name'], 'weight': choice['weight'], 'scheme': str(scheme),
                             'v5_rmse': old_score, 'candidate_rmse': score, 'gain': gain})
                for label, mask in [('<=40M', self.y <= 40e6), ('40-80M', (self.y > 40e6) & (self.y <= 80e6)), ('>80M', self.y > 80e6)]:
                    segments.append({'name': choice['name'], 'scheme': str(scheme), 'segment': label,
                                     'rows': int(mask.sum()), 'v5_rmse': ref.rmse(self.y[mask], b[mask]),
                                     'candidate_rmse': ref.rmse(self.y[mask], p[mask]),
                                     'candidate_sse_share': float(((self.y[mask] - p[mask]) ** 2).sum() / ((self.y - p) ** 2).sum())})
                for fold, (_, indices) in enumerate(splits(self.train, scheme)):
                    folds.append({'name': choice['name'], 'scheme': str(scheme), 'fold': fold,
                                  'rows': len(indices), 'tail_rows': int((self.y[indices] > 80e6).sum()),
                                  'v5_rmse': ref.rmse(self.y[indices], b[indices]),
                                  'candidate_rmse': ref.rmse(self.y[indices], p[indices])})
                pd.DataFrame(rows).to_csv(RESULTS / 'confirmation.csv', index=False)
            accepted = gains['18421'] > 0 and gains['18422'] > 0 and city_relative <= .01
            decisions.append({**choice, 'accepted': bool(accepted), 'confirmation_gains': gains,
                              'city_relative_increase': city_relative,
                              'mean_confirmation_gain': np.mean([gains['18421'], gains['18422']])})
        pd.DataFrame(segments).to_csv(RESULTS / 'segments.csv', index=False)
        pd.DataFrame(folds).to_csv(RESULTS / 'folds.csv', index=False)
        accepted = sorted([d for d in decisions if d['accepted']], key=lambda d: -d['mean_confirmation_gain'])
        choice = accepted[0] if accepted else {'name': 'v5', 'family': 'v5', 'weight': 0.}
        save_json(RESULTS / 'decision.json', {'decision': 'qualified for a portal trial' if accepted else 'retain portal-scored v5',
                  'choice': choice, 'candidates': decisions, 'qualified': accepted,
                  'lock_sha256': sha(RESULTS / 'candidate_lock.json'),
                  'limitation': 'All splits reuse previously examined labels; portal accuracy remains unmeasured.'})
        log({'DECISION': json.loads((RESULTS / 'decision.json').read_text())})

    def fit(self):
        decision = json.loads((RESULTS / 'decision.json').read_text())
        choice = decision['choice']
        log({'fresh_full_fit': choice})
        p, fitted = fit_final(self.train, self.test, self.y, choice, self.contract)
        np.savez_compressed(RESULTS / 'full_fit.npz', prediction=p)
        predicted = predict_final(fitted, self.test)
        assert np.max(np.abs(predicted - p)) < .01
        # Always independently refit v5 once, including when the new candidate is standalone.
        if fitted['baseline'] is not None:
            v5 = ref.predict_final(self.test, fitted['baseline'], self.contract)
        else:
            _, v5, _ = ref.full_fit(self.train, self.test, self.y, self.contract)
        scored = pd.read_csv(ROOT / 'scored_v5_backup' / f'{STEM}.csv')['Annual Turnover'].to_numpy()
        assert np.max(np.abs(v5 - scored)) < .01
        checks = behavior_checks(fitted, self.test, p)
        output = ROOT / f'{STEM}.csv'
        if choice['weight'] == 0:
            shutil.copy2(ROOT / 'scored_v5_backup' / output.name, output)
        else:
            pd.DataFrame({'Registration Number': self.test['Registration Number'], 'Annual Turnover': p}).to_csv(output, index=False)
        ref.check_submission(pd.read_csv(output), self.test)
        assert np.max(np.abs(pd.read_csv(output)['Annual Turnover'].to_numpy() - p)) < .01
        checks.update(v5_full_refit_max_difference=float(np.max(np.abs(v5 - scored))),
                      prediction_rms_difference_from_v5=float(np.sqrt(np.mean((p - scored) ** 2))),
                      csv_sha256=sha(output), full_fit_reproduced=True)
        save_json(RESULTS / 'submission_checks.json', checks)
        save_json(RESULTS / 'full_fit.json', {'choice': choice, 'diagnostics': [m['diagnostics'] for m in fitted['new']],
                  'inputs': self.inputs, 'sources': self.sources, 'sha256': sha(RESULTS / 'full_fit.npz')})
        log({'export': str(output), 'checks': checks})


def behavior_checks(fitted, test, prediction):
    permutation = np.random.default_rng(91).permutation(len(test))
    assert np.max(np.abs(predict_final(fitted, test.iloc[permutation]) - prediction[permutation])) < .01
    edge = test.iloc[:3].copy()
    edge['City'] = 'never_seen_city_v6'
    edge['Cuisine'] = pd.NA
    edge['Facebook Popularity Quotient'] = np.nan
    edge['Overall Restaurant Rating'] = np.nan
    assert np.isfinite(predict_final(fitted, edge)).all()
    altered = test.copy()
    altered['Registration Number'] = -777
    altered['Annual Turnover'] = 1e30
    assert np.max(np.abs(predict_final(fitted, altered) - prediction)) < .01
    submission = pd.DataFrame({'Registration Number': test['Registration Number'], 'Annual Turnover': prediction})
    ref.check_submission(submission, test)
    invalid = [submission.iloc[:-1], pd.concat([submission, submission.iloc[:1]], ignore_index=True),
               submission.iloc[::-1].reset_index(drop=True), submission.iloc[:, ::-1],
               submission.rename(columns={'Annual Turnover': 'Turnover'}), submission.assign(index=0)]
    for column, value in [('Registration Number', 'submission'), ('Registration Number', test['Registration Number'].iloc[1]),
                          ('Annual Turnover', np.nan), ('Annual Turnover', np.inf), ('Annual Turnover', -np.inf), ('Annual Turnover', -1.)]:
        bad = submission.copy()
        if isinstance(value, str):
            bad[column] = bad[column].astype(object)
        bad.loc[0, column] = value
        invalid.append(bad)
    for bad in invalid:
        try:
            ref.check_submission(bad, test)
        except (AssertionError, ValueError, TypeError):
            continue
        raise AssertionError('A malformed submission was accepted')
    return {'row_order': True, 'unique_ids': True, 'finite_nonnegative': True,
            'missing_unseen_values': True, 'target_and_id_invariance': True,
            'permutation_invariance': True, 'malformed_cases_rejected': len(invalid),
            'rows': len(test), 'columns': submission.columns.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'screen', 'confirm', 'fit', 'all'])
    parser.add_argument('--data-dir', type=Path, default=BASE)
    args = parser.parse_args()
    run = Run(args.data_dir)
    if args.stage == 'all':
        for stage in ['audit', 'screen', 'confirm', 'fit']:
            getattr(run, stage)()
    else:
        getattr(run, args.stage)()
