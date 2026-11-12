"""Hash-checked computation, data contracts and frozen V7 references."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, scipy, catboost
import baseline
import legacy_models as legacy
import modeling as models

ROOT = Path(__file__).resolve().parent
STEM = 'submission'
CONFIRM = [48211, 48212, 48213, 'strat:27421', 'city']
RANDOM_CONFIRM = CONFIRM[:3]
EXPECTED_RUNTIME = {'numpy': '2.5.4', 'pandas': '3.0.6', 'sklearn': '1.9.1', 'scipy': '1.18.1', 'catboost': '1.2.10'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log(value):
    print(value if isinstance(value, str) else json.dumps(value), flush=True)


def safe(value):
    return str(value).replace(':', '_')


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


class BudgetExceeded(RuntimeError):
    pass


class Cache:
    def __init__(self, out, budget_hours=18):
        self.out = Path(out)
        self.root = self.out / 'cache'
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = self.out / 'compute_budget.json'
        self.hours = budget_hours
        self.stage = 'audit'
        self.reads = 0

    def state(self):
        return json.loads(self.budget.read_text()) if self.budget.exists() else {'seconds': 0., 'by_stage': {}}

    def get(self, name, context, compute):
        path = self.root / (safe(name) + '.npz')
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if m['context'] != context:
                raise ValueError(f'Stale checkpoint: {name}')
            if m['sha256'] != sha(path):
                raise ValueError(f'Corrupt checkpoint: {name}')
            p = np.load(path, allow_pickle=False)['prediction'].copy()
            if not np.isfinite(p).all():
                raise ValueError(f'Invalid checkpoint values: {name}')
            self.reads += 1
            return p
        state = self.state()
        reserved = min(3., self.hours / 6) * 3600
        limit = self.hours * 3600 - (0 if self.stage == 'export' else reserved)
        stage_limit = {'screen': 5, 'refine': 2, 'confirm': 7}.get(self.stage, self.hours) * 3600
        if state['seconds'] >= limit or state['by_stage'].get(self.stage, 0) >= stage_limit:
            raise BudgetExceeded(f'Compute budget reached before {name}')
        started = time.monotonic()
        try:
            p = np.asarray(compute(), float)
            assert np.isfinite(p).all()
        finally:
            state = self.state()
            seconds = time.monotonic() - started
            state['seconds'] += seconds
            state['by_stage'][self.stage] = state['by_stage'].get(self.stage, 0) + seconds
            state['maximum_hours'] = self.hours
            state['measurement'] = 'Aggregate elapsed time of checkpoint computations; fresh notebook verification is recorded separately.'
            save(self.budget, state)
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, prediction=p)
        temporary.replace(path)
        save(meta, {'context': context, 'sha256': sha(path)})
        return p


class Engine:
    def __init__(self, data_dir=None, out_dir=None, budget_hours=18):
        self.data_dir = Path(data_dir or ROOT.parent).resolve()
        self.out = Path(out_dir or ROOT / 'results').resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.train, self.test, self.y = legacy.load_data(self.data_dir)
        self.inputs = {n: sha(self.data_dir / n) for n in legacy.REQUIRED}
        self.manifest = json.loads((ROOT / 'reference_manifest.json').read_text())
        assert self.inputs == self.manifest['inputs'], 'Official inputs differ from measured data.'
        self.contract = json.loads((ROOT / 'v5_contract.json').read_text())
        self.sources = {n: sha(ROOT / n) for n in ['engine.py', 'modeling.py', 'baseline.py',
            'legacy_reference.py', 'legacy_models.py', 'v5_contract.json', 'reference_manifest.json']}
        self.runtime = {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                        'scipy': scipy.__version__, 'catboost': catboost.__version__}
        assert self.runtime == EXPECTED_RUNTIME, f'Install pinned working versions: {EXPECTED_RUNTIME}'
        self.cache = Cache(self.out, budget_hours)

    def splits(self, scheme):
        return legacy.split_indices(self.train, scheme)

    def context(self, name, a, b, extra=None):
        return {'name': name, 'inputs': self.inputs, 'sources': self.sources, 'runtime': self.runtime,
                'train_ids': digest(self.train.iloc[a]['Registration Number'].tolist()),
                'valid_ids': digest(self.train.iloc[b]['Registration Number'].tolist()),
                'training_targets': hashlib.sha256(self.y[a].tobytes()).hexdigest(), 'extra': extra}

    def baseline(self, scheme):
        folds = self.splits(scheme)
        if str(scheme) in self.manifest['snapshots']:
            m = self.manifest['snapshots'][str(scheme)]
            assert m['inputs'] == self.inputs
            assert m['train_ids'] == digest(self.train['Registration Number'].tolist())
            assert m['fold_sha256'] == digest([b.tolist() for _, b in folds])
            p = ROOT / m['file']
            assert sha(p) == m['sha256'], 'Corrupt frozen V7 snapshot'
            return np.load(p, allow_pickle=False)['prediction'].copy()
        result = np.zeros(len(self.y))
        for f, (a, b) in enumerate(folds):
            ctx = self.context('v7', a, b, {'scheme': str(scheme), 'fold': f})
            result[b] = self.cache.get(f'v7_{safe(scheme)}_fold{f}', ctx,
                lambda: baseline.fold_prediction(self.train.iloc[a], self.y[a], self.train.iloc[b], self.contract, f,
                    lambda s: log(f'V7 / {scheme} / fold {f + 1}: {s}')))
            log(f'V7 / {scheme}: fold {f + 1}/5 complete')
        return result

    def component(self, name, mode, scheme):
        if mode == 'mix50':
            # Deriving this average is identical to fitting its two constituent estimators.
            plain = self.component(name, 'plain', scheme)
            bagged = self.component(name, 'bag5', scheme)
            return .5 * (plain + bagged)
        result = np.zeros(len(self.y))
        spec = models.SPECS[name]
        for f, (a, b) in enumerate(self.splits(scheme)):
            ctx = self.context(name, a, b, {'scheme': str(scheme), 'fold': f, 'mode': mode, 'spec': spec})
            result[b] = self.cache.get(f'{name}_{mode}_{safe(scheme)}_fold{f}', ctx,
                lambda: models.fit_predict(self.train.iloc[a], self.y[a], self.train.iloc[b], spec, mode,
                    lambda s: log(f'{name} / {scheme} / fold {f + 1}: {s}')))
        log({'component': name, 'mode': mode, 'scheme': str(scheme), 'rmse': rmse(self.y, result)})
        return result


def validate_csv(frame, test):
    assert frame.shape == (500, 2)
    assert frame.columns.tolist() == ['Registration Number', 'Annual Turnover']
    assert frame['Registration Number'].equals(test['Registration Number'].reset_index(drop=True))
    assert frame['Registration Number'].is_unique
    p = frame['Annual Turnover'].to_numpy(float)
    assert np.isfinite(p).all() and (p >= 0).all()


def check_behavior(predict, test, prediction):
    perm = np.random.default_rng(91).permutation(len(test))
    assert np.max(np.abs(predict(test.iloc[perm]) - prediction[perm])) < .01
    changed = test.copy()
    changed['Registration Number'] = -7
    changed['Annual Turnover'] = 1e30
    assert np.max(np.abs(predict(changed) - prediction)) < .01
    edge = test.iloc[:3].copy()
    edge['City'] = 'unseen_city_v8'
    edge['Cuisine'] = pd.NA
    edge['Restaurant Theme'] = 'unseen_theme_v8'
    edge['Facebook Popularity Quotient'] = np.nan
    edge['Instagram Popularity Quotient'] = np.nan
    edge['Resturant Tier'] = np.nan
    edge['Opening Day of Restaurant'] = 'invalid_date'
    e = predict(edge)
    assert np.isfinite(e).all() and (e >= 0).all()
    frame = pd.DataFrame({'Registration Number': test['Registration Number'].to_numpy(), 'Annual Turnover': prediction})
    validate_csv(frame, test)
    malformed = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]], ignore_index=True),
        frame.iloc[::-1].reset_index(drop=True), frame.iloc[:, ::-1],
        frame.rename(columns={'Annual Turnover': 'Turnover'}), frame.assign(index=0)]
    for col, value in [('Registration Number', 'submission'), ('Registration Number', test['Registration Number'].iloc[1]),
                       ('Annual Turnover', np.nan), ('Annual Turnover', np.inf), ('Annual Turnover', -np.inf), ('Annual Turnover', -1.)]:
        bad = frame.copy()
        if isinstance(value, str):
            bad[col] = bad[col].astype(object)
        bad.loc[0, col] = value
        malformed.append(bad)
    for bad in malformed:
        try:
            validate_csv(bad, test)
        except (AssertionError, TypeError, ValueError):
            continue
        raise AssertionError('Malformed CSV accepted')
    return {'rows': 500, 'columns': frame.columns.tolist(), 'original_row_order': True, 'unique_ids': True,
            'finite_nonnegative': True, 'permutation_invariance': True, 'target_id_invariance': True,
            'missing_and_unseen_values': True, 'malformed_cases_rejected': len(malformed)}
