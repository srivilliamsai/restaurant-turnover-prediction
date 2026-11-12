"""V13 data identity, fold identity, atomic caches and persisted time limits."""
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import features
import legacy_models

ROOT = Path(__file__).resolve().parent
STEM = 'submission'
DEVELOPMENT = [42, 71]
CONFIRM = [48211, 48212, 48213, 'strat:27421', 'city']
PINNED = {'numpy': '2.5.4', 'pandas': '3.0.6', 'scikit-learn': '1.9.1',
          'scipy': '1.18.1', 'catboost': '1.2.10', 'interpret-core': '0.7.8',
          'tabpfn': '9.2.0', 'torch': '2.11.0+cu130'}
STAGE_SECONDS = {'audit': 1800, 'pilot': 3600, 'screen': 7200,
                 'confirm': 9000, 'export': 7200}
MODEL_FILES = ['core.py', 'features.py', 'modeling.py', 'workflow.py', 'verify.py',
               'delivery.py', 'fresh_worker.py', 'baseline.py', 'legacy_models.py',
               'legacy_reference.py', 'frozen_v9_modeling.py', 'v5_contract.json',
               'city_map.json', 'reference_manifest.json', 'baseline_registry.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def log(value):
    print(value if isinstance(value, str) else json.dumps(value, allow_nan=False), flush=True)


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)))


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, out):
        self.path = Path(out) / 'compute_budget.json'
        self.authorization = json.loads((ROOT / 'authorization.json').read_text())
        assert self.authorization['maximum_hours'] == 8 and not self.authorization['paid_resources']
        self.stage = None

    def state(self):
        state = json.loads(self.path.read_text()) if self.path.exists() else {'by_stage': {}, 'seconds': 0.}
        state.update(maximum_hours=8, measurement='Conservative elapsed task wall time, including setup and interruptions.',
                     seconds=max(state['seconds'], time.time() - self.authorization['started_epoch']))
        return state

    def remaining(self, stage=None):
        stage = stage or self.stage
        state = self.state()
        active = state.get('active_stage')
        pending = time.time() - active['started_epoch'] if active and active['name'] == stage else 0.
        global_remaining = 8 * 3600 - state['seconds'] - (0 if stage == 'export' else 7200)
        return max(0., min(global_remaining, STAGE_SECONDS[stage] - state['by_stage'].get(stage, 0.) - pending))

    @contextmanager
    def charge(self, stage):
        state = self.state()
        if state.get('active_stage'):
            previous = state.pop('active_stage')
            state['by_stage'][previous['name']] = state['by_stage'].get(previous['name'], 0.) + time.time() - previous['started_epoch']
            state.setdefault('interruptions', []).append(previous)
            save(self.path, state)
        if self.remaining(stage) <= 0:
            raise BudgetExceeded(f'V13 {stage} or global allowance exhausted; preserve V9.')
        self.stage = stage
        started = time.time()
        state = self.state()
        state['active_stage'] = {'name': stage, 'started_epoch': started}
        save(self.path, state)
        try:
            yield
        finally:
            state = self.state()
            state.pop('active_stage', None)
            state['by_stage'][stage] = state['by_stage'].get(stage, 0.) + time.time() - started
            save(self.path, state)

    def require(self, estimate=0.):
        if self.remaining() < max(1., estimate):
            raise BudgetExceeded(f'Insufficient V13 {self.stage} allowance; no incomplete candidate is exported.')


class Cache:
    def __init__(self, out, budget):
        self.root = Path(out) / 'cache'
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = budget

    def get(self, key, context, compute, estimate=0.):
        path = self.root / (key.replace(':', '_') + '.npz')
        metadata = path.with_suffix('.json')
        if path.exists() or metadata.exists():
            if not (path.exists() and metadata.exists()):
                raise ValueError('Incomplete checkpoint: ' + key)
            record = json.loads(metadata.read_text())
            assert record['context'] == context, 'Stale checkpoint: ' + key
            assert sha(path) == record['sha256'], 'Corrupt checkpoint: ' + key
            with np.load(path, allow_pickle=False) as data:
                values = {name: data[name].copy() for name in data.files}
            assert all(np.isfinite(v).all() for v in values.values())
            return values, record['fit_metadata']
        self.budget.require(estimate)
        values, fit_metadata = compute()
        assert all(np.isfinite(v).all() for v in values.values())
        temp = path.with_name(path.stem + '.partial.npz')
        np.savez_compressed(temp, **values)
        temp.replace(path)
        save(metadata, {'context': context, 'sha256': sha(path), 'fit_metadata': fit_metadata})
        return values, fit_metadata


class Engine:
    def __init__(self, data=None, out=None):
        self.data = Path(data or ROOT.parent).resolve()
        self.out = Path(out or ROOT / 'results').resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.train, self.test, self.y = legacy_models.load_data(self.data)
        self.manifest = json.loads((ROOT / 'reference_manifest.json').read_text())
        self.registry = json.loads((ROOT / 'baseline_registry.json').read_text())
        self.inputs = {name: sha(self.data / name) for name in legacy_models.REQUIRED}
        assert self.inputs == self.manifest['inputs'] == self.registry['inputs']
        self.row_hash = digest(self.train['Registration Number'].tolist())
        assert self.row_hash == self.registry['row_ids_sha256']
        self.sources = {name: sha(ROOT / name) for name in MODEL_FILES}
        self.budget = Budget(self.out)
        self.cache = Cache(self.out, self.budget)

    def splits(self, scheme):
        if str(scheme) != 'city':
            return legacy_models.split_indices(self.train, scheme)
        groups = features.canonical_city(self.train['City']).fillna('__V10_UNKNOWN_CITY__')
        folds = list(GroupKFold(5).split(self.train, groups=groups))
        assert all(set(groups.iloc[a]).isdisjoint(groups.iloc[b]) for a, b in folds)
        return folds

    def reference(self, scheme):
        record = self.registry['schemes'][str(scheme)]
        assert sha(ROOT / record['file']) == record['sha256']
        with np.load(ROOT / record['file'], allow_pickle=False) as data:
            return data['v9'].copy()

    def context(self, name, scheme, fold, a, b):
        return {'name': name, 'spec': features.SPECS[name], 'scheme': str(scheme), 'fold': fold,
                'inputs': self.inputs, 'sources': self.sources,
                'runtime_contract': json.loads((ROOT / 'resolved_models.json').read_text()),
                'train_indices': digest(a.tolist()), 'valid_indices': digest(b.tolist()),
                'training_targets': hashlib.sha256(self.y[a].tobytes()).hexdigest(),
                'row_ids': self.row_hash, 'target_treatment': features.training_target(self.y[a], name)[1]}

    def component(self, name, scheme):
        import modeling
        predictions = np.zeros(len(self.y))
        naive = np.zeros(len(self.y)) if name == 'SQ2' else None
        records = []
        pilot = json.loads((self.out / 'pilot.json').read_text())
        estimate = 1.3 * pilot['settings'][name]['fold_seconds']
        for fold, (a, b) in enumerate(self.splits(scheme)):
            def compute(a=a, b=b):
                fitted = modeling.fit(self.train.iloc[a], self.y[a], name)
                try:
                    result = modeling.predict_outputs(fitted, self.train.iloc[b])
                    return result, fitted['target_metadata']
                finally:
                    modeling.release(fitted)
            result, metadata = self.cache.get(f'{name}_{scheme}_fold{fold}',
                                              self.context(name, scheme, fold, a, b), compute, estimate)
            assert result['prediction'].shape == (len(b),)
            predictions[b] = result['prediction']
            if naive is not None:
                naive[b] = result['naive_squared_mean']
            records.append(metadata)
            log({'setting': name, 'scheme': str(scheme), 'fold_complete': fold + 1})
        return predictions, naive, records


def validate_csv(frame, test):
    assert frame.shape == (500, 2)
    assert frame.columns.tolist() == ['Registration Number', 'Annual Turnover']
    assert frame['Registration Number'].equals(test['Registration Number'].reset_index(drop=True))
    assert frame['Registration Number'].is_unique
    values = frame['Annual Turnover'].to_numpy(float)
    assert np.isfinite(values).all() and (values >= 0).all()
