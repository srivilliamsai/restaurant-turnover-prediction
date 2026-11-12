"""V19 data identity, fold identity, atomic caches and persisted time limits."""
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
CONFIRM = [919601, 919602, 919603, 'strat:27421', 'city']
PINNED = {'numpy': '2.5.4', 'pandas': '3.0.6', 'scikit-learn': '1.9.1',
          'scipy': '1.18.1', 'catboost': '1.2.10', 'interpret-core': '0.7.8',
          'tabpfn': '9.2.0', 'tabicl': '2.2.0', 'torch': '2.11.0+cu130'}
STAGE_SECONDS = {stage:6.5*3600 for stage in ['audit','pilot','screen','confirm']}
STAGE_SECONDS['export']=1.5*3600
MODEL_FILES = ['core.py', 'features.py', 'modeling.py', 'workflow.py', 'verify.py', 'delivery.py', 'fresh_worker.py', 'baseline.py', 'legacy_models.py', 'legacy_reference.py', 'frozen_v9_modeling.py', 'v5_contract.json', 'city_map.json', 'reference_manifest.json', 'baseline_registry.json', 'frozen_pfn.py', 'new_models.py', 'candidate_models.py', 'candidate_worker.py', 'pilot_worker.py', 'checkpoint_registry.json', 'source_registry.json', 'protocol_proposed.json', 'mac_controls.py', 'control_worker.py', 'setup_environment.py', 'runtime_lock.json']



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
        hard_remaining = self.authorization['hard_stop_epoch'] - time.time()
        if stage != 'export':
            pass
        global_remaining = min(8 * 3600 - state['seconds'], hard_remaining) - (0 if stage == 'export' else 5400)
        if stage != 'export':
            global_remaining = min(global_remaining, self.authorization['training_stop_epoch'] - time.time())
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
            raise BudgetExceeded(f'V19 {stage} or global allowance exhausted; preserve V14.')
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
            raise BudgetExceeded(f'Insufficient V19 {self.stage} allowance; no incomplete candidate is exported.')


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
        return self.reference_parts(scheme)['v14']

    def reference_parts(self, scheme):
        if str(scheme) not in self.registry['schemes']:
            return self.new_reference_parts(scheme)
        record = self.registry['schemes'][str(scheme)]
        assert sha(ROOT / record['file']) == record['sha256']
        with np.load(ROOT / record['file'], allow_pickle=False) as data:
            parts = {key: data[key].copy() for key in ['v7', 't9', 'v9']}
        p = np.zeros(len(self.y))
        frozen = json.loads((ROOT / 'provenance/v14_source_frozen.json').read_text())
        contract = json.loads((ROOT / 'provenance/v14_resolved_models.json').read_text())
        for fold, (a, b) in enumerate(self.splits(scheme)):
            path = ROOT / 'provenance/v14_cache' / f"P35_8_{str(scheme).replace(':', '_')}_fold{fold}.npz"
            meta = json.loads(path.with_suffix('.json').read_text())
            context = meta['context']
            assert meta['sha256'] == sha(path)
            assert context['sources'] == frozen['sources'] and context['inputs'] == self.inputs
            assert context['row_ids'] == self.row_hash
            assert context['train_indices'] == digest(a.tolist()) and context['valid_indices'] == digest(b.tolist())
            assert context['training_targets'] == hashlib.sha256(self.y[a].tobytes()).hexdigest()
            assert context['runtime_contract'] == contract and context['name'] == 'P35_8'
            assert context['spec'] == contract['settings']['P35_8']
            for name, expected in frozen['sources'].items():
                assert sha(ROOT / 'provenance/v14_sources' / name) == expected
            with np.load(path, allow_pickle=False) as data:
                assert data['prediction'].shape == b.shape and np.isfinite(data['prediction']).all()
                p[b] = data['prediction']
        parts['p35'] = p
        parts['v14'] = .25 * parts['v9'] + .75 * p
        return parts

    def new_reference_parts(self, scheme):
        import modeling
        assert scheme in CONFIRM[:3]
        parts={name:np.zeros(len(self.y)) for name in ['v7','t9','p35']}
        for fold,(a,b) in enumerate(self.splits(scheme)):
            from mac_controls import read_fold
            parts['v7'][b]=read_fold(self,scheme,fold,a,b)
            for name,key in [('BASE','t9'),('P35_8','p35')]:
                def compute(a=a,b=b,name=name):
                    fit=modeling.fit(self.train.iloc[a],self.y[a],name)
                    try:return {'prediction':modeling.predict(fit,self.train.iloc[b])},fit['target_metadata']
                    finally:modeling.release(fit)
                estimate=json.loads((self.out/'pilot.json').read_text())['control_timings'][name]['seconds']*1.3
                result,_=self.cache.get(f'CONTROL_{name}_{scheme}_fold{fold}',
                    self.context(name,scheme,fold,a,b),compute,estimate)
                parts[key][b]=result['prediction']
                log({'new_control':name,'scheme':str(scheme),'fold_complete':fold+1})
        parts['v9']=.5*parts['v7']+.5*parts['t9']
        parts['v14']=.25*parts['v9']+.75*parts['p35']
        return parts

    def context(self, name, scheme, fold, a, b):
        return {'name': name, 'spec': features.SPECS.get(name,{'diagnostic_or_control':name}), 'scheme': str(scheme), 'fold': fold,
                'inputs': self.inputs, 'sources': self.sources,
                'runtime_contract': json.loads((ROOT / 'resolved_models.json').read_text()),
                'train_indices': digest(a.tolist()), 'valid_indices': digest(b.tolist()),
                'training_targets': hashlib.sha256(self.y[a].tobytes()).hexdigest(),
                'row_ids': self.row_hash, 'target_units': 'raw INR millions'}

    def component(self, name, scheme):
        import modeling
        predictions = np.zeros(len(self.y))
        naive = None
        records = []
        pilot = json.loads((self.out / 'pilot.json').read_text())
        estimate = 1.3 * (pilot['settings'][name]['fold_seconds'])
        for fold, (a, b) in enumerate(self.splits(scheme)):
            def compute(a=a, b=b):
                fitted = modeling.fit(self.train.iloc[a], self.y[a], name, fold=fold, scheme=scheme)
                try:
                    result = modeling.predict_outputs(fitted, self.train.iloc[b])
                    return result, fitted['target_metadata']
                finally:
                    modeling.release(fitted)
            result, metadata = self.cache.get(f'{name}_{scheme}_fold{fold}',
                                              self.context(name, scheme, fold, a, b), compute, estimate)
            assert result['prediction'].shape == (len(b),)
            predictions[b] = result['prediction']
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

