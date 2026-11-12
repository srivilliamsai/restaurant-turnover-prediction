"""V10 data, fold, provenance, budget and atomic checkpoint contracts."""
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import legacy_models
import features

ROOT = Path(__file__).resolve().parent
STEM = 'submission'
DEVELOPMENT = [42, 71]
CONFIRM = [48211, 48212, 48213, 'strat:27421', 'city']
PINNED = {'numpy': '2.5.4', 'pandas': '3.0.6', 'scikit-learn': '1.9.1', 'scipy': '1.18.1',
          'catboost': '1.2.10', 'interpret-core': '0.7.8', 'tabpfn': '9.2.0', 'torch': '2.11.0+cu130'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def log(value):
    print(value if isinstance(value, str) else json.dumps(value, allow_nan=False), flush=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


class BudgetExceeded(RuntimeError):
    pass


class Cache:
    def __init__(self, out, hours=18):
        assert 0 < hours <= 18
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.root = self.out/'cache'; self.root.mkdir(exist_ok=True)
        self.ledger = self.out/'compute_budget.json'; self.hours = hours; self.stage = 'audit'

    def state(self):
        return json.loads(self.ledger.read_text()) if self.ledger.exists() else {'seconds': 0., 'by_stage': {}}

    @contextmanager
    def charge(self, stage=None):
        stage = stage or self.stage
        state = self.state()
        stage_hours = {'audit': 1, 'pilot': 1, 'screen': 6, 'confirm': 7, 'export': 3}.get(stage, 18)
        limit = self.hours*3600 - (0 if stage == 'export' else min(3, self.hours/6)*3600)
        if state['seconds'] >= limit or state['by_stage'].get(stage, 0) >= stage_hours*3600:
            raise BudgetExceeded(f'V10 {stage} budget reached; preserve V9.')
        started = time.monotonic()
        try:
            yield
        finally:
            state = self.state(); elapsed = time.monotonic()-started
            state['seconds'] += elapsed
            state['by_stage'][stage] = state['by_stage'].get(stage, 0)+elapsed
            state.update(maximum_hours=self.hours, measurement='Accumulated stage computation/setup/reproduction wall time; persisted across resumes.')
            save(self.ledger, state)

    def get(self, key, context, compute):
        path = self.root/(key.replace(':','_').replace('+','_')+'.npz')
        meta = path.with_suffix('.json')
        if path.exists() or meta.exists():
            if not (path.exists() and meta.exists()):
                raise ValueError(f'Incomplete checkpoint: {key}')
            record = json.loads(meta.read_text())
            assert record['context'] == context, f'Stale checkpoint: {key}'
            assert record['sha256'] == sha(path), f'Corrupt checkpoint: {key}'
            with np.load(path, allow_pickle=False) as z:
                p = z['prediction'].copy()
            assert p.ndim == 1 and np.isfinite(p).all(), f'Invalid checkpoint: {key}'
            return p
        with self.charge():
            result = compute()
        p, extra = result if isinstance(result, tuple) else (result, {})
        p = np.asarray(p, float); assert p.ndim == 1 and np.isfinite(p).all()
        temp = path.with_name(path.stem+'.partial.npz')
        np.savez_compressed(temp, prediction=p); temp.replace(path)
        save(meta, {'context': context, 'sha256': sha(path), **extra})
        return p


class Engine:
    def __init__(self, data_dir=None, out_dir=None, hours=18):
        self.data = Path(data_dir or ROOT.parent).resolve()
        self.out = Path(out_dir or ROOT/'results').resolve()
        self.train, self.test, self.y = legacy_models.load_data(self.data)
        self.manifest = json.loads((ROOT/'reference_manifest.json').read_text())
        self.inputs = {n: sha(self.data/n) for n in legacy_models.REQUIRED}
        assert self.inputs == self.manifest['inputs'], 'Official input hashes changed.'
        self.sources = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))
                        if p.name not in ['initialize.py', 'colab_control.py', 'build_delivery.py']}
        self.sources['city_map.json'] = sha(ROOT/'city_map.json')
        self.sources['reference_manifest.json'] = sha(ROOT/'reference_manifest.json')
        self.cache = Cache(self.out, hours)
        self.row_hash = digest(self.train['Registration Number'].tolist())

    def splits(self, scheme):
        if str(scheme) != 'city':
            return legacy_models.split_indices(self.train, scheme)
        group = features.canonical_city(self.train['City']).fillna('__V10_UNKNOWN_CITY__')
        folds = list(GroupKFold(5).split(self.train, groups=group))
        for a, b in folds:
            assert set(group.iloc[a]).isdisjoint(group.iloc[b])
        return folds

    def context(self, name, scheme, fold, a, b):
        return {'name': name, 'scheme': str(scheme), 'fold': fold, 'inputs': self.inputs,
            'sources': self.sources, 'execution_contract': self.manifest['execution_contract'],
            'train_indices': digest(a.tolist()), 'valid_indices': digest(b.tolist()),
            'training_targets': hashlib.sha256(self.y[a].tobytes()).hexdigest(),
            'configuration': features.SPECS.get(name, {}), 'seed': 10242, 'estimators': 16,
            'city_fold_sha256': digest([v.tolist() for _, v in self.splits(scheme)])}

    def reference(self, scheme):
        if str(scheme) == 'city':
            path = self.out/'reference_city_v7.npz'
            meta = path.with_suffix('.json')
            if not (path.exists() and meta.exists()):
                raise FileNotFoundError('Original-platform V7 comparison for the revised city folds is pending.')
            record = json.loads(meta.read_text())
            assert record['inputs'] == self.inputs and record['platform']['system'] == 'Darwin'
            assert record['sources'] == {n: sha(ROOT/n) for n in ['baseline.py','legacy_models.py','legacy_reference.py','v5_contract.json']}
            assert record['fold_sha256'] == digest([b.tolist() for _,b in self.splits(scheme)])
            assert record['row_ids_sha256'] == self.row_hash and record['sha256'] == sha(path)
            with np.load(path,allow_pickle=False) as z:b=z['prediction'].copy()
            t = self.component('BASE','city')
            return b,t,.5*b+.5*t
        name = str(scheme).replace(':','_')
        record = self.manifest['oof'].get(name)
        if record is None:
            raise FileNotFoundError(f'Verified V9 reference absent for {scheme}; refit original-platform V7 and T9 on exact revised folds.')
        path = ROOT/record['file']; assert sha(path) == record['sha256']
        assert record['fold_sha256'] == digest([b.tolist() for _, b in self.splits(scheme)])
        assert record['row_ids_sha256'] == self.row_hash
        with np.load(path, allow_pickle=False) as z:
            b, original_t = z['v7'].copy(), z['t9'].copy()
        # Recompute the control on the exact same inference adapter as V10.
        # Original, hash-verified V9 OOF vectors remain preserved as evidence.
        t = self.component('BASE',scheme)
        differences_path=self.out/'baseline_inference_differences.json'
        differences=json.loads(differences_path.read_text()) if differences_path.exists() else {}
        differences[str(scheme)]={'max_component_difference_INR':float(np.max(np.abs(t-original_t))),
            'original_v9_rmse':rmse(self.y,.5*b+.5*original_t),'stable_v9_rmse':rmse(self.y,.5*b+.5*t),
            'reason':'Fixed 500-row internal inference batches required for INR 0.01 batch invariance; full 500-row submission shape is unchanged.'}
        save(differences_path,differences)
        assert b.shape == t.shape == self.y.shape and np.isfinite(b).all() and np.isfinite(t).all()
        return b, t, .5*b+.5*t

    def component(self, name, scheme):
        import modeling
        p = np.zeros(len(self.y))
        for f, (a, b) in enumerate(self.splits(scheme)):
            context = self.context(name, scheme, f, a, b)
            def compute(a=a, b=b):
                fitted = modeling.fit(self.train.iloc[a], self.y[a], name, log)
                values = modeling.predict(fitted, self.train.iloc[b])
                return values, {'feature_map_sha256': digest(fitted['builder'].state()),
                                'vocabulary_sha256': digest(fitted['vocabulary']),
                                'feature_count': len(fitted['builder'].columns)}
            values = self.cache.get(f'{name}_{str(scheme)}_fold{f}', context, compute)
            assert values.shape == (len(b),)
            p[b] = values
            log({'setting': name, 'scheme': str(scheme), 'fold_complete': f+1, 'folds': 5})
        return p


def validate_csv(frame, test):
    assert frame.shape == (500, 2)
    assert frame.columns.tolist() == ['Registration Number','Annual Turnover']
    assert frame['Registration Number'].equals(test['Registration Number'].reset_index(drop=True))
    assert frame['Registration Number'].is_unique
    p = frame['Annual Turnover'].to_numpy(float)
    assert np.isfinite(p).all() and (p >= 0).all()
