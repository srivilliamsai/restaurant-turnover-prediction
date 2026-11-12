"""Train-only feature codecs and isolated persistent candidate subprocesses."""
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import features

ROOT = Path(__file__).resolve().parent


class Codec:
    def fit(self, raw):
        frame = features.frame(raw)
        self.columns = list(frame)
        self.vocabulary = {c: sorted(frame[c].dropna().astype(str).unique().tolist())
                           for c in frame.select_dtypes(include=['object', 'string', 'category'])}
        values = self._numeric(frame)
        self.medians = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.
                                 for v in values.T])
        return self

    def _numeric(self, frame):
        result = frame.copy()
        for c, values in self.vocabulary.items():
            result[c] = frame[c].astype('string').map({v: i for i, v in enumerate(values)})
        return result.to_numpy(dtype=float, na_value=np.nan)

    def transform(self, raw):
        frame = features.frame(raw)
        assert list(frame) == self.columns
        values = self._numeric(frame)
        return np.where(np.isfinite(values), values, self.medians[None, :])

    def typed(self, raw):
        frame = features.frame(raw).copy()
        assert list(frame) == self.columns
        for c, values in self.vocabulary.items():
            column = frame[c].astype('string')
            frame[c] = pd.Categorical(column.where(column.isin(values)), categories=values)
        return frame


class RemoteModel:
    def __init__(self):
        (ROOT / 'results').mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='v19_candidate_', dir=ROOT / 'results'))
        self.snapshots = []
        self.counter = 0
        self._start()

    def _start(self):
        python = os.environ.get('MLA_V19_CANDIDATE_PYTHON', str(Path('/content/mla_v19/candidate_env/bin/python')))
        assert Path(python).is_file(), 'Isolated candidate interpreter missing'
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT)
        self.log = (self.directory / 'worker.log').open('a')
        self.process = subprocess.Popen([python, '-u', str(ROOT / 'candidate_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            env=environment, cwd=ROOT)

    def request(self, action, payload=None):
        self.counter += 1
        path = self.directory / f'input_{self.counter}.pkl'
        reply = self.directory / f'output_{self.counter}.pkl'
        with path.open('wb') as handle:
            pickle.dump(payload, handle, protocol=5)
        self.process.stdin.write(json.dumps({'action': action, 'input': str(path), 'output': str(reply)}) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Candidate worker stopped; inspect ' + str(self.directory / 'worker.log'))
            if line.startswith('MLA_RESPONSE '):
                status = json.loads(line[len('MLA_RESPONSE '):])
                break
            self.log.write(line)
        path.unlink(missing_ok=True)
        if not status['ok']:
            raise RuntimeError(status['error_type'] + ': ' + status['reason'])
        with reply.open('rb') as handle:
            result = pickle.load(handle)
        reply.unlink()
        return result

    def __getstate__(self):
        path = self.directory / f'saved_state_{len(self.snapshots)}.joblib'
        self.request('save', str(path))
        self.snapshots.append(path)
        return {'state_path': str(path)}

    def __setstate__(self, state):
        self.__init__()
        self.request('load', state['state_path'])

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if hasattr(self, 'log'):
            self.log.close()
        for path in getattr(self, 'snapshots', []):
            path.unlink(missing_ok=True)


def fit(raw, target, name, **kwargs):
    remote = RemoteModel()
    try:
        metadata = remote.request('fit', {'raw': raw, 'target': np.asarray(target, float), 'name': name})
        return {'name': name, 'model': remote, 'target_metadata': metadata}
    except BaseException:
        remote.close()
        raise


def predict(fitted, raw):
    result = fitted['model'].request('predict', raw)
    fitted['target_metadata'].update(result['resources'])
    return result['prediction']


def release(fitted):
    if fitted is not None:
        remote = fitted.get('model')
        if remote is not None:
            remote.close()
        fitted.clear()
