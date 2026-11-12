"""Frozen V14 control process and isolated V19 candidates."""
import importlib.metadata
import json
import numpy as np
import features
import frozen_pfn
from core import ROOT, save


def runtime():
    return frozen_pfn.runtime()


def new_runtime():
    return {name: importlib.metadata.version(name) for name in
            ['safetensors', 'huggingface-hub', 'nbformat', 'nbclient']}


def resolve():
    from new_models import RemoteModel
    previous = json.loads((ROOT / 'provenance/v14_resolved_models.json').read_text())
    remote = RemoteModel()
    try:
        candidate_runtime = remote.request('inspect')
    finally:
        remote.close()
    config = {'runtime': runtime(), 'new_runtime': new_runtime(), 'candidate_runtime': candidate_runtime,
              'models': {name: previous['models'][name] for name in ['BASE', 'P35_8']},
              'settings': features.SPECS, 'adapter': previous['adapter'], 'output_floor': 0.,
              'selection_uses_validation_rmse': False,
              'source_registry': json.loads((ROOT / 'source_registry.json').read_text()),
              'checkpoint_registry': json.loads((ROOT / 'checkpoint_registry.json').read_text())}
    save(ROOT / 'resolved_models.json', config)
    return config


def fit(raw, target, name, fold=0, scheme='full'):
    if name in ['BASE', 'P35_8']:
        return frozen_pfn.fit(raw, target, name)
    assert name in features.SPECS
    import new_models
    fitted = new_models.fit(raw, target, name, fold=fold, scheme=scheme)
    contract = json.loads((ROOT / 'resolved_models.json').read_text())
    assert fitted['target_metadata']['runtime'] == contract['candidate_runtime']
    return fitted


def predict(fitted, raw):
    if fitted['name'] in features.SPECS:
        import new_models
        return new_models.predict(fitted, raw)
    return frozen_pfn.predict(fitted, raw)


def predict_outputs(fitted, raw):
    return {'prediction': predict(fitted, raw)}


def combine(v9, component, weight, p35):
    assert weight in features.WEIGHTS
    return (1. - weight) * (.25 * np.asarray(v9) + .75 * np.asarray(p35)) + weight * np.asarray(component)


def release(fitted=None):
    if fitted and fitted.get('name') in features.SPECS:
        import new_models
        new_models.release(fitted)
    else:
        frozen_pfn.release(fitted)
