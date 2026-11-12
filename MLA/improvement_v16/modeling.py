"""V16 TabM wrapper and preserved V14 inference dependencies."""
import importlib.metadata
import inspect
import json
import os
import time
import features
import frozen_pfn
from core import ROOT, save


def runtime():
    return frozen_pfn.runtime()


def new_runtime():
    return {name: importlib.metadata.version(name) for name in ['tabm', 'rtdl_num_embeddings']}


def module_hashes():
    import tabm
    import rtdl_num_embeddings
    from core import sha
    return {name: sha(inspect.getfile(module)) for name, module in [('tabm', tabm), ('rtdl_num_embeddings', rtdl_num_embeddings)]}


def resolve():
    previous = json.loads((ROOT / 'provenance/v14_resolved_models.json').read_text())
    config = {'runtime': runtime(), 'new_runtime': new_runtime(), 'new_module_sha256': module_hashes(),
              'models': {name: previous['models'][name] for name in ['BASE', 'P35_8']},
              'engineering_failures': {}, 'settings': features.SPECS,
              'adapter': previous['adapter'], 'output_floor': 0.,
              'new_inference_layout': {'TM32-PLE': '500 query rows, padded with first encoded training row'},
              'selection_uses_validation_rmse': False}
    config['models']['TM32-PLE'] = {'parameters': features.SPECS['TM32-PLE']}
    save(ROOT / 'resolved_models.json', config)
    return config


def fit(raw, target, name, fold=0, scheme='full'):
    if name in ['BASE', 'P35_8']:
        return frozen_pfn.fit(raw, target, name)
    assert name == 'TM32-PLE', 'Mitra can enter only after its supported-reuse pilot passes'
    config = json.loads((ROOT / 'resolved_models.json').read_text())
    assert config['runtime'] == runtime() and config['new_runtime'] == new_runtime()
    assert config['new_module_sha256'] == module_hashes()
    import tabm_model
    import torch
    torch.cuda.reset_peak_memory_stats()
    deadline = float(os.environ['MLA_V16_TRAINING_DEADLINE']) if 'MLA_V16_TRAINING_DEADLINE' in os.environ else None
    return tabm_model.fit(raw, target, fold=fold, scheme=scheme, deadline=deadline)


def predict(fitted, raw):
    if fitted['name'] == 'TM32-PLE':
        import tabm_model
        return tabm_model.predict(fitted, raw)
    return frozen_pfn.predict(fitted, raw)


def predict_outputs(fitted, raw):
    return {'prediction': predict(fitted, raw)}


def combine(v9, component, weight, p35):
    import numpy as np
    assert weight in features.WEIGHTS
    v14 = .25 * np.asarray(v9) + .75 * np.asarray(p35)
    return (1. - weight) * v14 + weight * np.asarray(component)


def release(fitted=None):
    frozen_pfn.release(fitted)
