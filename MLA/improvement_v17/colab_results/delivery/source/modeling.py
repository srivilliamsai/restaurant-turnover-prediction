"""Frozen V14 controls and declared V17 components."""
import importlib.metadata
import json
from pathlib import Path
import features
import frozen_pfn
from core import ROOT, save, sha


def runtime():
    return frozen_pfn.runtime()


def new_runtime():
    return {name:importlib.metadata.version(name) for name in ['einops','numba','llvmlite','kditransform','pyvinecopulib','synthefy','pyyaml','pydantic','setuptools']}


def module_hashes():
    result = {}
    for key, folder in [('xiaomi','xiaomi/tabldm'),('nori','nori/src/synthefy_nori')]:
        result[key] = {str(p.relative_to(ROOT/'engineering_sources')):sha(p)
                       for p in sorted((ROOT/'engineering_sources'/folder).rglob('*'))
                       if p.is_file() and p.suffix in ['.py','.json']}
    return result


def resolve():
    previous = json.loads((ROOT/'provenance/v14_resolved_models.json').read_text())
    config = {'runtime':runtime(),'new_runtime':new_runtime(),'new_module_sha256':module_hashes(),
              'models':{name:previous['models'][name] for name in ['BASE','P35_8']},
              'engineering_failures':{},'settings':features.SPECS,'adapter':previous['adapter'],
              'output_floor':0.,'selection_uses_validation_rmse':False,
              'source_registry':json.loads((ROOT/'source_registry.json').read_text()),
              'checkpoint_registry':json.loads((ROOT/'checkpoint_registry.json').read_text()),
              'new_inference_layout':'500 rows, fixed first training-row padding; float32 inference'}
    save(ROOT/'resolved_models.json',config)
    return config


def fit(raw,target,name,fold=0,scheme='full'):
    if name in ['BASE','P35_8']:
        return frozen_pfn.fit(raw,target,name)
    assert name in features.SPECS
    import new_models
    config = json.loads((ROOT/'resolved_models.json').read_text())
    assert config['runtime'] == runtime() and config['new_runtime'] == new_runtime()
    assert config['new_module_sha256'] == module_hashes()
    return new_models.fit(raw,target,name,fold,scheme)


def predict(fitted,raw):
    if fitted['name'] in features.SPECS:
        import new_models
        return new_models.predict(fitted,raw)
    return frozen_pfn.predict(fitted,raw)


def predict_outputs(fitted,raw):
    return {'prediction':predict(fitted,raw)}


def combine(v9,component,weight,p35):
    import numpy as np
    assert weight in features.WEIGHTS
    return (1.-weight)*(.25*np.asarray(v9)+.75*np.asarray(p35))+weight*np.asarray(component)


def release(fitted=None):
    frozen_pfn.release(fitted)
