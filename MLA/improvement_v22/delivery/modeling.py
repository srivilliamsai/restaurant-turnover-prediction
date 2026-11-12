"""Two explicit official backbones; suffix-preserving verified checkpoint loading."""
import gc
import hashlib
import importlib.metadata
import inspect
import json
import shutil
import struct
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import ROOT, PINNED, sha, save

SEED = 10242


def runtime():
    import torch
    actual = {key: importlib.metadata.version(key) for key in PINNED}
    assert actual == PINNED, actual
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4'
    return {'dependencies': actual, 'gpu': 'Tesla T4', 'cuda': torch.version.cuda,
            'precision': 'float32', 'prediction_chunk': 500, 'seed': SEED,
            'datetime_dependencies': {k: importlib.metadata.version(k) for k in ['skrub', 'pyarrow']}}


def serial(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    return str(value)


def local_checkpoint(record):
    from huggingface_hub import hf_hub_download
    destination = ROOT / 'model_weights' / record['filename']
    if not destination.exists():
        path = Path(hf_hub_download(repo_id=record['repo_id'], filename=record['filename'],
                                   revision=record['revision'], token=False))
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
    assert sha(destination) == record['sha256']
    assert destination.suffix == Path(record['filename']).suffix
    if record.get('version') == '3.5':
        import tabpfn.checkpoint as loader
        assert sha(inspect.getfile(loader)) == record['installed_loader_sha256']
        assert loader.Checkpoint(destination).is_safetensors
    return str(destination)




def make_model(name, config, categorical_indices=()):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    record = config['models'][name]['checkpoint']
    version = ModelVersion.V2 if name == 'BASE' else ModelVersion.V3_5
    return TabPFNRegressor.create_default_for_version(
        version, model_path=local_checkpoint(record), n_estimators=16 if name == 'BASE' else 8,
        random_state=SEED, device='cuda', fit_mode='low_memory', memory_saving_mode=True,
        inference_precision=torch.float32, categorical_features_indices=list(categorical_indices))


def resolve():
    previous = json.loads((ROOT / 'provenance/v14_resolved_models.json').read_text())
    config = {'runtime': runtime(), 'models': {}, 'engineering_failures': {},
              'settings': features.SPECS, 'target_units': 'raw INR millions',
              'adapter': previous['adapter'], 'output_floor': 0.,
              'selection_uses_validation_rmse': False}
    for name in ['BASE', 'P35_8', *features.SPECS]:
        key = 'BASE' if name == 'BASE' else 'P35_8'
        config['models'][name] = {'checkpoint': previous['models'][key]['checkpoint']}
        try:
            model = make_model(name, config)
            params = serial(model.get_params(deep=False))
            params['model_path'] = 'checkpoint_sha256:' + config['models'][name]['checkpoint']['sha256']
            params.pop('categorical_features_indices', None)
            config['models'][name]['parameters'] = params
            del model
        except Exception as error:
            config['engineering_failures'][name] = {'error_type': type(error).__name__,
                'message': str(error)[:2000], 'traceback': traceback.format_exc()}
            config['models'].pop(name, None)
        gc.collect()
    save(ROOT / 'resolved_models.json', config)
    return config


def fit(raw, y, name):
    import torch
    config = json.loads((ROOT / 'resolved_models.json').read_text())
    assert config['runtime'] == runtime()
    assert name in config['models']
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    view = features.SPECS[name]['view'] if name in features.SPECS else 'F40'
    categorical = False
    builder = features.FeatureBuilder(view if name in features.SPECS else 'BASE').fit(raw)
    X, vocabulary = features.native_frame(builder.transform(raw))
    cats = [X.columns.get_loc(c) for c in vocabulary]
    model = make_model(name, config, cats)
    target = np.asarray(y, float) / 1e6
    assert len(target) == len(raw) and np.isfinite(target).all() and (target > 0).all()
    model.fit(X, target)
    evidence = {'name': name, 'training_rows': len(raw), 'units': 'raw INR millions',
                'checkpoint_sha256': config['models'][name]['checkpoint']['sha256'],
                'model_class': type(model).__module__ + '.' + type(model).__name__,
                'view': view, 'input_features': len(X.columns),
                'temperature': 'checkpoint default', 'feature_state_sha256': __import__('core').digest(builder.state()),
                'date_expansion': hasattr(model, 'date_transformer_'),
                'datetime_dependencies': config['runtime']['datetime_dependencies']}
    if name != 'BASE':
        architectures = [type(m).__module__ + '.' + type(m).__name__ for m in model.models_]
        configs = [type(c).__module__ + '.' + type(c).__name__ for c in model.configs_]
        evidence.update(architecture_classes=architectures, configuration_classes=configs)
        assert any('v3_5' in x.lower() or 'v3p5' in x.lower() or '3p5' in x.lower() for x in architectures + configs), evidence
    padding = 0 if name == 'BASE' else int(np.argmax(X.notna().sum(axis=1).to_numpy()))
    return {'name': name, 'vocabulary': vocabulary, 'categorical': categorical, 'feature_builder': builder,
            'view': view, 'model': model, 'padding_row': X.iloc[[padding]].copy(), 'target_metadata': evidence}


def predict(fitted, raw):
    X, _ = features.native_frame(fitted['feature_builder'].transform(raw), fitted['vocabulary'])
    parts = []
    for start in range(0, len(X), 500):
        block = X.iloc[start:start + 500].reset_index(drop=True)
        size = len(block)
        if size < 500:
            block = pd.concat([block, pd.concat([fitted['padding_row']] * (500 - size), ignore_index=True)], ignore_index=True)
        p = np.asarray(fitted['model'].predict(block, output_type='mean'), float)
        assert p.shape == (500,) and np.isfinite(p).all()
        parts.append(p[:size])
    result = np.maximum(np.concatenate(parts), 0.) * 1e6
    assert result.shape == (len(raw),) and np.isfinite(result).all()
    return result


def predict_outputs(fitted, raw):
    return {'prediction': predict(fitted, raw)}


def combine(base, component, weight, p35):
    assert weight in features.WEIGHTS
    return .25 * np.asarray(base) + .75 * ((1 - weight) * np.asarray(p35) + weight * np.asarray(component))


def release(fitted=None):
    import torch
    if fitted is not None:
        fitted.clear()
    gc.collect()
    torch.cuda.empty_cache()

