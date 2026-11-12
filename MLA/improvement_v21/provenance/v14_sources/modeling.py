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
REPOS = {
    'I8': ('jingang/TabICL', 'tabicl-regressor-v2-20260212.ckpt'),
    'P35_8': ('Prior-Labs/tabpfn_3_5', 'tabpfn-v3.5-20260909.safetensors'),
}


def runtime():
    import torch
    actual = {key: importlib.metadata.version(key) for key in PINNED}
    assert actual == PINNED, actual
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4'
    return {'dependencies': actual, 'gpu': 'Tesla T4', 'cuda': torch.version.cuda,
            'precision': 'float32', 'prediction_chunk': 500, 'seed': SEED}


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
    return str(destination)


def resolve_checkpoint(name):
    from huggingface_hub import HfApi, hf_hub_download
    repo, filename = REPOS[name]
    revision = HfApi(token=False).model_info(repo_id=repo).sha
    path = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision, token=False))
    record = {'repo_id': repo, 'revision': revision, 'filename': filename, 'sha256': sha(path),
              'bytes': path.stat().st_size, 'version': '3.5' if name == 'P35_8' else 'TabICLv2'}
    named = Path(local_checkpoint(record))
    record['download_path'] = str(path)
    record['local_loader_path'] = str(named)
    record['local_suffix'] = named.suffix
    if name == 'P35_8':
        with named.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            assert 2 <= length <= min(named.stat().st_size - 8, 100_000_000)
            header = json.loads(handle.read(length))
        record['format_check'] = {'format': 'safetensors', 'header_bytes': length,
                                  'tensor_entries': len(header) - int('__metadata__' in header),
                                  'metadata_keys': sorted(header.get('__metadata__', {}))}
        import tabpfn.checkpoint as loader
        record['installed_loader_path'] = inspect.getfile(loader)
        record['installed_loader_sha256'] = sha(inspect.getfile(loader))
        assert loader.Checkpoint(named).is_safetensors
    return record


def make_model(name, config, categorical_indices=()):
    import torch
    record = config['models'][name]['checkpoint']
    path = local_checkpoint(record)
    if name == 'I8':
        from tabicl import TabICLRegressor
        return TabICLRegressor(n_estimators=8, norm_methods=None, feat_shuffle_method='latin',
                              batch_size=1, kv_cache=True, model_path=path,
                              allow_auto_download=False, checkpoint_version=record['filename'],
                              device='cuda', use_amp=False, use_fa3=False,
                              random_state=SEED, n_jobs=2, verbose=False)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    version = ModelVersion.V2 if name == 'BASE' else ModelVersion.V3_5
    return TabPFNRegressor.create_default_for_version(
        version, model_path=path, n_estimators=16 if name == 'BASE' else 8,
        random_state=SEED, device='cuda', fit_mode='low_memory', memory_saving_mode=True,
        inference_precision=torch.float32, categorical_features_indices=list(categorical_indices))


def resolve():
    config = {'runtime': runtime(), 'models': {}, 'engineering_failures': {},
              'settings': features.SPECS, 'target_units': 'raw INR millions',
              'adapter': 'Fixed 500-row blocks; BASE first training row; new model maximum-observed training row; discard padding.',
              'output_floor': 0., 'selection_uses_validation_rmse': False}
    v2 = json.loads((ROOT / 'reference_manifest.json').read_text())['execution_contract']['checkpoint']
    config['models']['BASE'] = {'checkpoint': {**v2, 'version': '2'}}
    for name in features.SPECS:
        try:
            config['models'][name] = {'checkpoint': resolve_checkpoint(name)}
            model = make_model(name, config)
            params = serial(model.get_params(deep=False))
            params['model_path'] = 'checkpoint_sha256:' + config['models'][name]['checkpoint']['sha256']
            params.pop('categorical_features_indices', None)
            config['models'][name]['parameters'] = params
            del model
        except Exception as error:
            config['engineering_failures'][name] = {
                'error_type': type(error).__name__, 'message': str(error)[:2000],
                'traceback': traceback.format_exc(), 'stage': 'checkpoint_and_constructor'}
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
    categorical = name == 'I8'
    X, vocabulary = features.native_frame(features.frame(raw), categorical=categorical)
    cats = [X.columns.get_loc(c) for c in vocabulary]
    model = make_model(name, config, cats)
    target = np.asarray(y, float) / 1e6
    assert len(target) == len(raw) and np.isfinite(target).all() and (target > 0).all()
    model.fit(X, target)
    evidence = {'name': name, 'training_rows': len(raw), 'units': 'raw INR millions',
                'checkpoint_sha256': config['models'][name]['checkpoint']['sha256'],
                'model_class': type(model).__module__ + '.' + type(model).__name__}
    if name == 'P35_8':
        architectures = [type(m).__module__ + '.' + type(m).__name__ for m in model.models_]
        configs = [type(c).__module__ + '.' + type(c).__name__ for c in model.configs_]
        evidence.update(architecture_classes=architectures, configuration_classes=configs)
        assert any('v3_5' in x.lower() or 'v3p5' in x.lower() or '3p5' in x.lower() for x in architectures + configs), evidence
    padding = 0 if name == 'BASE' else int(np.argmax(X.notna().sum(axis=1).to_numpy()))
    return {'name': name, 'vocabulary': vocabulary, 'categorical': categorical,
            'model': model, 'padding_row': X.iloc[[padding]].copy(), 'target_metadata': evidence}


def predict(fitted, raw):
    X, _ = features.native_frame(features.frame(raw), fitted['vocabulary'], fitted['categorical'])
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


def combine(base, component, weight, route='A', v7=None, t9=None):
    if route == 'A':
        return (1 - weight) * np.asarray(base) + weight * np.asarray(component)
    assert route == 'B' and v7 is not None and t9 is not None
    return .5 * np.asarray(v7) + .5 * ((1 - weight) * np.asarray(t9) + weight * np.asarray(component))


def release(fitted=None):
    import torch
    if fitted is not None:
        fitted.clear()
    gc.collect()
    torch.cuda.empty_cache()
