"""Three target treatments of the frozen TabPFN-v2 sixteen-estimator model."""
import gc
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import ROOT, PINNED, sha, save


def runtime():
    import torch
    actual = {key: importlib.metadata.version(key) for key in PINNED}
    assert actual == PINNED, actual
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4'
    return {'dependencies': actual, 'gpu': 'Tesla T4', 'cuda': torch.version.cuda,
            'precision': 'float32', 'prediction_chunk': 500}


def checkpoint(record):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=record['repo_id'], filename=record['filename'],
                               revision=record['revision'], token=False))
    assert sha(path) == record['sha256']
    return str(path)


def make_model(config, categorical):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=checkpoint(config['checkpoint']),
        n_estimators=16, random_state=10242, device='cuda',
        fit_mode='low_memory', memory_saving_mode=True,
        inference_precision=torch.float32, categorical_features_indices=list(categorical))


def serial(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    return str(value)


def resolve():
    config = {'runtime': runtime(),
              'checkpoint': json.loads((ROOT / 'reference_manifest.json').read_text())['execution_contract']['checkpoint'],
              'settings': features.SPECS, 'target_units': 'INR millions before target treatment',
              'adapter': '500-row blocks; pad with first encoded training row; discard padded outputs',
              'output_floor': 0., 'SQ2': 'raw-space criterion.mean_of_square(logits) * 1e6'}
    model = make_model(config, [])
    params = serial(model.get_params(deep=False))
    params['model_path'] = 'checkpoint_sha256:' + config['checkpoint']['sha256']
    config['parameters'] = params
    del model
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    import inspect
    assert callable(FullSupportBarDistribution.mean_of_square)
    config['distribution_source_sha256'] = sha(inspect.getfile(FullSupportBarDistribution))
    save(ROOT / 'resolved_models.json', config)
    return config


def fit(raw, y, name):
    import torch
    config = json.loads((ROOT / 'resolved_models.json').read_text())
    assert runtime() == config['runtime']
    assert name == 'BASE' or name in features.SPECS
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(10242)
    torch.cuda.manual_seed_all(10242)
    X, vocabulary = features.native_frame(features.frame(raw))
    target, metadata = features.training_target(y, name)
    model = make_model(config, [X.columns.get_loc(c) for c in vocabulary])
    model.fit(X, target)
    print(f'{name}: fitted all {len(raw)} rows, 40 predictors; {metadata}', flush=True)
    return {'name': name, 'model': model, 'vocabulary': vocabulary,
            'columns': list(X.columns), 'padding_row': X.iloc[[0]].copy(),
            'target_metadata': metadata}


def decode_full(full):
    """The criterion is in the external square-root target units, not z units."""
    criterion, logits = full['criterion'], full['logits']
    moment = criterion.mean_of_square(logits).detach().cpu().numpy().astype(float)
    mean = criterion.mean(logits).detach().cpu().numpy().astype(float)
    assert np.isfinite(moment).all() and np.isfinite(mean).all()
    assert np.all(moment + 1e-8 >= mean ** 2), 'Invalid predictive second moment'
    return {'prediction': np.maximum(moment, 0.) * 1e6,
            'naive_squared_mean': np.maximum(mean, 0.) ** 2 * 1e6}


def predict_outputs(fitted, raw):
    if len(raw) == 0:
        return {'prediction': np.empty(0), **({'naive_squared_mean': np.empty(0)} if fitted['name'] == 'SQ2' else {})}
    X, _ = features.native_frame(features.frame(raw)[fitted['columns']], fitted['vocabulary'])
    results = {}
    for start in range(0, len(X), 500):
        block = X.iloc[start:start + 500].reset_index(drop=True)
        count = len(block)
        if count < 500:
            block = pd.concat([block, pd.concat([fitted['padding_row']] * (500 - count), ignore_index=True)], ignore_index=True)
        if fitted['name'] == 'SQ2':
            values = decode_full(fitted['model'].predict(block, output_type='full'))
        else:
            values = {'prediction': np.maximum(np.asarray(fitted['model'].predict(block, output_type='mean'), float), 0.) * 1e6}
        for key, value in values.items():
            results.setdefault(key, []).append(value[:count])
    result = {key: np.concatenate(value) for key, value in results.items()}
    assert all(v.shape == (len(raw),) and np.isfinite(v).all() and (v >= 0).all() for v in result.values())
    return result


def predict(fitted, raw):
    return predict_outputs(fitted, raw)['prediction']


def combine(baseline, component, weight):
    assert weight in features.WEIGHTS
    return (1 - weight) * np.asarray(baseline) + weight * np.asarray(component)


def release(fitted=None):
    import torch
    if fitted is not None:
        fitted.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
