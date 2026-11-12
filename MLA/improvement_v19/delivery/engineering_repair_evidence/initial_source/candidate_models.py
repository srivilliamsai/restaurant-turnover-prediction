"""The four fixed V19 candidates, running only in their isolated environment."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from new_models import Codec

ROOT = Path(__file__).resolve().parent
SEED = 10242


def resources():
    import psutil
    return {'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'host_rss_bytes': psutil.Process().memory_info().rss}


def runtime():
    names = ['torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'causilo',
             'tabdpt', 'structured-data-models', 'huggingface-hub', 'safetensors', 'pyarrow']
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'dependencies': packages, 'gpu': torch.cuda.get_device_name(0),
            'cuda': torch.version.cuda, 'python': __import__('sys').version,
            'total_gpu_bytes': torch.cuda.get_device_properties(0).total_memory}


def checked_checkpoint(name):
    from huggingface_hub import hf_hub_download
    record = json.loads((ROOT / 'checkpoint_registry.json').read_text())['models'][name]
    path = Path(hf_hub_download(repo_id=record['model_repository'], filename=record['filename'],
                              revision=record['model_revision'], token=False))
    assert path.stat().st_size == record['advertised_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['advertised_sha256']
    return path, record


def source_check(name):
    registry = json.loads((ROOT / 'source_registry.json').read_text())['model_modules']
    key = 'KS8' if name == 'KM8' else name
    actual = {}
    for module, expected in registry[key].items():
        path = Path(inspect.getfile(importlib.import_module(module)))
        actual[module] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual[module] == expected, 'Published source mismatch: ' + module
    return actual


def fit(raw, target, name):
    assert name in ['C8', 'KS8', 'D8', 'KM8']
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    source_hashes = source_check(name)
    path, checkpoint = checked_checkpoint(name)
    codec = Codec().fit(raw)
    y = np.asarray(target, float) / 1e6
    assert y.shape == (len(raw),) and np.isfinite(y).all()
    if name == 'C8':
        from packaging.version import Version
        from causilo import CausiloRegressor
        import causilo.checkpoints as loader
        assert Version(torch.__version__.split('+')[0]) >= Version('2.13')
        assert loader.RELEASE_COMMIT == checkpoint['model_revision']
        model = CausiloRegressor(n_estimators=8, random_state=SEED, device='cuda',
                                use_kv_cache=False, retain_preprocessing=True)
        model.fit(codec.typed(raw), y)
    elif name in ['KS8', 'KM8']:
        import sdm
        model = sdm.models.KumoTabular(task='regression', size='small' if name == 'KS8' else 'medium',
                                      pretrained=False, device='cpu')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        model.models[sdm.Task.regression].load_state_dict(weights, strict=True)
        del weights
        model = model.to('cuda').eval()
        frame = codec.typed(raw)
        stypes = {c: 'categorical' if c in codec.vocabulary else 'numerical' for c in frame}
        x = sdm.TableTensor.from_pandas(frame, stypes=stypes, device='cuda')
        y_table = sdm.TableTensor.from_pandas(pd.DataFrame({'target': y}),
                                             stypes={'target': 'numerical'}, device='cuda')
        with torch.inference_mode():
            model.fit(x=x, y=y_table, num_estimators=8, estimator_batch_size=1,
                      generator=torch.Generator(device='cuda').manual_seed(SEED))
    else:
        from tabdpt import TabDPTRegressor
        model = TabDPTRegressor(device='cuda', use_flash=False, compile=False,
                               model_weight_path=str(path), verbose=False)
        model.fit(codec.transform(raw), y)
        assert len(model.X_train) == len(raw)
    metadata = {'name': name, 'training_rows': len(raw), 'units': 'raw INR millions',
                'model_class': type(model).__module__ + '.' + type(model).__name__,
                'checkpoint_sha256': checkpoint['advertised_sha256'],
                'checkpoint_revision': checkpoint['model_revision'], 'actual_source_hashes': source_hashes,
                'runtime': runtime(), 'input_features': 40, 'categorical_columns': list(codec.vocabulary),
                'all_outer_training_rows': True, 'seed': SEED, **resources()}
    assert len(codec.vocabulary) == 8
    return {'name': name, 'model': model, 'codec': codec, 'metadata': metadata}


def predict(fitted, raw):
    name, model, codec = fitted['name'], fitted['model'], fitted['codec']
    if name == 'C8':
        prediction = np.asarray(model.predict(codec.typed(raw), output_type='mean'), float)
    elif name in ['KS8', 'KM8']:
        import sdm
        frame = codec.typed(raw)
        stypes = {c: 'categorical' if c in codec.vocabulary else 'numerical' for c in frame}
        tensor = sdm.TableTensor.from_pandas(frame, stypes=stypes, device='cuda')
        with torch.inference_mode():
            result = model.predict(tensor)
        names = list(result.numerical_columns)
        assert names == [f'q{i:03d}' for i in range(1, 1000)], names[:5]
        values = result.numerical.detach().cpu().numpy().astype(np.float64)
        assert values.shape == (len(raw), 999), values.shape
        prediction = values.mean(axis=1)
    else:
        prediction = np.asarray(model.predict(codec.transform(raw), n_ensembles=8, seed=SEED,
                                              context_size=None, batch_size=256, output_type='mean'), float)
    assert prediction.shape == (len(raw),) and np.isfinite(prediction).all()
    return np.maximum(prediction, 0.) * 1e6
