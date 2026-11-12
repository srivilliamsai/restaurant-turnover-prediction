"""Original bounded V9 TabPFN-v2 and EBM experiments; full original targets."""
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
import legacy_models as legacy

SEED = 10242
CHECKPOINT = {'repo_id': 'Prior-Labs/TabPFN-v2-reg', 'revision': '4972a65a1b30806315c6f92499959ffbfc69a673',
              'filename': 'tabpfn-v2-regressor.ckpt', 'tabpfn_version': '9.2.0'}
WEIGHTS = [.25, .5, .75, 1.]
SPECS = {}
for target in ['raw', 'sqrt']:
    for representation in ['compact', 'v7']:
        SPECS[f'tabpfn_{target}_{representation}'] = dict(family='tabpfn', group='tabpfn', target=target, representation=representation)
    for interactions in [0, 10]:
        SPECS[f'ebm_{target}_i{interactions}'] = dict(family='ebm', group='ebm', target=target, representation='v7', interactions=interactions)


def features(raw, representation='v7'):
    X = legacy.features(raw).reset_index(drop=True)
    if representation == 'compact':
        keep = [c for c in raw if c not in ['Annual Turnover', 'Registration Number', 'Opening Day of Restaurant']]
        keep = ['Endoresed By' if c == 'Endorsed By' else c for c in keep]
        X = X[keep + ['Opening Year', 'Opening Month']]
    assert 'Registration Number' not in X and 'Annual Turnover' not in X
    return X


def categorical_columns(X):
    return X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()


def native_frame(X, state=None):
    """Fold-local ordinal representation with explicitly declared nominal columns."""
    result = X.copy()
    if state is None:
        state = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                 for c in categorical_columns(X)}
    for c, vocabulary in state.items():
        result[c] = X[c].map(vocabulary).astype(float)
    for c in result:
        result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    return result, state


def ebm_parameters(spec, mode='plain'):
    return dict(objective='rmse', interactions=spec['interactions'], learning_rate=.02,
        outer_bags=16 if mode == 'ensemble16' else 8, inner_bags=0, max_rounds=5000,
        max_bins=128, max_interaction_bins=32, min_samples_leaf=20, max_leaves=3,
        validation_size=.15, early_stopping_rounds=100, random_state=SEED, n_jobs=2)


def resolved_defaults():
    base = ExplainableBoostingRegressor(**ebm_parameters(SPECS['ebm_raw_i0'])).get_params()
    return {'interpret-core': importlib.metadata.version('interpret-core'), 'ebm': base,
            'tabpfn': {'model_version': 'v2', 'n_estimators': 8, 'random_state': SEED,
                       'output_type': 'mean', 'inference_precision': 'float32',
                       'fit_mode': 'low_memory', 'device': 'cuda', 'memory_saving_mode': True}}


def ensure_checkpoint():
    import json
    from huggingface_hub import hf_hub_download
    record_path = Path(__file__).resolve().parent / 'tabpfn_checkpoint.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    path = Path(hf_hub_download(repo_id=CHECKPOINT['repo_id'], filename=CHECKPOINT['filename'], revision=CHECKPOINT['revision']))
    value = {**CHECKPOINT, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'attribution': 'TabPFN v2, Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6'}
    if previous:
        assert previous['sha256'] == value['sha256'] and previous['revision'] == value['revision']
    record_path.write_text(json.dumps(value, indent=2) + '\n')
    return value


def execution_contract():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the TabPFN execution contract.')
    checkpoint = ensure_checkpoint()
    return {'torch': importlib.metadata.version('torch'), 'tabpfn': importlib.metadata.version('tabpfn'),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'precision': 'float32',
        'checkpoint': {k: checkpoint[k] for k in ['repo_id', 'revision', 'filename', 'sha256']}}


def fit_model(raw, y, spec, mode='plain', progress=None):
    assert mode in ['plain', 'ensemble16']
    X = features(raw, spec['representation'])
    target = np.asarray(y, float) / 1e6
    transformed = np.sqrt(target) if spec['target'] == 'sqrt' else target
    fitted = {'spec': copy.deepcopy(spec), 'mode': mode, 'vocabulary': None, 'columns': X.columns.tolist()}
    cats = categorical_columns(X)
    if spec['family'] == 'ebm':
        types = ['nominal' if c in cats else 'continuous' for c in X]
        model = ExplainableBoostingRegressor(feature_types=types, **ebm_parameters(spec, mode))
    else:
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        if not torch.cuda.is_available():
            raise RuntimeError('TabPFN-v2 requires an available free CUDA GPU in this workflow; CPU rows are not silently subsampled.')
        torch.use_deterministic_algorithms(True)
        X, fitted['vocabulary'] = native_frame(X)
        assert importlib.metadata.version('tabpfn') == CHECKPOINT['tabpfn_version']
        record = ensure_checkpoint()
        path = Path(record['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,
            model_path=str(path), n_estimators=16 if mode == 'ensemble16' else 8,
            random_state=SEED, device='cuda', fit_mode='low_memory', memory_saving_mode=True,
            inference_precision=torch.float32, categorical_features_indices=[X.columns.get_loc(c) for c in cats])
    model.fit(X, transformed)
    fitted['model'] = model
    if progress:
        progress(f'{spec["family"]} fit complete: {len(raw)} rows, {len(X.columns)} features, {mode}')
    return fitted


def predict_model(fitted, raw):
    X = features(raw, fitted['spec']['representation'])[fitted['columns']]
    if fitted['vocabulary'] is not None:
        X, _ = native_frame(X, fitted['vocabulary'])
    if fitted['spec']['family'] == 'tabpfn':
        z = fitted['model'].predict(X, output_type='mean')
    else:
        z = fitted['model'].predict(X)
    z = np.asarray(z, float)
    p = np.maximum(z, 0) ** 2 if fitted['spec']['target'] == 'sqrt' else np.maximum(z, 0)
    p *= 1e6
    assert p.shape == (len(raw),) and np.isfinite(p).all()
    return p


def fit_predict(raw, y, valid, spec, mode='plain', progress=None):
    return predict_model(fit_model(raw, y, spec, mode, progress), valid)
