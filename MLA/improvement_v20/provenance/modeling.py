"""Fixed 16-estimator raw-target TabPFN v2 for all six V10 settings."""
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import features
import frozen_v9_modeling as reference

SEED = 10242
WEIGHTS = [.25, .5, .75, 1.]


def execution_contract():
    contract = reference.execution_contract()
    import json
    expected = json.loads((Path(__file__).resolve().parent/'reference_manifest.json').read_text())['execution_contract']
    assert contract == expected, f'Pinned execution contract differs: {contract}'
    return contract


def fit(raw, y, name, progress=print):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    contract = execution_contract()
    torch.use_deterministic_algorithms(True)
    builder = features.FeatureBuilder(name).fit(raw)
    frame = builder.transform(raw)
    cats = frame.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    X, vocabulary = features.native_frame(frame)
    checkpoint = reference.ensure_checkpoint()
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,
        model_path=checkpoint['path'], n_estimators=16, random_state=SEED, device='cuda',
        fit_mode='low_memory', memory_saving_mode=True, inference_precision=torch.float32,
        categorical_features_indices=[X.columns.get_loc(c) for c in cats])
    target = np.asarray(y, float)/1e6
    assert len(target) == len(raw) and np.isfinite(target).all() and (target > 0).all()
    model.fit(X, target)
    progress(f'{name}: fitted {len(raw)} original targets, {len(X.columns)} features, 16 estimators')
    return {'name': name, 'builder': builder, 'vocabulary': vocabulary, 'model': model,
            'execution_contract': contract, 'training_rows': len(raw), 'padding_row': X.iloc[:1].copy()}


def predict(fitted, raw):
    frame = fitted['builder'].transform(raw)
    X, _ = features.native_frame(frame, fitted['vocabulary'])
    # Keep GPU matrix shapes constant across full, split and individual calls.
    # Padding consists only of one frozen training-feature row; predictions for
    # padding rows are discarded. Exactly 500 test rows use the original V9
    # prediction shape without padding or any output rounding.
    chunks=[]
    for start in range(0,len(X),500):
        block=X.iloc[start:start+500].reset_index(drop=True)
        size=len(block)
        if size<500:
            padding=pd.concat([fitted['padding_row']]*(500-size),ignore_index=True)
            block=pd.concat([block,padding],ignore_index=True)
        values=np.asarray(fitted['model'].predict(block, output_type='mean'),float)
        chunks.append(values[:size])
    p = np.maximum(np.concatenate(chunks),0)*1e6
    assert p.shape == (len(raw),) and np.isfinite(p).all()
    return p


def fit_predict(raw, y, valid, name, progress=print):
    return predict(fit(raw, y, name, progress), valid)
