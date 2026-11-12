"""Fit and run the two frozen TabPFN components of the final ensemble."""

import gc
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

import features
from check_submission import ROOT, sha256

SEED = 10242
BLOCK_SIZE = 500


def local_checkpoint(record):
    from huggingface_hub import hf_hub_download

    destination = ROOT / 'artifacts/model_weights' / record['filename']
    if not destination.is_file():
        source = hf_hub_download(
            repo_id=record['repo_id'], filename=record['filename'],
            revision=record['revision'], token=False,
        )
        destination.parent.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=destination.parent, delete_on_close=False) as pending:
            pending_path = Path(pending.name)
            shutil.copyfile(source, pending_path)
            if sha256(pending_path) != record['sha256']:
                raise ValueError(f"Downloaded checkpoint checksum mismatch: {record['filename']}")
            pending_path.replace(destination)
    elif sha256(destination) != record['sha256']:
        raise ValueError(f'Cached checkpoint checksum mismatch: {destination.name}')
    return destination


def fit(raw, y, name, checkpoint):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.use_deterministic_algorithms(True)
    X, vocabulary = features.native_frame(features.frame(raw))
    categorical_indices = [X.columns.get_loc(column) for column in vocabulary]
    base = name == 'BASE'
    version = ModelVersion.V2 if base else ModelVersion.V3_5
    model = TabPFNRegressor.create_default_for_version(
        version, model_path=str(local_checkpoint(checkpoint)),
        n_estimators=16 if base else 8, random_state=SEED, device='cuda',
        fit_mode='low_memory', memory_saving_mode=True, inference_precision=torch.float32,
        categorical_features_indices=categorical_indices,
    )
    target = np.asarray(y, float) / 1e6
    assert len(target) == len(raw) and np.isfinite(target).all() and (target > 0).all()
    model.fit(X, target)
    if not base:
        classes = [type(item).__name__.lower() for item in [*model.models_, *model.configs_]]
        assert any('v3_5' in name or 'v3p5' in name or '3p5' in name for name in classes), classes
    padding_index = 0 if base else int(np.argmax(X.notna().sum(axis=1).to_numpy()))
    return {'model': model, 'vocabulary': vocabulary, 'padding_row': X.iloc[[padding_index]].copy()}


def predict(fitted, raw):
    X, _ = features.native_frame(features.frame(raw), fitted['vocabulary'])
    parts = []
    for start in range(0, len(X), BLOCK_SIZE):
        block = X.iloc[start:start + BLOCK_SIZE].reset_index(drop=True)
        size = len(block)
        if size < BLOCK_SIZE:
            padding = [fitted['padding_row']] * (BLOCK_SIZE - size)
            block = pd.concat([block, *padding], ignore_index=True)
        predictions = np.asarray(fitted['model'].predict(block, output_type='mean'), float)
        assert predictions.shape == (BLOCK_SIZE,) and np.isfinite(predictions).all()
        parts.append(predictions[:size])
    result = np.maximum(np.concatenate(parts), 0.) * 1e6
    assert result.shape == (len(raw),) and np.isfinite(result).all()
    return result


def release(fitted):
    import torch

    if fitted is not None:
        fitted.clear()
    gc.collect()
    torch.cuda.empty_cache()
