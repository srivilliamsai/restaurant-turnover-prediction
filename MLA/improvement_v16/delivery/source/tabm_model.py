"""Fold-isolated TabM epoch selection, full-subset refitting and fixed-layout inference."""
import copy
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import QuantileTransformer
import features


class Preprocessor:
    def fit(self, raw):
        frame = features.frame(raw)
        self.categorical = frame.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        self.numeric = [column for column in frame if column not in self.categorical]
        self.vocabularies = {}
        for column in self.categorical:
            values = self.category_values(frame[column])
            self.vocabularies[column] = {value: index + 2 for index, value in enumerate(sorted(set(values) - {'__MISSING__'}))}
        numeric = frame[self.numeric].apply(pd.to_numeric, errors='raise').to_numpy(float)
        self.medians = np.asarray([np.nanmedian(numeric[:, index]) if np.isfinite(numeric[:, index]).any() else 0. for index in range(numeric.shape[1])])
        extended = self.extend(numeric)
        self.variable = np.ptp(extended, axis=0) > 0
        assert self.variable.any(), 'No nonconstant numerical features for the frozen PLE setting'
        values = extended[:, self.variable].astype(np.float32)
        jitter = np.random.default_rng(0).normal(0., 1e-5, values.shape).astype(np.float32)
        self.quantiles = QuantileTransformer(n_quantiles=max(10, min(len(values) // 30, 1000)),
                                             output_distribution='normal', subsample=10**9,
                                             random_state=0).fit(values + jitter)
        self.cardinalities = [len(self.vocabularies[column]) + 2 for column in self.categorical]
        return self

    def extend(self, values):
        missing = ~np.isfinite(values)
        return np.concatenate([np.where(missing, self.medians, values), missing.astype(float)], axis=1)

    @staticmethod
    def category_values(values):
        values = values.astype('string')
        missing = values.isna() | values.str.strip().str.lower().isin(['missing', '__missing__', '-1', '', 'none', 'nan', 'na'])
        return values.mask(missing, '__MISSING__')

    def transform(self, raw):
        frame = features.frame(raw)
        values = frame[self.numeric].apply(pd.to_numeric, errors='raise').to_numpy(float)
        numeric = self.quantiles.transform(self.extend(values)[:, self.variable]).astype(np.float32)
        categorical = []
        for column in self.categorical:
            values = self.category_values(frame[column])
            categorical.append(np.asarray([0 if value == '__MISSING__' else self.vocabularies[column].get(value, 1) for value in values], dtype=np.int64))
        result = {'numeric': numeric, 'categorical': np.column_stack(categorical)}
        assert np.isfinite(numeric).all()
        return result


def inner_indices(raw, fold=0, scheme='full'):
    indices = np.arange(len(raw))
    seed = features.SPECS['TM32-PLE']['inner_seed'] + fold
    if str(scheme) == 'city':
        groups = features.canonical_city(raw['City']).fillna('__V10_UNKNOWN_CITY__').to_numpy()
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(indices, groups=groups))
        assert set(groups[train]).isdisjoint(groups[valid])
        return train, valid
    return train_test_split(indices, test_size=.15, random_state=seed, shuffle=True)


def seed_everything():
    import random
    import torch
    random.seed(10242)
    np.random.seed(10242)
    torch.manual_seed(10242)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10242)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def create_model(preprocessor, training, device):
    import torch
    import tabm
    import rtdl_num_embeddings
    seed_everything()
    numeric = torch.as_tensor(training['numeric'], dtype=torch.float32)
    bins = rtdl_num_embeddings.compute_bins(numeric, n_bins=32)
    embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=False, version='B')
    model = tabm.TabM.make(n_num_features=numeric.shape[1], cat_cardinalities=preprocessor.cardinalities,
                          d_out=1, num_embeddings=embeddings, k=32, n_blocks=3, d_block=256,
                          dropout=.1, arch_type='tabm').to(device=device, dtype=torch.float32)
    assert model.backbone.k == 32
    return model


def tensor_data(values, device):
    import torch
    return {'numeric': torch.as_tensor(values['numeric'], dtype=torch.float32, device=device),
            'categorical': torch.as_tensor(values['categorical'], dtype=torch.int64, device=device)}


def evaluate_model(model, values, padding, mean, scale, device):
    import torch
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values['numeric']), 500):
            count = min(500, len(values['numeric']) - start)
            block = {}
            for key in ['numeric', 'categorical']:
                block[key] = np.concatenate([values[key][start:start + count], np.repeat(padding[key], 500 - count, axis=0)], axis=0)
            tensors = tensor_data(block, device)
            output = model(tensors['numeric'], tensors['categorical']).squeeze(-1).mean(dim=1)
            outputs.append(output[:count].cpu().numpy().astype(float) * scale + mean)
    return np.maximum(np.concatenate(outputs), 0.) * 1e6


def train_model(raw, target, device, maximum_epochs, validation=None, deadline=None):
    import torch
    from core import rmse
    preprocessor = Preprocessor().fit(raw)
    values = preprocessor.transform(raw)
    model = create_model(preprocessor, values, device)
    tensors = tensor_data(values, device)
    target = np.asarray(target, float) / 1e6
    mean, scale = float(target.mean()), float(target.std())
    assert scale > 0 and np.isfinite(target).all()
    labels = torch.as_tensor((target - mean) / scale, dtype=torch.float32, device=device)
    padding = {key: value[:1].copy() for key, value in values.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    valid_values = preprocessor.transform(validation[0]) if validation is not None else None
    best_score, best_epoch, unimproved = float('inf'), 0, 0
    history = []
    generator = torch.Generator(device='cpu').manual_seed(10242)
    for epoch in range(1, maximum_epochs + 1):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError('TabM fit exceeded its persisted stage budget')
        model.train()
        order = torch.randperm(len(raw), generator=generator)
        for positions in order.split(256):
            positions = positions.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(tensors['numeric'][positions], tensors['categorical'][positions]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(predictions, labels[positions, None].expand_as(predictions))
            assert torch.isfinite(loss), 'Nonfinite member MSE'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if validation is not None:
            prediction = evaluate_model(model, valid_values, padding, mean, scale, device)
            score = rmse(validation[1], prediction)
            history.append({'epoch': epoch, 'inner_rmse_INR': score})
            if score < best_score:
                best_score, best_epoch, unimproved = score, epoch, 0
            else:
                unimproved += 1
            if unimproved >= 20:
                break
    return {'model': model, 'preprocessor': preprocessor, 'padding': padding,
            'target_mean': mean, 'target_scale': scale, 'device': device,
            'best_epoch': best_epoch if validation is not None else maximum_epochs,
            'epochs_run': epoch, 'history': history}


def fit(raw, target, fold=0, scheme='full', device='cuda', deadline=None):
    import gc
    import torch
    started = time.monotonic()
    raw = raw.reset_index(drop=True)
    target = np.asarray(target, float)
    training, validation = inner_indices(raw, fold, scheme)
    selected = train_model(raw.iloc[training], target[training], device, 256,
                           validation=(raw.iloc[validation], target[validation]), deadline=deadline)
    best_epoch, epochs_run, history = selected['best_epoch'], selected['epochs_run'], selected['history']
    del selected
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fitted = train_model(raw, target, device, best_epoch, deadline=deadline)
    fitted['name'] = 'TM32-PLE'
    fitted['target_metadata'] = {'name': 'TM32-PLE', 'training_rows': len(raw), 'units': 'raw INR millions',
        'all_original_training_targets_retained': True, 'members': 32, 'inner_training_rows': len(training),
        'inner_validation_rows': len(validation), 'inner_seed': 9091 + fold,
        'inner_policy': 'grouped-city' if str(scheme) == 'city' else 'random-15-percent',
        'inner_training_indices_sha256': hashlib.sha256(training.tobytes()).hexdigest(),
        'inner_validation_indices_sha256': hashlib.sha256(validation.tobytes()).hexdigest(),
        'selected_epochs': best_epoch, 'inner_epochs_run': epochs_run, 'inner_history': history,
        'outer_validation_used_for_epoch_selection': False, 'full_subset_refitted': True,
        'numeric_input_dimension': int(fitted['preprocessor'].variable.sum()),
        'constant_numeric_inputs_dropped': int((~fitted['preprocessor'].variable).sum()),
        'target_mean_millions': fitted['target_mean'], 'target_scale_millions': fitted['target_scale'],
        'fit_seconds': time.monotonic() - started, 'prediction_chunk': 500, 'seed': 10242}
    return fitted


def predict(fitted, raw):
    values = fitted['preprocessor'].transform(raw)
    prediction = evaluate_model(fitted['model'], values, fitted['padding'], fitted['target_mean'],
                                fitted['target_scale'], fitted['device'])
    assert prediction.shape == (len(raw),) and np.isfinite(prediction).all()
    return prediction
