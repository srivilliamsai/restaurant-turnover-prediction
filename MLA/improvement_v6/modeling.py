"""Six predeclared experiments. Targets are in millions internally, INR at I/O."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, train_test_split
import reference as ref

REQUIRED = ['Train_dataset.csv', 'Test_dataset.csv', 'Data_Dictionary.csv', 'Sample_Solution.csv']
SEEDS = [10242, 20242]
SPECS = {
    'native_l2_7': {'family': 'native_l2', 'kind': 'lgb', 'objective': 'regression', 'leaves': 7},
    'native_l2_15': {'family': 'native_l2', 'kind': 'lgb', 'objective': 'regression', 'leaves': 15},
    'native_huber_7': {'family': 'native_huber', 'kind': 'lgb', 'objective': 'huber', 'leaves': 7},
    'native_huber_15': {'family': 'native_huber', 'kind': 'lgb', 'objective': 'huber', 'leaves': 15},
    'sqrt_crossfit': {'family': 'sqrt_crossfit', 'kind': 'sqrt'},
    'domain_huber': {'family': 'domain_huber', 'kind': 'weighted'},
}


def load_data(data_dir):
    directory = Path(data_dir)
    for name in REQUIRED:
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Missing {directory / name}; use --data-dir')
    train = pd.read_csv(directory / REQUIRED[0]).rename(columns={'Endorsed By': 'Endoresed By'})
    test = pd.read_csv(directory / REQUIRED[1])
    sample = pd.read_csv(directory / REQUIRED[3])
    pd.read_csv(directory / REQUIRED[2])
    assert train.drop(columns='Annual Turnover').columns.tolist() == test.columns.tolist()
    assert len(test) == 500 and train['Registration Number'].is_unique and test['Registration Number'].is_unique
    assert set(train['Registration Number']).isdisjoint(test['Registration Number'])
    assert sample.columns.tolist() == ['Registration Number', 'Annual Turnover']
    assert sample['Registration Number'].equals(test['Registration Number'])
    y = train['Annual Turnover'].to_numpy(float)
    assert np.isfinite(y).all() and (y > 0).all()
    return train, test, y


def feature_frame(raw):
    # Same deterministic cleaned representation already screened in v5.
    return ref.v5_features(raw, 'clean')


def splits(train, scheme):
    if str(scheme) == 'city':
        groups = feature_frame(train)['City']
        result = list(GroupKFold(5).split(train, groups=groups))
        for a, b in result:
            assert set(groups.iloc[a]).isdisjoint(groups.iloc[b])
    else:
        result = list(KFold(5, shuffle=True, random_state=int(scheme)).split(train))
    counts = np.zeros(len(train), int)
    for a, b in result:
        assert not set(a).intersection(b)
        counts[b] += 1
    assert (counts == 1).all()
    return result


def native_frame(X, categories=None):
    result = X.copy()
    if categories is None:
        categories = {c: sorted(X[c].unique().tolist())
                      for c in X.select_dtypes(include=['object', 'string']).columns}
    for c, values in categories.items():
        result[c] = pd.Categorical(result[c], categories=values, ordered=False)
    return result, categories


def lgb_parameters(spec, seed, trees):
    return dict(n_estimators=trees, learning_rate=.03, num_leaves=spec['leaves'],
                min_child_samples=50, reg_lambda=20, reg_alpha=.1,
                colsample_bytree=.8, subsample=.8, subsample_freq=1,
                cat_smooth=20, min_data_per_group=20, objective=spec['objective'],
                alpha=.9, metric='rmse', random_state=seed, n_jobs=2, verbosity=-1,
                deterministic=True, force_col_wise=True)


def cat_parameters(seed, depth=5):
    return dict(iterations=850, learning_rate=.04, depth=depth, l2_leaf_reg=10,
                bootstrap_type='Bernoulli', subsample=.7, random_seed=seed,
                thread_count=2, verbose=False, allow_writing_files=False)


def domain_weights(X, test_X, seed):
    """Training-row probabilities are cross-fitted; outer validation is never passed."""
    both = pd.concat([X, test_X], ignore_index=True)
    domain = np.r_[np.zeros(len(X), int), np.ones(len(test_X), int)]
    probabilities = np.zeros(len(both))
    categorical = both.select_dtypes(include=['object', 'string']).columns.tolist()
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(both, domain)):
        model = CatBoostClassifier(iterations=300, depth=3, learning_rate=.035,
                                   l2_leaf_reg=20, loss_function='Logloss',
                                   random_seed=seed + fold, thread_count=2,
                                   verbose=False, allow_writing_files=False)
        model.fit(both.iloc[a], domain[a], cat_features=categorical)
        probabilities[b] = model.predict_proba(both.iloc[b])[:, 1]
    p = np.clip(probabilities[:len(X)], 1e-6, 1 - 1e-6)
    ratio = p / (1 - p) * len(X) / len(test_X)
    clipped = np.clip(ratio, .5, 2.)
    weights = clipped / clipped.mean()
    return weights, {'ratio_clip': [.5, 2.], 'normalized_min': float(weights.min()),
                     'normalized_max': float(weights.max()),
                     'effective_sample_size': float(weights.sum() ** 2 / (weights ** 2).sum()),
                     'mean_weight': float(weights.mean()), 'domain_seed': seed}


def fit_component(raw, y, test_covariates, spec, seed):
    X = feature_frame(raw)
    target = np.asarray(y, float) / 1e6
    categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    diagnostics = {'seed': seed, 'train_rows': len(X)}
    if spec['kind'] == 'lgb':
        a, b = train_test_split(np.arange(len(X)), test_size=.2, random_state=seed + 301)
        inner_X, inner_categories = native_frame(X.iloc[a])
        inner_valid, _ = native_frame(X.iloc[b], inner_categories)
        pilot = LGBMRegressor(**lgb_parameters(spec, seed, 2000))
        pilot.fit(inner_X, target[a], eval_set=[(inner_valid, target[b])],
                  eval_metric='rmse', callbacks=[early_stopping(100, first_metric_only=True, verbose=False)])
        trees = max(1, pilot.best_iteration_)
        native_X, categories = native_frame(X)
        model = LGBMRegressor(**lgb_parameters(spec, seed, trees))
        model.fit(native_X, target)
        diagnostics['trees'] = trees
        return {'model': model, 'categories': categories, 'kind': 'lgb', 'diagnostics': diagnostics}
    if spec['kind'] == 'sqrt':
        transformed = np.sqrt(target)
        inner_prediction = np.zeros(len(X))
        for fold, (a, b) in enumerate(KFold(3, shuffle=True, random_state=seed + 317).split(X)):
            model = CatBoostRegressor(**cat_parameters(seed + 100 + fold, 6), loss_function='RMSE')
            model.fit(X.iloc[a], transformed[a], cat_features=categorical)
            inner_prediction[b] = np.maximum(model.predict(X.iloc[b]), 0) ** 2
        correction = float(np.mean(target - inner_prediction))
        model = CatBoostRegressor(**cat_parameters(seed, 6), loss_function='RMSE')
        model.fit(X, transformed, cat_features=categorical)
        diagnostics.update(correction_millions=correction, correction_source='three-fold inner OOF')
        return {'model': model, 'correction': correction, 'kind': 'sqrt', 'diagnostics': diagnostics}
    assert spec['kind'] == 'weighted'
    weights, info = domain_weights(X, feature_frame(test_covariates), seed + 911)
    model = CatBoostRegressor(**cat_parameters(seed), loss_function='Huber:delta=40')
    model.fit(X, target, cat_features=categorical, sample_weight=weights)
    diagnostics.update(info)
    return {'model': model, 'kind': 'weighted', 'diagnostics': diagnostics}


def predict_component(fitted, raw):
    X = feature_frame(raw)
    if fitted['kind'] == 'lgb':
        X, _ = native_frame(X, fitted['categories'])
    result = fitted['model'].predict(X)
    if fitted['kind'] == 'sqrt':
        result = np.maximum(result, 0) ** 2 + fitted['correction']
    return np.maximum(result, 0) * 1e6


def component_oof(train, test, y, spec, scheme, progress=None):
    individual = np.zeros((len(y), len(SEEDS)))
    diagnostics = []
    for fold, (a, b) in enumerate(splits(train, scheme)):
        for j, seed in enumerate(SEEDS):
            fitted = fit_component(train.iloc[a], y[a], test, spec, seed + fold)
            individual[b, j] = predict_component(fitted, train.iloc[b])
            diagnostics.append({'fold': fold, **fitted['diagnostics']})
        if progress:
            progress(f'fold {fold + 1}/5 complete')
    return individual.mean(axis=1), individual, diagnostics


def fit_final(train, test, y, choice, contract):
    fitted = {'choice': choice, 'contract': contract, 'new': [], 'baseline': None}
    prediction = np.zeros(len(test))
    if choice['weight'] < 1:
        _, baseline, legacy = ref.full_fit(train, test, y, contract)
        fitted['baseline'] = legacy
        prediction += (1 - choice['weight']) * baseline
    if choice['weight']:
        for seed in SEEDS:
            model = fit_component(train, y, test, SPECS[choice['name']], seed)
            fitted['new'].append(model)
            prediction += choice['weight'] * predict_component(model, test) / len(SEEDS)
    return prediction, fitted


def predict_final(fitted, raw):
    choice = fitted['choice']
    p = np.zeros(len(raw))
    if choice['weight'] < 1:
        p += (1 - choice['weight']) * ref.predict_final(raw, fitted['baseline'], fitted['contract'])
    if choice['weight']:
        p += choice['weight'] * np.mean([predict_component(m, raw) for m in fitted['new']], axis=0)
    return p
