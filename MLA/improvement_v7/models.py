"""V7: turnover bands, partially pooled tier specialists and nested residual models."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import reference as ref

REQUIRED = ['Train_dataset.csv', 'Test_dataset.csv', 'Data_Dictionary.csv', 'Sample_Solution.csv']
SEEDS = [10242, 20242]
BANDS = [20., 40., 80.]
SPECS = {}
for depth in [3, 5]:
    for calibrate in [False, True]:
        SPECS[f'bands_d{depth}_cal{int(calibrate)}'] = {
            'family': 'bands', 'depth': depth, 'calibrate': calibrate}
for depth in [4, 6]:
    for shrink in [100, 300]:
        SPECS[f'tiers_d{depth}_shrink{shrink}'] = {
            'family': 'tiers', 'depth': depth, 'shrink': shrink}
for alpha in [100, 1000]:
    SPECS[f'residual_ridge{alpha}'] = {'family': 'residual', 'kind': 'ridge', 'alpha': alpha}
for loss in ['RMSE', 'Huber:delta=20']:
    SPECS['residual_tree_' + ('l2' if loss == 'RMSE' else 'h20')] = {
        'family': 'residual', 'kind': 'cat', 'loss': loss}


def load_data(directory):
    directory = Path(directory)
    assert all((directory / p).is_file() for p in REQUIRED), f'Four official inputs required in {directory}'
    train = pd.read_csv(directory / REQUIRED[0]).rename(columns={'Endorsed By': 'Endoresed By'})
    test = pd.read_csv(directory / REQUIRED[1])
    sample = pd.read_csv(directory / REQUIRED[3])
    pd.read_csv(directory / REQUIRED[2])
    assert len(test) == 500 and train['Registration Number'].is_unique and test['Registration Number'].is_unique
    assert train.drop(columns='Annual Turnover').columns.tolist() == test.columns.tolist()
    assert set(train['Registration Number']).isdisjoint(test['Registration Number'])
    assert sample.columns.tolist() == ['Registration Number', 'Annual Turnover']
    assert sample['Registration Number'].equals(test['Registration Number'])
    y = train['Annual Turnover'].to_numpy(float)
    assert np.isfinite(y).all() and (y > 0).all()
    return train, test, y


def features(raw):
    return ref.v5_features(raw, 'clean')


def split_indices(raw, scheme):
    scheme = str(scheme)
    if scheme == 'city':
        groups = features(raw)['City']
        result = list(GroupKFold(5).split(raw, groups=groups))
        for a, b in result:
            assert set(groups.iloc[a]).isdisjoint(groups.iloc[b])
    elif scheme.startswith('strat:'):
        seed = int(scheme.split(':')[1])
        bins = np.digitize(raw['Annual Turnover'].to_numpy(float) / 1e6, BANDS, right=True)
        result = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(raw, bins))
    elif scheme.startswith('inner:'):
        seed = int(scheme.split(':')[1])
        result = list(KFold(3, shuffle=True, random_state=seed).split(raw))
    else:
        result = list(KFold(5, shuffle=True, random_state=int(scheme)).split(raw))
    counts = np.zeros(len(raw), int)
    for a, b in result:
        assert not set(a).intersection(b)
        counts[b] += 1
    assert (counts == 1).all()
    return result


def cat_params(seed, depth, trees, l2=20):
    return dict(iterations=trees, depth=depth, learning_rate=.04, l2_leaf_reg=l2,
                bootstrap_type='Bernoulli', subsample=.7, random_seed=seed,
                thread_count=2, verbose=False, allow_writing_files=False)


def residual_features(raw, baseline):
    X = features(raw)
    keep = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient',
            'Resturant Tier', 'Hygiene Rating', 'Restaurant Zomato Rating',
            'Restaurant City Tier', 'Opening Year', 'Overall Restaurant Rating',
            'Cuisine', 'Restaurant Type', 'Restaurant Location']
    Z = X[keep].copy()
    Z['Base prediction'] = np.asarray(baseline) / 1e6
    Z['Social mean'] = Z[['Facebook Popularity Quotient', 'Instagram Popularity Quotient']].mean(axis=1)
    Z['Social hygiene'] = Z['Social mean'] * Z['Hygiene Rating'] / 100
    Z['Tier1 base'] = Z['Base prediction'] * (Z['Resturant Tier'] == 1)
    Z['Hygiene base'] = Z['Base prediction'] * (Z['Hygiene Rating'] - 5) / 5
    return Z


def fit_component(raw, y, spec, seed, inner_baseline=None):
    X = features(raw)
    target = np.asarray(y, float) / 1e6
    cats = X.select_dtypes(include=['object', 'string']).columns.tolist()
    info = {'seed': seed, 'train_rows': len(raw), 'family': spec['family']}
    if spec['family'] == 'bands':
        labels = np.digitize(target, BANDS, right=True)
        counts = np.bincount(labels, minlength=4)
        assert (counts >= 3).all()
        centers = np.array([target[labels == k].mean() for k in range(4)])
        temperature = 1.
        if spec['calibrate']:
            prob = np.zeros((len(X), 4))
            for f, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed + 411).split(X, labels)):
                model = CatBoostClassifier(**cat_params(seed + 100 + f, spec['depth'], 800), loss_function='MultiClass')
                model.fit(X.iloc[a], labels[a], cat_features=cats)
                assert np.array_equal(model.classes_, np.arange(4))
                prob[b] = model.predict_proba(X.iloc[b])
            log_prob = np.log(np.clip(prob, 1e-12, 1))
            temperatures = [.7, 1., 1.4, 2.]
            losses = [-np.mean(np.log(np.clip(softmax(log_prob / t, axis=1)[np.arange(len(X)), labels], 1e-12, 1))) for t in temperatures]
            temperature = temperatures[int(np.argmin(losses))]
            info.update(temperature_candidates=temperatures, inner_oof_logloss=losses)
        model = CatBoostClassifier(**cat_params(seed, spec['depth'], 800), loss_function='MultiClass')
        model.fit(X, labels, cat_features=cats)
        assert np.array_equal(model.classes_, np.arange(4))
        info.update(band_counts=counts.tolist(), band_centers_millions=centers.tolist(), temperature=temperature)
        return {'family': 'bands', 'model': model, 'centers': centers, 'temperature': temperature, 'info': info}
    if spec['family'] == 'tiers':
        global_model = CatBoostRegressor(**cat_params(seed, 5, 850, 10), loss_function='Huber:delta=40')
        global_model.fit(X, target, cat_features=cats)
        local = {}
        for tier in [1., 2.]:
            mask = X['Resturant Tier'].to_numpy() == tier
            n = int(mask.sum())
            if n < 120:
                continue
            model = CatBoostRegressor(**cat_params(seed + int(tier) * 71, spec['depth'], 650, 30), loss_function='Huber:delta=40')
            model.fit(X.loc[mask], target[mask], cat_features=cats)
            local[tier] = {'model': model, 'weight': n / (n + spec['shrink']), 'rows': n}
        info['local_tiers'] = {str(t): {'weight': value['weight'], 'rows': value['rows']} for t, value in local.items()}
        return {'family': 'tiers', 'global': global_model, 'local': local, 'info': info}
    assert spec['family'] == 'residual' and inner_baseline is not None
    inner_baseline = np.asarray(inner_baseline, float)
    assert inner_baseline.shape == target.shape and np.isfinite(inner_baseline).all()
    Z = residual_features(raw, inner_baseline)
    residual = target - inner_baseline / 1e6
    if spec['kind'] == 'ridge':
        categorical = Z.select_dtypes(include=['object', 'string']).columns.tolist()
        numeric = [c for c in Z if c not in categorical]
        pre = ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False), categorical)])
        model = make_pipeline(pre, Ridge(alpha=spec['alpha']))
        model.fit(Z, residual)
    else:
        model = CatBoostRegressor(**cat_params(seed, 2, 300, 50), loss_function=spec['loss'])
        model.fit(Z, residual, cat_features=Z.select_dtypes(include=['object', 'string']).columns.tolist())
    info.update(inner_baseline='three-fold v5 OOF within this outer training fold', mean_inner_residual_millions=float(residual.mean()))
    return {'family': 'residual', 'model': model, 'info': info}


def predict_component(fitted, raw, baseline=None):
    if fitted['family'] == 'bands':
        p = fitted['model'].predict_proba(features(raw))
        p = softmax(np.log(np.clip(p, 1e-12, 1)) / fitted['temperature'], axis=1)
        prediction = p @ fitted['centers'] * 1e6
    elif fitted['family'] == 'tiers':
        X = features(raw)
        prediction = fitted['global'].predict(X)
        for tier, local in fitted['local'].items():
            mask = X['Resturant Tier'].to_numpy() == tier
            if mask.any():
                prediction[mask] = (1 - local['weight']) * prediction[mask] + local['weight'] * local['model'].predict(X.loc[mask])
        prediction *= 1e6
    else:
        assert baseline is not None and len(baseline) == len(raw)
        prediction = np.asarray(baseline) + fitted['model'].predict(residual_features(raw, baseline)) * 1e6
    prediction = np.maximum(prediction, 0)
    assert prediction.shape == (len(raw),) and np.isfinite(prediction).all()
    return prediction


def evaluate_component(raw, y, spec, scheme, baseline, inner_provider=None, progress=None):
    individual = np.zeros((len(y), len(SEEDS)))
    diagnostics = []
    for fold, (a, b) in enumerate(split_indices(raw, scheme)):
        inner = inner_provider(a, fold) if spec['family'] == 'residual' else None
        for j, seed in enumerate(SEEDS):
            fitted = fit_component(raw.iloc[a], y[a], spec, seed + fold, inner)
            individual[b, j] = predict_component(fitted, raw.iloc[b], baseline[b])
            diagnostics.append({'fold': fold, **fitted['info']})
        if progress:
            progress(f'fold {fold + 1}/5 complete')
    return individual.mean(axis=1), individual, diagnostics


def v5_fold_prediction(train, y, valid, contract, fold):
    """Exact legacy outer-fold seed schedule; no validation labels enter fitting."""
    n = len(train)
    a = np.arange(n)
    b = np.arange(n, n + len(valid))
    joined_y = np.r_[np.asarray(y), np.full(len(valid), np.nan)]
    prior = np.zeros(len(valid))
    for name, weight in contract['v2_weights'].items():
        spec = contract['v2_configs'][name]
        X = pd.concat([ref.features(train, spec['variant']), ref.features(valid, spec['variant'])], ignore_index=True)
        p, _ = ref.fit_predict(X, joined_y, a, b, spec, 43 + fold, spec.get('iterations'))
        prior += .5 * weight * p
    for seed in contract['v3_seeds']:
        model = ref.cat_fit(ref.features(train, 'base'), np.asarray(y), contract['v3_parameters'], seed + fold)
        prior += .5 * np.maximum(model.predict(ref.features(valid, 'base')), 0) / len(contract['v3_seeds'])
    p = contract['v3_weight'] * prior
    for name, component in contract['components'].items():
        for seed in contract['seeds']:
            model = ref.fit_v5(train, np.asarray(y), component['spec'], seed + fold)
            p += component['weight'] * ref.predict_v5(model, valid) / len(contract['seeds'])
    return p


def inner_v5_oof(raw, y, contract, seed=731, progress=None):
    prediction = np.zeros(len(raw))
    for f, (a, b) in enumerate(split_indices(raw, f'inner:{seed}')):
        prediction[b] = v5_fold_prediction(raw.iloc[a], np.asarray(y)[a], raw.iloc[b], contract, f)
        if progress:
            progress(f'inner fold {f + 1}/3 complete')
    return prediction


def fit_submission(train, test, y, contract, choice, inner_baseline=None):
    # Fresh baseline, including independent verification against the scored v5 file outside this function.
    _, baseline, legacy = ref.full_fit(train, test, y, contract)
    fitted = {'choice': choice, 'legacy': legacy, 'contract': contract, 'models': []}
    prediction = (1 - choice['weight']) * baseline
    for seed in SEEDS:
        model = fit_component(train, y, SPECS[choice['name']], seed, inner_baseline)
        fitted['models'].append(model)
        prediction += choice['weight'] * predict_component(model, test, baseline) / len(SEEDS)
    return prediction, baseline, fitted


def predict_submission(fitted, raw):
    baseline = ref.predict_final(raw, fitted['legacy'], fitted['contract'])
    choice = fitted['choice']
    component = np.mean([predict_component(m, raw, baseline) for m in fitted['models']], axis=0)
    return (1 - choice['weight']) * baseline + choice['weight'] * component
