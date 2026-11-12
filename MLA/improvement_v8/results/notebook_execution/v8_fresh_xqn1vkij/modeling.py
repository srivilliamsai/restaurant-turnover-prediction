"""Original V8 experiments: native HGB, cross-fitted statistics and Ridge."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
import legacy_models as legacy

SEED = 10242
WEIGHTS = [.25, .5, .75, 1.]
SPECS = {}
for target in ['raw', 'sqrt']:
    for leaves in [7, 15]:
        SPECS[f'native_{target}_{leaves}'] = dict(family='native', group='hgb', target=target, leaves=leaves)
    for smooth in [50, 200]:
        SPECS[f'encoded_{target}_{smooth}'] = dict(family='encoded', group='hgb', target=target, leaves=15, smooth=smooth)
    for alpha in [10, 100]:
        for structured in [False, True]:
            SPECS[f'ridge_{target}_{alpha}_{"structured" if structured else "base"}'] = dict(
                family='ridge', group='ridge', target=target, alpha=alpha, structured=structured)
TE_COLUMNS = ['Cuisine', 'Restaurant Theme', 'Opening Year']


def features(raw):
    return legacy.features(raw).reset_index(drop=True)


def structured_frame(raw, maps=None):
    X = features(raw)
    if maps is None:
        maps = {c: X[c].value_counts().to_dict() for c in ['City', 'Cuisine']}
    social = X[['Facebook Popularity Quotient', 'Instagram Popularity Quotient']].mean(axis=1)
    date = pd.to_datetime(raw['Opening Day of Restaurant'].reset_index(drop=True), format='%d-%m-%Y', errors='coerce')
    age = ((pd.Timestamp('2026-01-01') - date).dt.days / 365.25).clip(lower=0)
    X['V8 Social mean'] = social
    X['V8 Age years'] = age
    X['V8 Social tier'] = social * X['Resturant Tier']
    X['V8 Social hygiene'] = social * X['Hygiene Rating']
    X['V8 Age social'] = age * social
    X['V8 Tier city tier'] = X['Resturant Tier'] * X['Restaurant City Tier']
    for c in ['City', 'Cuisine']:
        X[f'V8 Log {c} count'] = np.log1p(X[c].map(maps[c]).fillna(0).astype(float))
    return X, maps


def categorical_columns(X):
    return X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()


def native_frame(X, state=None):
    """Known rare levels pool together; truly unseen levels become missing."""
    result = X.copy()
    if state is None:
        state = {}
        for c in categorical_columns(X):
            counts = X[c].value_counts().to_dict()
            ranked = sorted(counts, key=lambda value: (-counts[value], str(value)))
            kept = [v for v in ranked if counts[v] >= 10][:253]
            # Integer codes avoid collisions with literal data values.
            mapping = {v: (kept.index(v) if v in kept else 253) for v in ranked}
            state[c] = mapping
    for c, mapping in state.items():
        codes = X[c].map(mapping).astype(float)
        result[c] = pd.Categorical(codes, categories=sorted(set(mapping.values())))
        assert len(result[c].cat.categories) <= 254
    for c in result.columns.difference(list(state)):
        result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    return result, state


def target_frame(X):
    result = X[TE_COLUMNS].copy()
    result['Opening Year'] = result['Opening Year'].map(lambda v: 'Missing' if pd.isna(v) else f'{v:g}')
    return result


def fit_single(raw, y, spec):
    X = features(raw)
    target = np.asarray(y, float) / 1e6
    transformed = np.sqrt(target) if spec['target'] == 'sqrt' else target
    fitted = {'spec': spec.copy(), 'frequency_maps': None, 'encoder': None, 'native_state': None}
    if spec['family'] == 'ridge':
        if spec['structured']:
            X, fitted['frequency_maps'] = structured_frame(raw)
        cats = categorical_columns(X)
        nums = [c for c in X if c not in cats]
        pre = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler()), nums),
            ('category', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False), cats)])
        model = make_pipeline(pre, Ridge(alpha=spec['alpha'], fit_intercept=True, solver='cholesky'))
    else:
        if spec['family'] == 'encoded':
            enc = TargetEncoder(target_type='continuous', smooth=spec['smooth'],
                                cv=KFold(5, shuffle=True, random_state=831))
            values = enc.fit_transform(target_frame(X), target)
            fitted['encoder'] = enc
            for j, c in enumerate(TE_COLUMNS):
                X[f'V8 TE {c}'] = values[:, j]
        X, fitted['native_state'] = native_frame(X)
        model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.03,
            max_iter=600, max_leaf_nodes=spec['leaves'], min_samples_leaf=50,
            l2_regularization=20, max_bins=255, early_stopping=False,
            categorical_features='from_dtype', random_state=SEED)
    model.fit(X, transformed)
    fitted['model'] = model
    fitted['columns'] = X.columns.tolist()
    return fitted


def predict_single(fitted, raw):
    spec = fitted['spec']
    X = features(raw)
    if fitted['frequency_maps'] is not None:
        X, _ = structured_frame(raw, fitted['frequency_maps'])
    if fitted['encoder'] is not None:
        values = fitted['encoder'].transform(target_frame(X))
        for j, c in enumerate(TE_COLUMNS):
            X[f'V8 TE {c}'] = values[:, j]
    if fitted['native_state'] is not None:
        X, _ = native_frame(X, fitted['native_state'])
    p = np.maximum(fitted['model'].predict(X[fitted['columns']]), 0)
    if spec['target'] == 'sqrt':
        p = p ** 2
    p = np.asarray(p, float) * 1e6
    assert p.shape == (len(raw),) and np.isfinite(p).all()
    return p


def fit_predict(raw, y, valid, spec, mode='plain', progress=None):
    fitted = fit_model(raw, y, spec, mode, progress)
    return predict_model(fitted, valid)


def fit_model(raw, y, spec, mode='plain', progress=None):
    assert mode in ['plain', 'bag5', 'mix50']
    fitted = {'mode': mode, 'full': None, 'bags': []}
    if mode in ['plain', 'mix50']:
        fitted['full'] = fit_single(raw, y, spec)
    if mode != 'plain':
        for i, (a, _) in enumerate(KFold(5, shuffle=True, random_state=831).split(raw)):
            fitted['bags'].append(fit_single(raw.iloc[a], np.asarray(y)[a], spec))
            if progress:
                progress(f'bag {i + 1}/5 complete')
    return fitted


def predict_model(fitted, raw):
    if fitted['mode'] == 'plain':
        return predict_single(fitted['full'], raw)
    p = np.mean([predict_single(m, raw) for m in fitted['bags']], axis=0)
    return p if fitted['mode'] == 'bag5' else .5 * (p + predict_single(fitted['full'], raw))
