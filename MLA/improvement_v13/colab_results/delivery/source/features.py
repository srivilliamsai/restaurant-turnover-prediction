"""Unmodified V9 forty-column representation and training-only vocabularies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

SPECS = {
    'Q99': {'target': 'cap', 'quantile': .99},
    'Q995': {'target': 'cap', 'quantile': .995},
    'SQ2': {'target': 'sqrt_second_moment'},
}
WEIGHTS = [.25, .5, .75, 1.]
SOCIAL = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
EVENTS = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
FACILITIES = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
              'Dedicated Parking', 'Open Sitting Available']


def frame(raw):
    result = legacy_models.features(raw).reset_index(drop=True)
    assert len(result.columns) == 40
    assert not {'Annual Turnover', 'Registration Number'} & set(result.columns)
    return result


def native_frame(X, vocabulary=None):
    if vocabulary is None:
        vocabulary = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                      for c in X.select_dtypes(include=['object', 'string', 'category'])}
    result = X.copy()
    for c, values in vocabulary.items():
        result[c] = X[c].map(values).astype(float)
    for c in result:
        result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    return result, vocabulary


def canonical_city(values):
    mapping = json.loads((Path(__file__).parent / 'city_map.json').read_text())
    city = values.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    return city.mask(city.isin(['-1', '', 'none', 'na', 'nan', 'missing', 'unknown'])).replace(mapping)


def training_target(y, name):
    original = np.asarray(y, dtype=float)
    assert original.ndim == 1 and np.isfinite(original).all() and (original > 0).all()
    metadata = {'name': name, 'training_rows': len(original), 'units': 'INR millions'}
    if name == 'BASE':
        result = original / 1e6
    elif SPECS[name]['target'] == 'cap':
        q = SPECS[name]['quantile']
        threshold = float(np.quantile(original, q, method='linear'))
        result = np.minimum(original, threshold) / 1e6
        metadata.update(quantile=q, quantile_method='linear', threshold_INR=threshold,
                        capped_rows=int((original > threshold).sum()))
    else:
        result = np.sqrt(original / 1e6)
        metadata['units'] = 'sqrt(INR millions)'
    return result, metadata
