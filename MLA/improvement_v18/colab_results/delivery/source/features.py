"""Frozen V14 inputs and the two predeclared V18 settings."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

PROTOCOL = json.loads((Path(__file__).parent/'protocol_proposed.json').read_text())
SPECS = PROTOCOL['settings']
WEIGHTS = PROTOCOL['weights']
ROUTES = PROTOCOL['routes']
CATEGORICAL = ['Cuisine', 'City', 'Restaurant Location', 'Endoresed By',
               'Restaurant Type', 'Restaurant Theme']
SOCIAL = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
EVENTS = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
FACILITIES = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
              'Dedicated Parking', 'Open Sitting Available']


def frame(raw, view='F40'):
    raw = raw.rename(columns={'Endorsed By': 'Endoresed By'})
    result = legacy_models.features(raw).reset_index(drop=True)
    assert len(result.columns) == 40
    if view == 'N32':
        native = raw.drop(columns=['Annual Turnover', 'Registration Number'], errors='ignore').copy().reset_index(drop=True)
        for c in CATEGORICAL:
            native[c] = result[c]
        native['Opening Day of Restaurant'] = pd.to_datetime(
            native['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
        result = native
        assert len(result.columns) == 32
    else:
        assert view == 'F40'
    assert not {'Annual Turnover', 'Registration Number'} & set(result.columns)
    return result


def native_frame(X, vocabulary=None, categorical=False):
    if vocabulary is None:
        vocabulary = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                      for c in X.select_dtypes(include=['object', 'string', 'category'])}
    result = X.copy()
    for c, values in vocabulary.items():
        if categorical:
            result[c] = pd.Categorical(X[c].astype('string'), categories=list(values))
        else:
            result[c] = X[c].map(values).astype(float)
    for c in result:
        if pd.api.types.is_datetime64_any_dtype(result[c]):
            continue
        if c not in vocabulary or not categorical:
            result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    return result, vocabulary


def canonical_city(values):
    mapping = json.loads((Path(__file__).parent / 'city_map.json').read_text())
    city = values.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    return city.mask(city.isin(['-1', '', 'none', 'na', 'nan', 'missing', 'unknown'])).replace(mapping)

