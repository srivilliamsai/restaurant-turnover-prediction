"""Unchanged V9 forty-column features; fold-local categorical representations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

SPECS = {
    'I8': {'family': 'TabICLv2', 'estimators': 8, 'target': 'raw INR millions'},
    'P35_8': {'family': 'TabPFN-3.5 Base', 'estimators': 8, 'target': 'raw INR millions'},
}
WEIGHTS = [.25, .5, .75, 1.]
ROUTES = ['A', 'B']
SOCIAL = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
EVENTS = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
FACILITIES = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
              'Dedicated Parking', 'Open Sitting Available']


def frame(raw):
    result = legacy_models.features(raw).reset_index(drop=True)
    assert len(result.columns) == 40
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
        if c not in vocabulary or not categorical:
            result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    return result, vocabulary


def canonical_city(values):
    mapping = json.loads((Path(__file__).parent / 'city_map.json').read_text())
    city = values.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    return city.mask(city.isin(['-1', '', 'none', 'na', 'nan', 'missing', 'unknown'])).replace(mapping)
