"""Original V14 forty-column view and one exact two-column projection."""
import json
from pathlib import Path
import pandas as pd
import legacy_models

DROP = ['Service', 'Comfortablility']
SPECS = {'P35-SC38': {'family': 'TabPFN-3.5 Base', 'estimators': 8,
                     'target': 'raw INR millions', 'view': 'SC38', 'drop': DROP}}
WEIGHTS = [.25, .5, 1.]
ROUTES = ['P35-replacement']
COUNTS = {'P35-SC38': 38}
SOCIAL = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
EVENTS = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
FACILITIES = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
              'Dedicated Parking', 'Open Sitting Available']


def frame(raw):
    result = legacy_models.features(raw).reset_index(drop=True)
    assert len(result.columns) == 40
    assert not {'Annual Turnover', 'Registration Number'} & set(result.columns)
    return result


class FeatureBuilder:
    def __init__(self, view):
        assert view in ['BASE', 'SC38', 'P35-SC38']
        self.view = 'SC38' if view == 'P35-SC38' else view

    def fit(self, raw):
        self.columns = self._project(raw).columns.tolist()
        return self

    def _project(self, raw):
        result = frame(raw)
        if self.view == 'SC38':
            result = result.drop(columns=DROP)
            assert len(result.columns) == 38 and not set(DROP) & set(result)
        return result

    def transform(self, raw):
        result = self._project(raw)
        assert result.columns.tolist() == self.columns
        return result

    def state(self):
        return {'view': self.view, 'columns': self.columns,
                'drop': DROP if self.view == 'SC38' else []}


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
