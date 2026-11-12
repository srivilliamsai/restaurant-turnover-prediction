"""Two frozen V21 representations, using reviewed V10 definitions and train-only maps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

ROOT = Path(__file__).resolve().parent
SPECS = {
    'S': {'family': 'TabPFN-3.5 Base', 'estimators': 8, 'view': 'S',
          'target': 'raw INR millions', 'clean': False, 'structured': True, 'group': False},
    'C+S+G': {'family': 'TabPFN-3.5 Base', 'estimators': 8, 'view': 'C+S+G',
              'target': 'raw INR millions', 'clean': True, 'structured': True, 'group': True},
}
COUNTS = {'BASE': 40, 'S': 56, 'C+S+G': 66}
WEIGHTS = [.25, .5, 1.]
ROUTES = ['P35-replacement']

REMOVED = ['Food Rating', 'Order Wait Time']
EVENTS = ['Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating', 'Live Sports Rating']
FACILITIES = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
              'Dedicated Parking', 'Open Sitting Available']
SOCIAL = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient']
GROUP_NUMBERS = ['social', 'hygiene', 'age']


def canonical_city(values):
    mapping = json.loads((ROOT / 'city_map.json').read_text())
    city = values.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    city = city.mask(city.isin(['-1', '', 'none', 'na', 'nan', 'missing', 'unknown']))
    return city.replace(mapping)


def row_frame(raw, name):
    spec = SPECS.get(name, {})
    data = raw.reset_index(drop=True).copy()
    if spec.get('prune'):
        data = data.drop(columns=REMOVED, errors='ignore')
    X = legacy_models.features(data).reset_index(drop=True)
    if spec.get('clean'):
        X['V10 Canonical city'] = canonical_city(data['City'])
        X['V10 Zomato zero'] = X['Restaurant Zomato Rating'].eq(0).astype(float)
        X['Restaurant Zomato Rating'] = X['Restaurant Zomato Rating'].replace(0, np.nan)
    if spec.get('structured'):
        quality = {'Staff Responsivness': 8., 'Value for Money': 7., 'Hygiene Rating': 10.,
                   'Food Rating': 10., 'Restaurant Zomato Rating': 5., 'Overall Restaurant Rating': 10.}
        quality = pd.DataFrame({c: X[c] / divisor for c, divisor in quality.items() if c in X})
        event = X[EVENTS].div(pd.Series(dict(zip(EVENTS, [10., 6., 7., 6.]))))
        vibe = X[['Ambience', 'Lively', 'Service', 'Comfortablility', 'Privacy']]
        social = X[SOCIAL].mean(axis=1)
        date = pd.to_datetime(data['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
        age = ((pd.Timestamp('2026-01-01') - date).dt.days / 365.25).clip(lower=0)
        additions = {
            'Quality mean': quality.mean(axis=1), 'Quality std': quality.std(axis=1, ddof=1),
            'Quality core sum': X['Hygiene Rating']/10 + X['Staff Responsivness']/8 + X['Value for Money']/7,
            'Facilities count': X[FACILITIES].sum(axis=1, min_count=1),
            'Events count': X[EVENTS].notna().sum(axis=1).astype(float),
            'Events mean': event.mean(axis=1), 'Events max': event.max(axis=1),
            'Experience mean': vibe.mean(axis=1), 'Experience std': vibe.std(axis=1, ddof=1),
            'Social mean': social, 'Social product': X[SOCIAL[0]] * X[SOCIAL[1]],
            'Social absolute difference': (X[SOCIAL[0]] - X[SOCIAL[1]]).abs(),
            'Age': age, 'Age hygiene': age * X['Hygiene Rating']/10,
            'Age social': age * social, 'Tier city tier': X['Resturant Tier'] * X['Restaurant City Tier'],
        }
        assert len(additions) == 16
        for key, value in additions.items():
            X['V10 ' + key] = value.astype(float)
    assert 'Annual Turnover' not in X and 'Registration Number' not in X
    return X


def contexts(X):
    return {'city': X['V10 Canonical city'], 'cuisine': X['Cuisine'].astype('string')}, {
        'social': X['V10 Social mean'], 'hygiene': X['Hygiene Rating']/10, 'age': X['V10 Age']}


class FeatureBuilder:
    def __init__(self, name):
        assert name in SPECS or name == 'BASE'
        self.name, self.maps, self.columns = name, None, None

    def fit(self, raw):
        X = row_frame(raw, self.name)
        if SPECS.get(self.name, {}).get('group'):
            groups, numbers = contexts(X)
            self.maps = {}
            for key, values in groups.items():
                record = {'counts': values.value_counts().to_dict(), 'means': {}, 'global': {}}
                for num, series in numbers.items():
                    global_mean = float(series.mean()) if series.notna().any() else None
                    frame = pd.DataFrame({'key': values, 'value': series})
                    stats = frame.groupby('key', observed=True)['value'].agg(['count', 'sum'])
                    means = (stats['sum'] + 20 * global_mean)/(stats['count'] + 20) if global_mean is not None else stats['sum'] * np.nan
                    record['global'][num] = global_mean
                    record['means'][num] = means.dropna().to_dict()
                self.maps[key] = record
        self.columns = self.transform(raw).columns.tolist()
        assert len(self.columns) == COUNTS[self.name]
        return self

    def transform(self, raw):
        X = row_frame(raw, self.name)
        if SPECS.get(self.name, {}).get('group'):
            assert self.maps is not None, 'Fit group maps before inference.'
            groups, numbers = contexts(X)
            for key, values in groups.items():
                record = self.maps[key]
                X[f'V10 Log {key} count'] = np.log1p(values.map(record['counts']).fillna(0).astype(float))
                for num, series in numbers.items():
                    mean = values.map(record['means'][num]).astype(float)
                    fallback = record['global'][num]
                    if fallback is not None:
                        mean = mean.fillna(fallback)
                    X[f'V10 {num} relative {key}'] = series - mean
        if self.columns is not None:
            assert X.columns.tolist() == self.columns, 'Feature schema changed.'
        return X

    def state(self):
        return {'name': self.name, 'columns': self.columns, 'maps': self.maps}


def native_frame(X, vocabulary=None, categorical=False):
    assert not categorical
    cats = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if vocabulary is None:
        vocabulary = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))} for c in cats}
    result = X.copy()
    for c, values in vocabulary.items():
        result[c] = X[c].map(values).astype(float)
    for c in result:
        result[c] = pd.to_numeric(result[c], errors='raise').astype(float)
    assert not np.isinf(result.to_numpy()).any()
    return result, vocabulary


def frame(raw, view="F40"):
    return row_frame(raw, "BASE" if view == "F40" else view)

