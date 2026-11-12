"""Unchanged V9 row features, with fold-local nominal vocabularies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

ROOT=Path(__file__).resolve().parent
SPECS={name:{'family':name[0],'estimators':n,'target':'raw INR millions'}
       for name,n in [('I8',8),('I16',16),('N8',8),('N16',16)]}
COUNTS={n:40 for n in [*SPECS,'BASE']}
SOCIAL=['Facebook Popularity Quotient','Instagram Popularity Quotient']
EVENTS=['Live Music Rating','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating']
FACILITIES=['Fire Audit','Liquor License Obtained','Situated in a Multi Complex','Dedicated Parking','Open Sitting Available']
REMOVED=[]

def frame(raw):
    X=legacy_models.features(raw).reset_index(drop=True)
    assert len(X.columns)==40 and not {'Annual Turnover','Registration Number'}&set(X)
    return X

def canonical_city(values):
    mapping=json.loads((ROOT/'city_map.json').read_text())
    city=values.astype('string').str.strip().str.lower().str.replace(r'\s+',' ',regex=True)
    return city.mask(city.isin(['-1','','none','na','nan','missing','unknown'])).replace(mapping)

def native_frame(X,vocabulary=None,categorical=False):
    if vocabulary is None:
        vocabulary={c:{s:i for i,s in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                    for c in X.select_dtypes(include=['object','string','category'])}
    result=X.copy()
    for c,values in vocabulary.items():
        if categorical:
            result[c]=pd.Categorical(X[c].astype('string'),categories=list(values))
        else:result[c]=X[c].map(values).astype(float)
    for c in result:
        if c not in vocabulary or not categorical:result[c]=pd.to_numeric(result[c],errors='raise').astype(float)
    return result,vocabulary

class FeatureBuilder:
    def __init__(self,name):self.name=name;self.columns=None
    def fit(self,raw):self.columns=frame(raw).columns.tolist();return self
    def transform(self,raw):return frame(raw)[self.columns] if self.columns else frame(raw)
    def state(self):return {'feature_contract':'unchanged V9 v7 40-column features','columns':self.columns}
