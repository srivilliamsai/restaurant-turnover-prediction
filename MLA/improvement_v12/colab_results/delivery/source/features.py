"""Frozen V9 source features; fold-local neural and TabPFN adapters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import legacy_models

ROOT=Path(__file__).resolve().parent
SEEDS=[10242,20242,30242,40242]
SPECS={'T32':{'family':'T','atoms':['T32_10242'],'cost':32},
       'T4x16':{'family':'T','atoms':['BASE']+[f'T16_{s}' for s in SEEDS[1:]],'cost':64},
       'R1':{'family':'R','atoms':['R_10242'],'cost':1},
       'R4':{'family':'R','atoms':[f'R_{s}' for s in SEEDS],'cost':4}}
SOCIAL=['Facebook Popularity Quotient','Instagram Popularity Quotient']
EVENTS=['Live Music Rating','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating']
FACILITIES=['Fire Audit','Liquor License Obtained','Situated in a Multi Complex','Dedicated Parking','Open Sitting Available']

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
        result[c]=pd.Categorical(X[c].astype('string'),categories=list(values)) if categorical else X[c].map(values).astype(float)
    for c in result:
        if c not in vocabulary or not categorical:result[c]=pd.to_numeric(result[c],errors='raise').astype(float)
    return result,vocabulary

class FeatureBuilder:
    def __init__(self,name):self.name=name;self.columns=None;self.vocabulary=None
    def fit(self,raw):
        X=frame(raw);self.columns=X.columns.tolist()
        self.vocabulary={c:sorted(X[c].dropna().astype(str).unique().tolist()) for c in X.select_dtypes(include=['object','string','category'])}
        self.numeric=[c for c in self.columns if c not in self.vocabulary]
        return self
    def transform(self,raw):
        X=frame(raw)[self.columns] if self.columns else frame(raw)
        if self.name.startswith('R'):
            for c,values in self.vocabulary.items():
                s=X[c].astype('string');s=s.where(s.isin(values),'__V12_UNKNOWN__').fillna('__V12_UNKNOWN__')
                X[c]=pd.Categorical(s,categories=values+['__V12_UNKNOWN__'])
            for c in self.numeric:
                s=pd.to_numeric(X[c],errors='raise').astype(float)
                X[c+'__missing']=s.isna().astype(float);X[c]=s.fillna(0.)
        return X
    def state(self):return {'source_features':self.columns,'numeric':self.numeric,'vocabulary':self.vocabulary,'neural':self.name.startswith('R')}
