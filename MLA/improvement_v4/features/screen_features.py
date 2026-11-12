"""Bounded feature-representation experiments, full unmodified validation targets.

No fitted feature transforms or target statistics are shared between folds.
The functions here are self-contained for inclusion in the final notebook.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVENTS = ['Live Music Rating','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating']
SOCIAL = ['Facebook Popularity Quotient','Instagram Popularity Quotient']
RATINGS = ['Resturant Tier','Restaurant City Tier','Restaurant Zomato Rating','Order Wait Time','Staff Responsivness','Value for Money','Hygiene Rating','Food Rating','Overall Restaurant Rating','Ambience','Lively','Service','Comfortablility','Privacy'] + EVENTS
FLAGS = ['Fire Audit','Liquor License Obtained','Situated in a Multi Complex','Dedicated Parking','Open Sitting Available']
VARIANTS = ['base','categorical_ratings','dual_ratings','categorical_tiers_flags','social_quantized','events_zero','date_detail','city_cuisine','reduced','cuisine_canonical']
SPEC = dict(depth=5, iterations=850, delta=40, l2=10, subsample=.7)

def feature_frame(raw, variant='base'):
    data=raw.rename(columns={'Endorsed By':'Endoresed By'}).drop(columns=['Annual Turnover','Registration Number'],errors='ignore').copy()
    dates=pd.to_datetime(data.pop('Opening Day of Restaurant'),format='%d-%m-%Y',errors='coerce')
    data['Opening Year']=dates.dt.year.astype(float)
    data['Opening Month']=dates.dt.month.astype(float)
    for c in EVENTS:
        data['Offers '+c.replace(' Rating','')]=data[c].notna().astype(int)
    parts=data['Cuisine'].astype('string').fillna('Missing').str.split(',',n=1,expand=True)
    data['Cuisine 1']=parts[0].str.strip().str.lower()
    data['Cuisine 2']=parts.reindex(columns=[0,1])[1].astype('string').str.strip().str.lower()
    for c in ['Cuisine','City','Restaurant Location','Endoresed By','Restaurant Type','Restaurant Theme']:
        data[c]=data[c].astype('string').str.strip().str.lower().str.replace(r'\s+',' ',regex=True).fillna('Missing')
    data['City Is Unknown']=(data['City']=='-1').astype(int)
    for c in data.select_dtypes(include=['object','string']).columns:
        data[c]=data[c].fillna('Missing').astype(str)
    if variant == 'categorical_ratings':
        for c in RATINGS:
            data[c]=data[c].map(lambda v: 'Missing' if pd.isna(v) else f'{v:g}')
    elif variant == 'dual_ratings':
        for c in RATINGS:
            data[c+' Category']=data[c].map(lambda v: 'Missing' if pd.isna(v) else f'{v:g}')
    elif variant == 'categorical_tiers_flags':
        for c in ['Resturant Tier','Restaurant City Tier']+FLAGS:
            data[c]=data[c].map(lambda v: 'Missing' if pd.isna(v) else f'{v:g}')
    elif variant == 'social_quantized':
        for c in SOCIAL:
            data[c]=np.round(data[c]/5)*5
    elif variant == 'events_zero':
        data[EVENTS]=data[EVENTS].fillna(0)
        data['Event Count']=data[['Offers '+c.replace(' Rating','') for c in EVENTS]].sum(axis=1)
    elif variant == 'date_detail':
        data['Opening Days Since 1970']=(dates-pd.Timestamp('1970-01-01')).dt.days.astype(float)
        data['Opening Day of Month']=dates.dt.day.astype(float)
        data['Opening Day of Week']=dates.dt.dayofweek.astype(float)
        data['Opening Month Sin']=np.sin(2*np.pi*dates.dt.month/12)
        data['Opening Month Cos']=np.cos(2*np.pi*dates.dt.month/12)
    elif variant == 'city_cuisine':
        data['City']=data['City'].replace({'bengaluru':'bangalore','gurugram':'gurgaon','bombay':'mumbai','calcutta':'kolkata','delhi':'new delhi'})
        data['City Cuisine']=data['City']+'|'+data['Cuisine']
        data['Location Cuisine']=data['Restaurant Location']+'|'+data['Cuisine']
        data['Tier Cuisine']=data['Resturant Tier'].fillna(-1).astype(str)+'|'+data['Cuisine']
    elif variant == 'reduced':
        # A fixed hypothesis excluding subjective service/comfort scores and sparse event fields.
        data=data.drop(columns=['Service','Comfortablility','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating'])
    elif variant == 'cuisine_canonical':
        data['Cuisine']=['|'.join(sorted(part.strip() for part in value.split(','))) for value in data['Cuisine']]
        data=data.drop(columns=['Cuisine 1','Cuisine 2'])
    elif variant != 'base':
        raise ValueError(variant)
    return data

def fit_feature_model(X,y,seed,spec=None):
    spec=SPEC if spec is None else spec
    fitted=CatBoostRegressor(iterations=spec['iterations'],learning_rate=.04,depth=spec['depth'],l2_leaf_reg=spec['l2'],loss_function=f"Huber:delta={spec['delta']*1e6:g}",bootstrap_type='Bernoulli',subsample=spec['subsample'],random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    fitted.fit(X,np.asarray(y),cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
    return fitted

def run_variant(train,variant,split_seed=42,seeds=(1042,2042)):
    y=train['Annual Turnover'].to_numpy(float)
    file=OUT/f'{variant}_{split_seed}.npz'
    if file.exists(): return np.load(file)['oof']
    X=feature_frame(train,variant)
    individual=np.zeros((len(train),len(seeds)))
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=split_seed).split(train)):
        for j,seed in enumerate(seeds):
            fitted=fit_feature_model(X.iloc[tr],y[tr],seed+fold)
            individual[va,j]=np.maximum(fitted.predict(X.iloc[va]),0)
    prediction=individual.mean(axis=1)
    np.savez_compressed(file,oof=prediction,individual=individual,seeds=seeds)
    return prediction

def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='*',default=VARIANTS);args=parser.parse_args()
    train=pd.read_csv('~/Downloads/Train_dataset.csv')
    y=train['Annual Turnover'].to_numpy(float)
    baseline=.5*np.load(ROOT/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(ROOT/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
    (OUT/'specifications.json').write_text(json.dumps(dict(spec=SPEC,variants=VARIANTS,fold_seed=42,model_seeds=[1042,2042],baseline='0.5*v2_42+0.5*bag_subsample_h40_42',selection_metric='full original-target RMSE',reserved_seeds_not_used=[1701,1711]),indent=2))
    rows=[]
    print('V3 development baseline',rmse(y,baseline),flush=True)
    for variant in args.variants:
        start=time.time();p=run_variant(train,variant)
        row=dict(variant=variant,rmse=rmse(y,p),blend25_rmse=rmse(y,.75*baseline+.25*p),blend50_rmse=rmse(y,.5*baseline+.5*p),seconds=time.time()-start)
        rows.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'screen_scores.csv',index=False)
