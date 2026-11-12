"""Fixed, bounded v4 experiments. No test labels or leaderboard fitting."""
from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'improvement_v2'))
from experiment import load_data,features,rmse
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
COMMON=dict(iterations=850,learning_rate=.04,depth=5,l2_leaf_reg=10,
            loss_function='Huber:delta=40000000',bootstrap_type='Bernoulli',subsample=.7)
TRIALS={
 'borders32':dict(border_count=32),
 'borders64':dict(border_count=64),
 'columns70':dict(rsm=.7),
 'rows50':dict(subsample=.5,iterations=1000),
 'rows90':dict(subsample=.9),
 'l2_30':dict(l2_leaf_reg=30,iterations=1000),
 'l2_3':dict(l2_leaf_reg=3),
 'huber30':dict(loss_function='Huber:delta=30000000'),
 'huber50':dict(loss_function='Huber:delta=50000000'),
 'slow':dict(learning_rate=.02,iterations=1700),
 'depth4':dict(depth=4,iterations=1200),
 'depth6':dict(depth=6,iterations=700),
}

def parameter_fit(X,y,spec,seed):
    model=CatBoostRegressor(**spec,random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    model.fit(X,y,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
    return model

def run(name,train,y,split_seed=42,seeds=(1042,2042)):
    file=OUT/f'{name}_{split_seed}.npz'
    if file.exists(): return np.load(file)['oof']
    X=features(train,'base');individual=np.zeros((len(y),len(seeds)))
    for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=split_seed).split(train)):
        for j,seed in enumerate(seeds):
            model=parameter_fit(X.iloc[tr],y[tr],{**COMMON,**TRIALS[name]},seed+f)
            individual[va,j]=np.maximum(model.predict(X.iloc[va]),0)
    oof=individual.mean(axis=1)
    np.savez_compressed(file,oof=oof,individual=individual,seeds=seeds)
    return oof

if __name__=='__main__':
    (OUT/'parameter_specs.json').write_text(json.dumps({k:{**COMMON,**v} for k,v in TRIALS.items()},indent=2))
    train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
    baseline=.5*np.load(BASE/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(BASE/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
    print('V3 baseline',rmse(y,baseline),flush=True);rows=[]
    for name in TRIALS:
        start=time.time();p=run(name,train,y)
        row=dict(name=name,rmse=rmse(y,p),blend25_rmse=rmse(y,.75*baseline+.25*p),blend50_rmse=rmse(y,.5*baseline+.5*p),seconds=time.time()-start)
        rows.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'parameter_scores.csv',index=False)
