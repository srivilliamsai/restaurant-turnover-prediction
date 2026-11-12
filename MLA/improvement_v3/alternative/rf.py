"""Bounded Random Forest diversity check; validation labels are never capped."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
sys.path.insert(0,str(WORKSPACE/'improvement_v2'))
from experiment import features,load_data,TARGET,rmse

def rf_preprocessor(X):
    cats=X.select_dtypes(include=['object','string']).columns.tolist()
    nums=[c for c in X if c not in cats]
    return ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),nums),('category',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=False),cats)])

def fit_model(raw,y,spec,seed=43):
    X=features(raw,'base'); pre=rf_preprocessor(X);z=pre.fit_transform(X)
    yy=np.asarray(y,float)/1e6
    if spec['cap'] is not None: yy=np.minimum(yy,spec['cap']/1e6)
    model=RandomForestRegressor(n_estimators=250,max_features=spec['max_features'],min_samples_leaf=spec['min_samples_leaf'],random_state=seed,n_jobs=2)
    model.fit(z,yy)
    return dict(pre=pre,model=model)

def predict(fitted,raw):
    return np.maximum(0,fitted['model'].predict(fitted['pre'].transform(features(raw,'base')))*1e6)

SPECS={f'rf_mf{mf}_leaf{leaf}_cap{cap}':dict(max_features=mf,min_samples_leaf=leaf,cap=cap) for mf in [.7,1.] for leaf in [2,8] for cap in [None,100e6]}

def run():
    train,test=load_data();y=train[TARGET].to_numpy(float)
    baseline=np.load(WORKSPACE/'improvement_v3/results/v2_42.npz')['oof']
    splits=list(KFold(5,shuffle=True,random_state=42).split(train))
    rows=[]
    with threadpool_limits(limits=2):
        for name,spec in SPECS.items():
            start=time.time();fn=HERE/f'{name}_42.npz'
            if fn.exists():oof=np.load(fn)['oof']
            else:
                oof=np.zeros(len(y))
                for fold,(tr,va) in enumerate(splits):
                    fitted=fit_model(train.iloc[tr],y[tr],spec,43+fold)
                    oof[va]=predict(fitted,train.iloc[va])
                np.savez_compressed(fn,oof=oof)
            for weight in [.25,.5,.75]:
                blend=(1-weight)*baseline+weight*oof
                row=dict(name=name,weight=weight,model_rmse=rmse(y,oof),blend_rmse=rmse(y,blend),baseline_rmse=rmse(y,baseline),improvement=rmse(y,baseline)-rmse(y,blend),seconds=time.time()-start)
                rows.append(row)
            print(json.dumps(rows[-3:]),flush=True)
            pd.DataFrame(rows).sort_values('blend_rmse').to_csv(HERE/'rf_scores_42.csv',index=False)
    (HERE/'rf_specs.json').write_text(json.dumps(SPECS,indent=2))

if __name__=='__main__':run()
