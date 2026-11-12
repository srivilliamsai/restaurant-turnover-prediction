"""Small structured feature, spline, robust-ridge and kernel experiment.

All reported RMSEs retain the original validation labels, including extreme
turnovers. Target capping and Huber weights affect training rows only.
"""
from pathlib import Path
import json, sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
sys.path.insert(0,str(WORKSPACE/'improvement_v2'))
from experiment import features,load_data,TARGET,rmse

CONTINUOUS=['Facebook Popularity Quotient','Instagram Popularity Quotient','Opening Year']
CORE_NUM=CONTINUOUS+['Restaurant Zomato Rating','Hygiene Rating','Value for Money','Overall Restaurant Rating','Restaurant City Tier','Resturant Tier']

def feature_frame(raw, spec):
    X=features(raw,'engineered_drop')
    if spec.get('subset')=='core':
        X=X[CORE_NUM+['Cuisine','City','Restaurant Theme','Restaurant Location','Endoresed By']]
    else:
        X=X.drop(columns=['Opening Day','Opening Days Since 1970','Cuisine unordered','Social x Zomato','Social x Hygiene','Quality x Value'],errors='ignore')
    if spec.get('interactions'):
        social=X[CONTINUOUS[:2]].mean(axis=1)
        X['Social x Hygiene']=social*X['Hygiene Rating']
        X['Social x Zomato']=social*X['Restaurant Zomato Rating']
        X['Social x Tier']=social*X['Resturant Tier']
        X['Age x Tier']=(2020-X['Opening Year'])*X['Resturant Tier']
        X['Social squared']=social**2
        X['FB x IG']=X[CONTINUOUS[0]]*X[CONTINUOUS[1]]
    return X

def make_preprocessor(X,spec):
    cats=X.select_dtypes(include=['object','string']).columns.tolist()
    nums=[c for c in X if c not in cats]
    spline_cols=nums if spec.get('all_spline') else [c for c in CONTINUOUS if c in nums]
    if spec.get('kind')=='kernel': spline_cols=[]
    plain_cols=[c for c in nums if c not in spline_cols]
    transforms=[]
    if spline_cols:
        transforms.append(('spline',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),SplineTransformer(n_knots=spec.get('knots',4),degree=2,include_bias=False,knots='quantile'),StandardScaler()),spline_cols))
    if plain_cols:
        transforms.append(('linear',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),plain_cols))
    transforms.append(('category',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=False),cats))
    return ColumnTransformer(transforms)

def fit_model(raw,y,spec,seed=42):
    X=feature_frame(raw,spec)
    pre=make_preprocessor(X,spec)
    Z=pre.fit_transform(X)
    kernel=None
    if spec.get('kind')=='kernel':
        kernel=Nystroem(kernel='rbf',gamma=spec['gamma'],n_components=250,random_state=seed,n_jobs=2)
        Z=kernel.fit_transform(Z)
    yy=np.asarray(y,dtype=float)/1e6
    if spec.get('cap'): yy=np.minimum(yy,spec['cap'])
    model=Ridge(alpha=spec['alpha'])
    model.fit(Z,yy)
    if spec.get('huber'):
        for _ in range(15):
            residual=yy-model.predict(Z)
            weights=np.minimum(1,spec['huber']/np.maximum(np.abs(residual),1e-8))
            model.fit(Z,yy,sample_weight=weights)
    return dict(spec=spec,pre=pre,kernel=kernel,model=model)

def predict(model,raw):
    Z=model['pre'].transform(feature_frame(raw,model['spec']))
    if model['kernel'] is not None: Z=model['kernel'].transform(Z)
    return np.maximum(0,model['model'].predict(Z)*1e6)

SPECS={}
for subset in ['core','full']:
    for alpha in [30,300]:
        SPECS[f'spline_{subset}_a{alpha}']=dict(subset=subset,alpha=alpha,knots=4)
        SPECS[f'spline_{subset}_a{alpha}_h40']=dict(subset=subset,alpha=alpha,knots=4,huber=40)
    SPECS[f'spline_{subset}_a100_h20']=dict(subset=subset,alpha=100,knots=4,huber=20)
    SPECS[f'spline_{subset}_a100_c120']=dict(subset=subset,alpha=100,knots=4,cap=120)
    SPECS[f'interaction_{subset}_h40']=dict(subset=subset,alpha=100,knots=4,huber=40,interactions=True)
    SPECS[f'allspline_{subset}_h40']=dict(subset=subset,alpha=300,knots=4,huber=40,all_spline=True)
for gamma in [.01,.04]:
    for alpha in [1,10]:
        SPECS[f'kernel_g{gamma}_a{alpha}']=dict(kind='kernel',subset='full',alpha=alpha,gamma=gamma,cap=120)

def run(names,seed):
    assert seed!=909,'Seed 909 is reserved for independent root confirmation.'
    train,test=load_data(); y=train[TARGET].to_numpy(float)
    v2=json.loads((WORKSPACE/'improvement_v2/results/locked_candidate.json').read_text())
    suffix='' if seed==42 else '_fixed'
    baseline=sum(w*np.load(WORKSPACE/'improvement_v2/results'/f'{name}_{seed}{suffix}.npz')['oof'] for name,w in v2['weights'].items())
    splits=list(KFold(5,shuffle=True,random_state=seed).split(train))
    rows=[]
    with threadpool_limits(limits=2):
        for name in names:
            start=time.time(); spec=SPECS[name]; fn=HERE/f'{name}_{seed}.npz'
            if fn.exists(): oof=np.load(fn)['oof']
            else:
                oof=np.zeros(len(y))
                for fold,(tr,va) in enumerate(splits):
                    fitted=fit_model(train.iloc[tr],y[tr],spec,43+fold)
                    oof[va]=predict(fitted,train.iloc[va])
                np.savez_compressed(fn,oof=oof)
            row=dict(name=name,seed=seed,rmse=rmse(y,oof),blend15_rmse=rmse(y,.85*baseline+.15*oof),baseline_rmse=rmse(y,baseline),mean_prediction=oof.mean(),seconds=time.time()-start)
            rows.append(row); print(json.dumps(row),flush=True)
    pd.DataFrame(rows).sort_values('blend15_rmse').to_csv(HERE/f'scores_{seed}.csv',index=False)
    (HERE/'specs.json').write_text(json.dumps(SPECS,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--names',nargs='*',default=list(SPECS));a=p.parse_args();run(a.names,a.seed)
