"""Reproducible regression experiments; every scored target remains unmodified."""
from pathlib import Path
import argparse, json, time, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

warnings.filterwarnings('ignore', category=FutureWarning)
ROOT = Path(__file__).resolve().parent
TARGET = 'Annual Turnover'
ID = 'Registration Number'
EVENTS = ['Live Music Rating','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating']
SOCIAL = ['Facebook Popularity Quotient','Instagram Popularity Quotient']
CORE = SOCIAL + ['Hygiene Rating','Restaurant Zomato Rating','Live Music Rating','Value for Money','Overall Restaurant Rating','Restaurant City Tier','Cuisine','Restaurant Theme','Restaurant Location','Endoresed By','City','Opening Year']

def load_data(data_dir='~/Downloads'):
    d=Path(data_dir)
    train=pd.read_csv(d/'Train_dataset.csv').rename(columns={'Endorsed By':'Endoresed By'})
    test=pd.read_csv(d/'Test_dataset.csv')
    return train,test

def features(raw, variant='base'):
    data=raw.drop(columns=[TARGET,ID],errors='ignore').copy()
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
    if variant in ['engineered','engineered_drop']:
        data['City']=data['City'].replace({'bengaluru':'bangalore','gurugram':'gurgaon','bombay':'mumbai','calcutta':'kolkata','delhi':'new delhi'})
        data['Cuisine unordered']=['|'.join(sorted(v.split(','))) for v in data['Cuisine']]
        data['Opening Day']=dates.dt.day.astype(float)
        data['Opening Days Since 1970']=(dates-pd.Timestamp('1970-01-01')).dt.days.astype(float)
        data['Social mean']=data[SOCIAL].mean(axis=1)
        data['Social difference']=data[SOCIAL[0]]-data[SOCIAL[1]]
        data['Social missing count']=data[SOCIAL].isna().sum(axis=1)
        data['Rating missing count']=data.isna().sum(axis=1)
        data['Event count']=data[EVENTS].notna().sum(axis=1)
        data['Event mean']=data[EVENTS].mean(axis=1)
        data['Quality mean']=data[['Food Rating','Hygiene Rating','Overall Restaurant Rating']].mean(axis=1)
        data['Social x Zomato']=data['Social mean']*data['Restaurant Zomato Rating']
        data['Social x Hygiene']=data['Social mean']*data['Hygiene Rating']
        data['Quality x Value']=data['Quality mean']*data['Value for Money']
    if variant in ['drop_shift','engineered_drop']:
        data=data.drop(columns=['Service','Comfortablility'])
    if variant=='core': data=data[CORE]
    if variant=='ratings':
        keep=SOCIAL+['Resturant Tier','Restaurant Zomato Rating','Restaurant City Tier','Staff Responsivness','Value for Money','Hygiene Rating','Overall Restaurant Rating']+EVENTS+['Lively','Ambience','Privacy']
        data=data[keep]
    if variant=='no_city_date':
        data=data.drop(columns=['City','City Is Unknown','Opening Year','Opening Month'])
    return data

def rmse(y,p): return float(np.sqrt(np.mean(np.square(np.asarray(y)-np.asarray(p)))))

SPECS = {
 'raw_baseline':dict(kind='cat',variant='base',depth=6,iterations=302,loss='RMSE'),
 'huber_baseline':dict(kind='cat',variant='base',depth=6,iterations=768,loss='Huber:delta=12000000'),
 'drop_shift':dict(kind='cat',variant='drop_shift',depth=6),
 'engineered':dict(kind='cat',variant='engineered',depth=6),
 'engineered_drop':dict(kind='cat',variant='engineered_drop',depth=6),
 'core_depth4':dict(kind='cat',variant='core',depth=4),
 'core_depth3':dict(kind='cat',variant='core',depth=3),
 'shallow_depth3':dict(kind='cat',variant='base',depth=3,l2=15),
 'ordered_depth5':dict(kind='cat',variant='base',depth=5,boosting_type='Ordered'),
 'huber20_engineered':dict(kind='cat',variant='engineered_drop',depth=5,loss='Huber:delta=20000000'),
 'huber40':dict(kind='cat',variant='base',depth=5,loss='Huber:delta=40000000'),
 'clip80_engineered':dict(kind='cat',variant='engineered_drop',depth=5,cap=80e6),
 'clip120':dict(kind='cat',variant='base',depth=5,cap=120e6),
 'ridge':dict(kind='ridge',variant='base',alpha=100),
 'spline_ridge':dict(kind='spline',variant='base',alpha=100),
 'spline_core':dict(kind='spline',variant='core',alpha=30),
 'svr_core':dict(kind='svr',variant='core',C=20),
 'hist_core':dict(kind='hist',variant='core'),
 'ratings_depth4':dict(kind='cat',variant='ratings',depth=4),
 'ratings_huber40':dict(kind='cat',variant='ratings',depth=5,loss='Huber:delta=40000000'),
 'ratings_clip100':dict(kind='cat',variant='ratings',depth=5,cap=100e6),
 'no_city_date':dict(kind='cat',variant='no_city_date',depth=5),
 'no_city_huber40':dict(kind='cat',variant='no_city_date',depth=5,loss='Huber:delta=40000000'),
 'clip120_depth6':dict(kind='cat',variant='base',depth=6,cap=120e6),
 'clip120_depth4':dict(kind='cat',variant='base',depth=4,cap=120e6),
 'huber40_depth6':dict(kind='cat',variant='base',depth=6,loss='Huber:delta=40000000'),
 'huber80_depth5':dict(kind='cat',variant='base',depth=5,loss='Huber:delta=80000000'),
 'huber40_engineered':dict(kind='cat',variant='engineered',depth=5,loss='Huber:delta=40000000'),
 'huber40_monotone':dict(kind='cat',variant='base',depth=5,loss='Huber:delta=40000000',monotone=True),
 'extra_clip100':dict(kind='extra',variant='base',cap=100e6),
 'lgb15':dict(kind='lgb',variant='base',leaves=15,leaf_samples=60),
 'lgb7':dict(kind='lgb',variant='base',leaves=7,leaf_samples=80),
 'lgb_huber':dict(kind='lgb',variant='base',leaves=15,leaf_samples=60,objective='huber'),
 'lgb_clip120':dict(kind='lgb',variant='base',leaves=15,leaf_samples=60,cap=120e6),
 'xgb_depth3':dict(kind='xgb',variant='base',depth=3),
 'xgb_depth5':dict(kind='xgb',variant='base',depth=5),
 'xgb_huber':dict(kind='xgb',variant='base',depth=3,objective='reg:pseudohubererror'),
 'xgb_clip120':dict(kind='xgb',variant='base',depth=3,cap=120e6),
}

def sklearn_model(X,spec,seed):
    cats=X.select_dtypes(include=['object','string']).columns.tolist()
    nums=[c for c in X if c not in cats]
    num=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
    if spec['kind']=='spline':
        num=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),SplineTransformer(n_knots=4,degree=2,include_bias=False),StandardScaler())
    pre=ColumnTransformer([('num',num,nums),('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=False),cats)])
    kind=spec['kind']
    if kind in ['ridge','spline']: model=Ridge(alpha=spec.get('alpha',100))
    elif kind=='svr': model=SVR(C=spec.get('C',20),epsilon=.5,kernel='rbf')
    elif kind=='hist': model=HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=10,max_depth=4,learning_rate=.035,l2_regularization=30,min_samples_leaf=60,random_state=seed)
    elif kind=='extra': model=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,max_features=.7,n_jobs=4,random_state=seed)
    return make_pipeline(pre,model)

def fit_predict(X,y,train_idx,valid_idx,spec,seed,fixed_iterations=None,return_fitted=False):
    ys=np.asarray(y)[train_idx].copy()
    if 'cap' in spec: ys=np.minimum(ys,spec['cap'])
    if spec['kind']=='cat':
        params=dict(loss_function=spec.get('loss','RMSE'),eval_metric='RMSE',iterations=fixed_iterations or spec.get('iterations',1600),learning_rate=.04,depth=spec.get('depth',6),l2_leaf_reg=spec.get('l2',5),random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
        if 'boosting_type' in spec: params['boosting_type']=spec['boosting_type']
        if spec.get('monotone'): params['monotone_constraints']={**{c:1 for c in SOCIAL+['Hygiene Rating','Restaurant Zomato Rating']},'Resturant Tier':-1}
        model=CatBoostRegressor(**params)
        cat=X.select_dtypes(include=['object','string']).columns.tolist()
        stopping={} if fixed_iterations or 'iterations' in spec else dict(eval_set=(X.iloc[valid_idx],np.asarray(y)[valid_idx]),early_stopping_rounds=120)
        model.fit(X.iloc[train_idx],ys,cat_features=cat,**stopping)
        p=model.predict(X.iloc[valid_idx]); trees=model.tree_count_
    elif spec['kind'] in ['lgb','xgb']:
        pre=sklearn_model(X,dict(kind='ridge'),seed)[:-1]
        ztr=pre.fit_transform(X.iloc[train_idx]); zva=pre.transform(X.iloc[valid_idx])
        early=not (fixed_iterations or 'iterations' in spec)
        iterations=fixed_iterations or spec.get('iterations',1400)
        if spec['kind']=='lgb':
            from lightgbm import LGBMRegressor,early_stopping
            model=LGBMRegressor(n_estimators=iterations,learning_rate=.025,num_leaves=spec.get('leaves',15),min_child_samples=spec.get('leaf_samples',60),reg_lambda=20,reg_alpha=.1,colsample_bytree=.8,subsample=.8,subsample_freq=1,random_state=seed,n_jobs=4,verbosity=-1,objective=spec.get('objective','regression'))
            params=dict(eval_set=[(zva,np.asarray(y)[valid_idx]/1e6)],eval_metric='rmse',callbacks=[early_stopping(100,verbose=False,first_metric_only=True)]) if early else {}
            model.fit(ztr,ys/1e6,**params)
            trees=model.best_iteration_ if early else iterations
        else:
            from xgboost import XGBRegressor
            params=dict(n_estimators=iterations,max_depth=spec.get('depth',3),learning_rate=.025,min_child_weight=30,reg_lambda=20,reg_alpha=.1,colsample_bytree=.8,subsample=.8,random_state=seed,n_jobs=4,tree_method='hist',objective=spec.get('objective','reg:squarederror'),eval_metric='rmse',base_score=float(ys.mean()/1e6))
            if early: params['early_stopping_rounds']=100
            if spec.get('objective')=='reg:pseudohubererror': params['huber_slope']=12
            model=XGBRegressor(**params)
            model.fit(ztr,ys/1e6,**(dict(eval_set=[(zva,np.asarray(y)[valid_idx]/1e6)],verbose=False) if early else {}))
            trees=model.best_iteration+1 if early else iterations
        p=model.predict(zva)*1e6
    else:
        model=sklearn_model(X,spec,seed)
        model.fit(X.iloc[train_idx],ys/1e6)
        p=model.predict(X.iloc[valid_idx])*1e6; trees=0
    if return_fitted:
        fitted=dict(model=model,preprocessor=pre if spec['kind'] in ['lgb','xgb'] else None,scale=1 if spec['kind']=='cat' else 1e6)
        return np.maximum(p,0),trees,fitted
    return np.maximum(p,0),trees

def predict_fitted(fitted,X):
    transformed=fitted['preprocessor'].transform(X) if fitted['preprocessor'] is not None else X
    return np.maximum(fitted['model'].predict(transformed)*fitted['scale'],0)

def run(spec_names,split_seed=42,confirmation=False):
    train,test=load_data(); y=train[TARGET].to_numpy(float)
    splits=list(KFold(n_splits=5,shuffle=True,random_state=split_seed).split(train))
    data={v:features(train,v) for v in {SPECS[s]['variant'] for s in spec_names}}
    rows=[]
    for name in spec_names:
        t=time.time(); spec=SPECS[name]; X=data[spec['variant']]
        file=ROOT/'results'/f'{name}_{split_seed}{"_fixed" if confirmation else ""}.npz'
        if file.exists():
            cache=np.load(file); p=cache['oof']; trees=cache['trees'].tolist()
        else:
            fixed=None
            if confirmation and spec['kind'] in ['cat','lgb','xgb']:
                fixed=int(np.median(np.load(ROOT/'results'/f'{name}_42.npz')['trees']))
            p=np.zeros(len(y)); trees=[]
            for fold,(tr,va) in enumerate(splits):
                pred,n=fit_predict(X,y,tr,va,spec,42+fold+1,fixed)
                p[va]=pred; trees.append(n)
            np.savez_compressed(file,oof=p,trees=trees)
        row=dict(model=name,split_seed=split_seed,rmse=rmse(y,p),rmse_below_100m=rmse(y[y<=1e8],p[y<=1e8]),mean_prediction=p.mean(),trees=trees)
        rows.append(row); print(json.dumps(row),'seconds',round(time.time()-t,1),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'scores_{split_seed}{"_fixed" if confirmation else ""}.csv',index=False)
    if {'raw_baseline','huber_baseline'}<=set(spec_names):
        raw=np.load(ROOT/'results'/f'raw_baseline_{split_seed}{"_fixed" if confirmation else ""}.npz')['oof']
        huber=np.load(ROOT/'results'/f'huber_baseline_{split_seed}{"_fixed" if confirmation else ""}.npz')['oof']
        p=.55*raw+.45*huber
        np.savez_compressed(ROOT/'results'/f'current_blend_{split_seed}{"_fixed" if confirmation else ""}.npz',oof=p,trees=[])
        print('CURRENT BLEND',rmse(y,p),'BODY',rmse(y[y<=1e8],p[y<=1e8]),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--models',nargs='*',default=list(SPECS)); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args(); run(args.models,args.seed,args.confirmation)
