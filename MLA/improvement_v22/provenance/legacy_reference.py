"""Frozen v5 predictor, copied verbatim from the preserved implementations."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR


TARGET = 'Annual Turnover'

ID = 'Registration Number'

EVENTS = ['Live Music Rating','Comedy Gigs Rating','Value Deals Rating','Live Sports Rating']

SOCIAL = ['Facebook Popularity Quotient','Instagram Popularity Quotient']

CORE = SOCIAL + ['Hygiene Rating','Restaurant Zomato Rating','Live Music Rating','Value for Money','Overall Restaurant Rating','Restaurant City Tier','Cuisine','Restaurant Theme','Restaurant Location','Endoresed By','City','Opening Year']

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

def cat_fit(X, y, spec, seed):
    params=dict(iterations=spec['iterations'], learning_rate=.04, depth=spec['depth'],
                l2_leaf_reg=spec.get('l2',5), random_seed=seed, thread_count=2,
                verbose=False, allow_writing_files=False,
                loss_function=f"Huber:delta={spec['delta']*1e6:g}" if 'delta' in spec else 'RMSE')
    if spec.get('monotone'):
        params['monotone_constraints']={**{c:1 for c in SOCIAL+['Hygiene Rating','Restaurant Zomato Rating']},'Resturant Tier':-1}
    if spec.get('ordered'): params['boosting_type']='Ordered'
    if 'subsample' in spec: params.update(bootstrap_type='Bernoulli',subsample=spec['subsample'])
    if 'random_strength' in spec: params['random_strength']=spec['random_strength']
    fitted=CatBoostRegressor(**params)
    target=np.minimum(y,spec['cap']*1e6) if 'cap' in spec else y
    fitted.fit(X,target,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
    return fitted

def get_splits(train,scheme):
    if str(scheme)=='city':
        group=features(train,'engineered')['City']
        splits=list(GroupKFold(5).split(train,groups=group))
        for tr,va in splits:assert set(group.iloc[tr]).isdisjoint(group.iloc[va])
    else:splits=list(KFold(5,shuffle=True,random_state=int(scheme)).split(train))
    counts=np.zeros(len(train),int)
    for tr,va in splits:
        assert set(tr).isdisjoint(va);counts[va]+=1
    assert (counts==1).all()
    return splits

def compute_v3(train,y,scheme,contract):
    splits=get_splits(train,scheme);pred=np.zeros(len(y))
    for name,spec in contract['v2_configs'].items():
        if name not in contract['v2_weights']:continue
        X=features(train,spec['variant']);component=np.zeros(len(y))
        for f,(tr,va) in enumerate(splits):
            component[va],_=fit_predict(X,y,tr,va,spec,43+f,spec.get('iterations'))
        pred+=.5*contract['v2_weights'][name]*component
    X=features(train,'base');spec=contract['v3_parameters']
    for f,(tr,va) in enumerate(splits):
        parts=[]
        for seed in contract['v3_seeds']:
            model=cat_fit(X.iloc[tr],y[tr],spec,seed+f)
            parts.append(np.maximum(model.predict(X.iloc[va]),0))
        pred[va]+=.5*np.mean(parts,axis=0)
    return pred

def check_submission(frame,test):
    assert frame.shape==(500,2),'Expected 500 rows and two columns'
    assert frame.columns.tolist()==['Registration Number','Annual Turnover'],'Incorrect header'
    assert frame['Registration Number'].equals(test['Registration Number']),'Restaurant IDs/order changed'
    assert frame['Registration Number'].is_unique,'Duplicate restaurant IDs'
    assert np.isfinite(frame['Annual Turnover']).all(),'Non-finite prediction'
    assert (frame['Annual Turnover']>=0).all(),'Negative turnover'

def v5_features(raw,variant='base'):
    X=features(raw,'engineered' if variant=='rich' else 'base')
    if variant in ['clean','rich']:
        X['City']=X['City'].replace({'bengaluru':'bangalore','banglore':'bangalore','gurugram':'gurgaon','gurgoan':'gurgaon','bombay':'mumbai','calcutta':'kolkata','delhi':'new delhi','bhubneshwar':'bhubaneswar','hderabad':'hyderabad','punr':'pune'})
    if variant=='rich':
        fb=X['Facebook Popularity Quotient'];ig=X['Instagram Popularity Quotient']
        X['Social ratio']=fb/(ig+1)
        X['Social product']=fb*ig
        X['Social minimum']=np.minimum(fb,ig)
        X['Social absolute gap']=np.abs(fb-ig)
        X['Quality spread']=X[['Food Rating','Hygiene Rating','Overall Restaurant Rating']].std(axis=1)
        X['Zomato x Hygiene']=X['Restaurant Zomato Rating']*X['Hygiene Rating']
        X['City tier x Restaurant tier']=X['Restaurant City Tier']*X['Resturant Tier']
    return X

def inverse_target(z,power):
    if power==0:return np.expm1(np.minimum(z,20))
    return np.maximum(z,0)**(1/power)

def fit_v5(raw,y,spec,seed):
    X=v5_features(raw,spec.get('variant','base'));target=np.asarray(y,dtype=float)/1e6
    if 'cap' in spec:target=np.minimum(target,spec['cap'])
    power=spec.get('power',1.)
    transformed=np.log1p(target) if power==0 else target**power
    loss=spec.get('loss','RMSE')
    params=dict(iterations=spec.get('iterations',850),learning_rate=.04,depth=spec.get('depth',5),
                l2_leaf_reg=10,bootstrap_type='Bernoulli',subsample=.7,loss_function=loss,
                random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    if 'rsm' in spec:params['rsm']=spec['rsm']
    if 'topk' in spec:
        pilot=CatBoostRegressor(**{**params,'iterations':300})
        pilot.fit(X,transformed,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
        ranked=np.argsort(-pilot.feature_importances_);selected=[X.columns[i] for i in ranked[:spec['topk']]]
        X=X[selected]
    model=CatBoostRegressor(**params)
    model.fit(X,transformed,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
    correction=0.
    if spec.get('correct_mean'):
        # Training-only retransformation correction; all validation labels remain unseen.
        fitted_original=inverse_target(model.predict(X),power)
        correction=float(np.mean(target-fitted_original))
    return {'model':model,'columns':X.columns.tolist(),'power':power,'correction':correction,'spec':spec}

def predict_v5(fitted,raw):
    X=v5_features(raw,fitted['spec'].get('variant','base'))[fitted['columns']]
    return np.maximum(0,(inverse_target(fitted['model'].predict(X),fitted['power'])+fitted['correction'])*1e6)

def evaluate(train,y,scheme,contract,previous=None):
    if previous is None:previous=compute_v3(train,y,scheme,contract)
    prediction=contract['v3_weight']*previous
    parts={}
    for name,c in contract['components'].items():
        p=np.zeros(len(y))
        for f,(tr,va) in enumerate(get_splits(train,scheme)):
            one=[]
            for seed in contract['seeds']:
                fitted=fit_v5(train.iloc[tr],y[tr],c['spec'],seed+f)
                one.append(predict_v5(fitted,train.iloc[va]))
            p[va]=np.mean(one,axis=0)
        parts[name]=p;prediction+=c['weight']*p
    return {'previous':previous,'candidate':prediction,**parts}

def full_fit(train,test,y,contract):
    prior=np.zeros(len(test));fitted={'v2':{},'v3':[],'new':{}}
    tr=np.arange(len(train));va=np.arange(len(train),len(train)+len(test));yy=np.r_[y,np.full(len(test),np.nan)]
    for name,w in contract['v2_weights'].items():
        spec=contract['v2_configs'][name]
        X=pd.concat([features(train,spec['variant']),features(test,spec['variant'])],ignore_index=True)
        p,_,model=fit_predict(X,yy,tr,va,spec,42,spec.get('iterations'),return_fitted=True)
        fitted['v2'][name]=model;prior+=.5*w*p
    for seed in contract['v3_seeds']:
        model=cat_fit(features(train,'base'),y,contract['v3_parameters'],seed);fitted['v3'].append(model)
        prior+=.5*np.maximum(model.predict(features(test,'base')),0)/len(contract['v3_seeds'])
    candidate=contract['v3_weight']*prior
    for name,c in contract['components'].items():
        fitted['new'][name]=[]
        for seed in contract['seeds']:
            model=fit_v5(train,y,c['spec'],seed);fitted['new'][name].append(model)
            candidate+=c['weight']*predict_v5(model,test)/len(contract['seeds'])
    return prior,candidate,fitted

def predict_final(raw,fitted,contract):
    prior=sum(.5*w*predict_fitted(fitted['v2'][name],features(raw,contract['v2_configs'][name]['variant'])) for name,w in contract['v2_weights'].items())
    prior+=.5*np.mean([np.maximum(m.predict(features(raw,'base')),0) for m in fitted['v3']],axis=0)
    p=contract['v3_weight']*prior
    for name,c in contract['components'].items():p+=c['weight']*np.mean([predict_v5(m,raw) for m in fitted['new'][name]],axis=0)
    return p
