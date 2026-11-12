"""Frozen-model evaluation and fitting for the fourth improvement round."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold,GroupKFold
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'improvement_v2'))
from experiment import features,rmse,fit_predict,predict_fitted,load_data
sys.path.insert(0,str(BASE/'improvement_v3'))
from experiments import cat_fit
sys.path.insert(0,str(ROOT/'features'))
from screen_features import feature_frame,EVENTS,SOCIAL,RATINGS,FLAGS

def load_contract():
    return json.loads((OUT/'candidate_lock.json').read_text())

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def current_input_hashes():
    data=Path('~/Downloads')
    return {name:file_sha256(data/name) for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']}

def validate_supplied_training(train,y):
    actual,_=load_data()
    pd.testing.assert_frame_equal(train,actual,check_exact=True)
    assert np.array_equal(np.asarray(y),actual['Annual Turnover'].to_numpy(float)),'Validation targets/order differ from source data'

def checked_prior_v3(train,y,scheme,contract):
    """Read original v3 OOF only after manifest and model-contract validation."""
    prior=BASE/'improvement_v3/results';manifest=json.loads((prior/'cache_manifest.json').read_text())
    validate_supplied_training(train,y)
    assert manifest['input_hashes']==current_input_hashes(),'V3 cached inputs changed'
    v2=json.loads((BASE/'improvement_v2/results/locked_candidate.json').read_text())
    weights=json.loads((BASE/'improvement_v2/results/final_execution.json').read_text())['weights']
    v3=json.loads((prior/'candidate_lock.json').read_text())
    expected={'v2_configs':v2['configs'],'v2_weights':weights,'v3':v3}
    assert manifest['configuration_sha256']==json_sha256(expected),'V3 cache configuration changed'
    assert contract['v2_configs']==expected['v2_configs'] and contract['v2_weights']==weights,'V4 embeds a different v2 model'
    assert v3['v2_weight']==.5 and v3['additional_weights']=={'bag_subsample_h40':.5},'V3 mixture differs from evaluation implementation'
    assert contract['v3_parameters']==v3['specs']['bag_subsample_h40'] and contract['v3_seeds']==v3['model_seeds'],'V4 embeds a different v3 component'
    name=f'comparison_{scheme}.npz';path=prior/name
    assert name in manifest['files'] and file_sha256(path)==manifest['files'][name],'V3 cache prediction file changed'
    with np.load(path,allow_pickle=False) as data:pred=data['candidate'].copy()
    assert pred.shape==np.asarray(y).shape and np.isfinite(pred).all() and (pred>=0).all(),'Invalid v3 prediction array'
    return pred

def validation_cache_context(train,y,scheme,contract):
    validate_supplied_training(train,y)
    sources=[ROOT/'pipeline.py',BASE/'improvement_v2/experiment.py',BASE/'improvement_v3/experiments.py',ROOT/'features/screen_features.py',ROOT/'confirm_candidate.py']
    return {'format_version':1,'scheme':str(scheme),'input_hashes':current_input_hashes(),
            'contract_sha256':json_sha256(contract),'rows':len(train),
            'implementation_sha256':{str(p.relative_to(BASE)):file_sha256(p) for p in sources}}

def write_validation_cache(path,predictions,train,y,scheme,contract):
    path=Path(path)
    assert set(predictions)=={'previous','candidate'}
    assert all(np.asarray(p).shape==np.asarray(y).shape and np.isfinite(p).all() and (np.asarray(p)>=0).all() for p in predictions.values())
    np.savez_compressed(path,**predictions)
    metadata={'context':validation_cache_context(train,y,scheme,contract),'file_sha256':file_sha256(path),'origin':'computed by guarded confirmation'}
    path.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2)+'\n')

def read_validation_cache(path,train,y,scheme,contract):
    """Reject unstamped/stale caches; never silently fall back or adopt them."""
    path=Path(path);meta_path=path.with_suffix('.meta.json')
    assert meta_path.exists(),'Prediction cache lacks provenance metadata: '+str(path)
    metadata=json.loads(meta_path.read_text())
    assert metadata['context']==validation_cache_context(train,y,scheme,contract),'Prediction cache inputs, configuration or implementation changed'
    assert metadata['file_sha256']==file_sha256(path),'Prediction cache contents changed'
    with np.load(path,allow_pickle=False) as data:predictions={k:data[k].copy() for k in data.files}
    assert set(predictions)=={'previous','candidate'},'Unexpected cached arrays'
    assert all(p.shape==np.asarray(y).shape and np.isfinite(p).all() and (p>=0).all() for p in predictions.values()),'Invalid cached prediction values'
    return predictions

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

def new_fit(X,y,component,seed):
    model=CatBoostRegressor(**component['parameters'],random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    weights=None
    if 'support_columns' in component:
        weights=np.where(X[component['support_columns']].isna().any(axis=1),component['missing_weight'],1.)
    model.fit(X,np.asarray(y),cat_features=X.select_dtypes(include=['object','string']).columns.tolist(),sample_weight=weights)
    return model

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

def new_oof(train,y,scheme,component,seeds):
    X=feature_frame(train,component['variant']);individual=np.zeros((len(y),len(seeds)))
    for f,(tr,va) in enumerate(get_splits(train,scheme)):
        for j,seed in enumerate(seeds):
            model=new_fit(X.iloc[tr],y[tr],component,seed+f)
            individual[va,j]=np.maximum(model.predict(X.iloc[va]),0)
    return individual.mean(axis=1)

def compute_validation(train,y,scheme,contract,use_prior_cache=False):
    previous=BASE/'improvement_v3/results'/f'comparison_{scheme}.npz'
    if use_prior_cache and previous.exists():
        v3=checked_prior_v3(train,y,scheme,contract)
    else:v3=compute_v3(train,y,scheme,contract)
    pred=contract['v3_weight']*v3
    for name,component in contract['components'].items():
        p=new_oof(train,y,scheme,component,contract['seeds'])
        pred+=component['weight']*p
    return {'previous':v3,'candidate':pred}

def full_fit(train,test,y,contract):
    fitted_v2={};fitted_v3=[];fitted_new={}
    previous=np.zeros(len(test));idx=np.arange(len(train));vidx=np.arange(len(train),len(train)+len(test))
    combined_y=np.r_[y,np.full(len(test),np.nan)]
    for name,w in contract['v2_weights'].items():
        spec=contract['v2_configs'][name]
        X=pd.concat([features(train,spec['variant']),features(test,spec['variant'])],ignore_index=True)
        p,_,fitted=fit_predict(X,combined_y,idx,vidx,spec,42,spec.get('iterations'),return_fitted=True)
        fitted_v2[name]=fitted;previous+=.5*w*p
    X=features(train,'base');T=features(test,'base')
    for seed in contract['v3_seeds']:
        model=cat_fit(X,y,contract['v3_parameters'],seed);fitted_v3.append(model)
        previous+=.5*np.maximum(model.predict(T),0)/len(contract['v3_seeds'])
    pred=contract['v3_weight']*previous
    for name,component in contract['components'].items():
        X=feature_frame(train,component['variant']);T=feature_frame(test,component['variant']);fitted_new[name]=[]
        for seed in contract['seeds']:
            model=new_fit(X,y,component,seed);fitted_new[name].append(model)
            pred+=component['weight']*np.maximum(model.predict(T),0)/len(contract['seeds'])
    return previous,pred,{'v2':fitted_v2,'v3':fitted_v3,'new':fitted_new}

def predict_final(raw,fitted,contract):
    old=sum(.5*w*predict_fitted(fitted['v2'][name],features(raw,contract['v2_configs'][name]['variant'])) for name,w in contract['v2_weights'].items())
    X=features(raw,'base')
    old+=.5*np.mean([np.maximum(model.predict(X),0) for model in fitted['v3']],axis=0)
    p=contract['v3_weight']*old
    for name,component in contract['components'].items():
        X=feature_frame(raw,component['variant'])
        p+=component['weight']*np.mean([np.maximum(model.predict(X),0) for model in fitted['new'][name]],axis=0)
    return p

def check_submission(frame,test):
    assert frame.shape==(500,2),'Expected 500 rows and two columns'
    assert frame.columns.tolist()==['Registration Number','Annual Turnover'],'Incorrect header'
    assert frame['Registration Number'].equals(test['Registration Number']),'Restaurant IDs/order changed'
    assert frame['Registration Number'].is_unique,'Duplicate restaurant IDs'
    assert np.isfinite(frame['Annual Turnover']).all(),'Non-finite prediction'
    assert (frame['Annual Turnover']>=0).all(),'Negative turnover'
