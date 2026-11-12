from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;OUT=ROOT/'results'
module_spec=importlib.util.spec_from_file_location('v4_pipeline',BASE/'improvement_v4/pipeline.py')
previous_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(previous_module)
for function_name in ['compute_v3','get_splits','features','fit_predict','predict_fitted','cat_fit','load_data','rmse','check_submission']:
    globals()[function_name]=getattr(previous_module,function_name)
sys.path.insert(0,str(ROOT))
from new_models import fit_v5,predict_v5,v5_features

def load_contract():return json.loads((OUT/'candidate_lock.json').read_text())

def source_hashes():
    names=['improvement_v2/experiment.py','improvement_v3/experiments.py','improvement_v4/pipeline.py','improvement_v5/new_models.py','improvement_v5/pipeline.py']
    return {name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in names}

def input_hashes():
    d=Path('~/Downloads')
    return {name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']}

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
