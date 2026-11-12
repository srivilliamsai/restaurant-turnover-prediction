"""Four fixed V12 settings with shared atomic fits and pinned execution."""
import gc,importlib.metadata,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import features
from core import ROOT,PINNED,sha,save

WEIGHTS=[.25,.5,.75,1.]

def runtime():
    import torch
    versions={k:importlib.metadata.version(k) for k in PINNED}
    assert versions==PINNED,versions
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
    auxiliary={}
    for name in ['pytorch-lightning','lightning','torchmetrics','huggingface-hub']:
        try:auxiliary[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    return {'dependencies':versions,'auxiliary_dependencies':auxiliary,'gpu':'Tesla T4','cuda':torch.version.cuda,'precision':'float32','prediction_chunk':500}

def local_checkpoint(record):
    from huggingface_hub import hf_hub_download
    p=Path(hf_hub_download(repo_id=record['repo_id'],filename=record['filename'],revision=record['revision'],token=False))
    assert sha(p)==record['sha256'];return str(p)

def parameters(name):
    seed=10242 if name=='BASE' else int(name.split('_')[1])
    if name.startswith('R'):
        return dict(device='cuda',random_state=seed,n_cv=1,n_refit=1,n_repeats=1,n_threads=2,
                    n_epochs=256,hidden_sizes=[256,256,256],lr=.2,wd=.02,p_drop=.15,
                    train_metric_name='mse',val_metric_name='rmse',normalize_output=True,clamp_output=False,
                    verbosity=0,batch_size=256,predict_batch_size=500)
    return dict(n_estimators=32 if name.startswith('T32') else 16,random_state=seed,
                device='cuda',fit_mode='low_memory',memory_saving_mode=True)

def make_model(name,config,cats=()):
    if name.startswith('R'):
        from pytabkit import RealMLP_TD_Regressor
        return RealMLP_TD_Regressor(**parameters(name))
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2,
        model_path=local_checkpoint(config['checkpoint']),inference_precision=torch.float32,
        categorical_features_indices=list(cats),**parameters(name))

def serial(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(tuple,list)):return [serial(x) for x in value]
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    return str(value)

def resolve():
    config={'runtime':runtime(),'checkpoint':json.loads((ROOT/'reference_manifest.json').read_text())['execution_contract']['checkpoint'],
            'models':{},'engineering_failures':{},'selection_uses_validation_rmse':False,
            'inner_validation':{'fraction':.2,'seed':831,'full_training_refit':True},
            'adapter':'fixed 500-row chunks, padding with first training row, discard padded outputs',
            'target':'raw INR millions; no target removal, upper clipping or inverse correction'}
    for name in ['BASE','T32_10242']+[f'T16_{s}' for s in features.SEEDS[1:]]+[f'R_{s}' for s in features.SEEDS]:
        family='R' if name.startswith('R') else 'T'
        if family in config['engineering_failures']:continue
        try:
            model=make_model(name,config)
            params=model.get_config() if family=='R' else model.get_params(deep=False)
            params=serial(params);params.pop('categorical_features_indices',None)
            if family=='T':params['model_path']='checkpoint_sha256:'+config['checkpoint']['sha256']
            config['models'][name]=params;del model
        except Exception as error:
            config['engineering_failures'][family]={'type':type(error).__name__,'message':str(error)[:500]}
    save(ROOT/'resolved_models.json',config);return config

def fit_atomic(raw,y,name,progress=print):
    import torch
    config=json.loads((ROOT/'resolved_models.json').read_text());assert runtime()==config['runtime']
    assert name in config['models'],name
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(parameters(name)['random_state']);torch.cuda.manual_seed_all(parameters(name)['random_state'])
    builder=features.FeatureBuilder(name).fit(raw);X=builder.transform(raw)
    if name.startswith('R'):
        vocabulary=builder.vocabulary;cats=list(vocabulary);model=make_model(name,config)
        _,valid=train_test_split(np.arange(len(raw)),test_size=.2,random_state=831,shuffle=True)
    else:
        X,vocabulary=features.native_frame(X);cats=[X.columns.get_loc(c) for c in vocabulary]
        model=make_model(name,config,cats)
    target=np.asarray(y,float)/1e6
    assert len(target)==len(raw) and np.isfinite(target).all() and (target>0).all()
    if name.startswith('R'):
        model.fit(X,target,val_idxs=valid,cat_col_names=cats)
        assert hasattr(model,'refit_alg_interface_'),'RealMLP full-training refit missing'
    else:model.fit(X,target)
    progress(f'{name}: fitted {len(raw)} original targets; {len(X.columns)} predictor columns')
    return {'name':name,'builder':builder,'model':model,'vocabulary':vocabulary,'padding_row':X.iloc[[0]].copy()}

def predict_atomic(fitted,raw):
    X=fitted['builder'].transform(raw)
    if not fitted['name'].startswith('R'):X,_=features.native_frame(X,fitted['vocabulary'])
    parts=[]
    for start in range(0,len(X),500):
        block=X.iloc[start:start+500].reset_index(drop=True);size=len(block)
        if size<500:block=pd.concat([block,pd.concat([fitted['padding_row']]*(500-size),ignore_index=True)],ignore_index=True)
        p=fitted['model'].predict(block) if fitted['name'].startswith('R') else fitted['model'].predict(block,output_type='mean')
        parts.append(np.asarray(p,float).reshape(-1)[:size])
    p=np.concatenate(parts)*1e6
    if fitted['name']=='BASE':p=np.maximum(p,0)
    assert p.shape==(len(raw),) and np.isfinite(p).all();return p

def fit(raw,y,name,progress=print):
    result={'name':name,'atoms':[]}
    try:
        for atom in features.SPECS[name]['atoms']:result['atoms'].append(fit_atomic(raw,y,atom,progress))
        return result
    except BaseException:
        release(result);raise

def predict(fitted,raw):return np.mean([predict_atomic(a,raw) for a in fitted['atoms']],axis=0)

def combine(v7,t9,component,name,weight):
    p=.5*v7+.5*((1-weight)*t9+weight*component) if name.startswith('T') else (1-weight)*(.5*v7+.5*t9)+weight*component
    return np.maximum(p,0.)

def release(fitted=None):
    import torch
    if fitted is not None:
        for atom in fitted.get('atoms',[]):atom.clear()
        fitted.clear()
    gc.collect();torch.cuda.empty_cache()
