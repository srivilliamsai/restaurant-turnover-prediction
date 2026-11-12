"""Explicit official synthetic-pretrained checkpoints; fixed raw-target models."""
import gc,hashlib,importlib.metadata,inspect,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import ROOT,PINNED,sha,save,digest

SEED=10242
WEIGHTS=[.25,.5,.75,1.]
REPOS={'I':('jingang/TabICL','tabicl-regressor-v2-20260212.ckpt'),
       'N':('Prior-Labs/tabpfn_3_5','tabpfn-v3.5-20260909.safetensors'),
       'FALLBACK':('Prior-Labs/tabpfn_2_5','tabpfn-v2.5-regressor-v2.5_default.ckpt')}

def checkpoint(family):
    from huggingface_hub import HfApi,hf_hub_download
    repo,name=REPOS[family];revision=HfApi(token=False).model_info(repo_id=repo).sha
    p=Path(hf_hub_download(repo_id=repo,filename=name,revision=revision,token=False))
    return {'repo_id':repo,'filename':name,'revision':revision,'sha256':sha(p),'version':'2.5' if family=='FALLBACK' else ('3.5' if family=='N' else 'TabICLv2')}

def local_checkpoint(record):
    from huggingface_hub import hf_hub_download
    p=Path(hf_hub_download(repo_id=record['repo_id'],filename=record['filename'],revision=record['revision'],token=False))
    assert sha(p)==record['sha256'];return str(p)

def runtime():
    import torch
    versions={k:importlib.metadata.version(k) for k in PINNED}
    assert versions==PINNED,versions
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4','Free T4 contract required'
    return {'dependencies':versions,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'precision':'float32','seed':SEED,'target':'raw INR millions','prediction_chunk':500}

def make_model(name,config,categorical_indices=()):
    import torch
    n=16 if name=='BASE' else features.SPECS[name]['estimators'];family='BASE' if name=='BASE' else name[0]
    record=config['models'][family];path=local_checkpoint(record['checkpoint'])
    if family=='I':
        from tabicl import TabICLRegressor
        return TabICLRegressor(n_estimators=n,norm_methods=None,feat_shuffle_method='latin',batch_size=1,
            kv_cache=True,model_path=path,allow_auto_download=False,checkpoint_version=record['checkpoint']['filename'],
            device='cuda',use_amp=False,use_fa3=False,random_state=SEED,n_jobs=2,verbose=False)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    version={'2':'V2','2.5':'V2_5','3.5':'V3_5'}[record['checkpoint']['version']]
    kwargs=dict(model_path=path,n_estimators=n,random_state=SEED,device='cuda',fit_mode='low_memory',
                memory_saving_mode=True,inference_precision=torch.float32,categorical_features_indices=list(categorical_indices))
    if family!='BASE' and 'batch_size' in inspect.signature(TabPFNRegressor).parameters:kwargs['batch_size']=1
    return TabPFNRegressor.create_default_for_version(getattr(ModelVersion,version),**kwargs)

def serial_params(model,record):
    def convert(x):
        if x is None or isinstance(x,(str,int,float,bool)):return x
        if isinstance(x,(list,tuple)):return [convert(v) for v in x]
        if isinstance(x,dict):return {str(k):convert(v) for k,v in x.items()}
        return str(x)
    p=convert(model.get_params(deep=False));p['model_path']='checkpoint_sha256:'+record['checkpoint']['sha256']
    p.pop('categorical_features_indices',None)
    return p

def resolve():
    config={'runtime':runtime(),'models':{},'engineering_failures':{},'selection_uses_validation_rmse':False,
            'adapter':'Fixed 500 rows; BASE pads its first training row; new models pad a maximum-observed training row; discard padding predictions.'}
    v2=json.loads((ROOT/'reference_manifest.json').read_text())['execution_contract']['checkpoint']
    config['models']['BASE']={'checkpoint':{**v2,'version':'2'}}
    for family in ['I','N']:
        try:
            config['models'][family]={'checkpoint':checkpoint(family)}
            for n in [8,16]:
                model=make_model(family+str(n),config);config['models'][family]['params_'+str(n)]=serial_params(model,config['models'][family]);del model
            gc.collect()
        except Exception as error:
            config['engineering_failures'][family]={'error_type':type(error).__name__,'message':str(error)[:500]}
            config['models'].pop(family,None)
            if family=='N':
                try:
                    config['models']['N']={'checkpoint':checkpoint('FALLBACK'),'fallback_reason':config['engineering_failures']['N']}
                    for n in [8,16]:
                        model=make_model('N'+str(n),config);config['models']['N']['params_'+str(n)]=serial_params(model,config['models']['N']);del model
                except Exception as fallback_error:
                    config['engineering_failures']['N_fallback']={'error_type':type(fallback_error).__name__,'message':str(fallback_error)[:500]};config['models'].pop('N',None)
    save(ROOT/'resolved_models.json',config);return config

def fallback(config,reason):
    assert config['models']['N']['checkpoint']['version']=='3.5'
    config['engineering_failures']['N_pilot']=reason
    config['models']['N']={'checkpoint':checkpoint('FALLBACK'),'fallback_reason':reason}
    for n in [8,16]:
        model=make_model('N'+str(n),config);config['models']['N']['params_'+str(n)]=serial_params(model,config['models']['N']);del model
    save(ROOT/'resolved_models.json',config);return config

def fit(raw,y,name,progress=print):
    import torch
    config=json.loads((ROOT/'resolved_models.json').read_text());assert runtime()==config['runtime']
    torch.use_deterministic_algorithms(True)
    builder=features.FeatureBuilder(name).fit(raw);X=builder.transform(raw)
    categorical=name.startswith('I');X,vocab=features.native_frame(X,categorical=categorical)
    cats=[X.columns.get_loc(c) for c in vocab]
    model=make_model(name,config,cats);target=np.asarray(y,float)/1e6
    assert len(target)==len(raw) and np.isfinite(target).all() and (target>0).all()
    model.fit(X,target)
    padding_index=0 if name=='BASE' else int(np.argmax(X.notna().sum(axis=1).to_numpy()))
    progress(f'{name}: fitted {len(raw)} original targets; 40 unchanged V9 features')
    return {'name':name,'builder':builder,'vocabulary':vocab,'model':model,'padding_row':X.iloc[[padding_index]].copy(),'categorical':categorical}

def predict(fitted,raw):
    X,_=features.native_frame(fitted['builder'].transform(raw),fitted['vocabulary'],categorical=fitted['categorical']);parts=[]
    for start in range(0,len(X),500):
        block=X.iloc[start:start+500].reset_index(drop=True);size=len(block)
        if size<500:block=pd.concat([block,pd.concat([fitted['padding_row']]*(500-size),ignore_index=True)],ignore_index=True)
        p=np.asarray(fitted['model'].predict(block,output_type='mean'),float)
        parts.append(p[:size])
    p=np.maximum(np.concatenate(parts),0)*1e6
    assert p.shape==(len(raw),) and np.isfinite(p).all();return p

def release(fitted=None):
    import torch
    if fitted is not None:fitted.clear()
    gc.collect();torch.cuda.empty_cache()
