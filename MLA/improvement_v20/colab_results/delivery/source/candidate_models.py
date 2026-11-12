"""Fixed TabICLv2 and clean-environment Kumo Small regression recipes."""
import hashlib, importlib, importlib.metadata, inspect, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from new_models import Codec,query_blocks
ROOT = Path(__file__).resolve().parent
SEED = 10242

def resources():
    import psutil
    return {'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
            'host_rss_bytes':psutil.Process().memory_info().rss}

def runtime():
    names=['torch','numpy','pandas','scikit-learn','scipy','catboost','tabicl',
           'structured-data-models','numba','cudf-cu13','huggingface-hub','safetensors','pyarrow','joblib']
    packages={}
    for name in names:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return {'dependencies':packages,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'python':__import__('sys').version,'total_gpu_bytes':torch.cuda.get_device_properties(0).total_memory,
            'cudf_discoverable':importlib.util.find_spec('cudf') is not None,
            'numba_discoverable':importlib.util.find_spec('numba') is not None}

def checked_checkpoint(name):
    from huggingface_hub import hf_hub_download
    key='I8' if name=='I16' else name
    record=json.loads((ROOT/'checkpoint_registry.json').read_text())['models'][key]
    path=Path(hf_hub_download(repo_id=record['model_repository'],filename=record['filename'],
                            revision=record['model_revision'],token=False))
    assert path.stat().st_size==record['advertised_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['advertised_sha256']
    return path,record

def source_check(name):
    registry=json.loads((ROOT/'source_registry.json').read_text())['model_modules']
    key='I8' if name=='I16' else name
    actual={}
    for module,expected in registry[key].items():
        path=Path(inspect.getfile(importlib.import_module(module)))
        actual[module]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual[module]==expected,'Model source mismatch: '+module
    return actual

def fit(raw,target,name):
    assert name in ['I8','KS8-CLEAN','I16']
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
    torch.manual_seed(SEED);np.random.seed(SEED);torch.cuda.reset_peak_memory_stats()
    hashes=source_check(name);path,checkpoint=checked_checkpoint(name)
    codec=Codec().fit(raw);y=np.asarray(target,float)/1e6
    assert y.shape==(len(raw),) and np.isfinite(y).all()
    if name in ['I8','I16']:
        from tabicl import TabICLRegressor
        assert importlib.metadata.version('tabicl')=='2.2.0'
        model=TabICLRegressor(n_estimators=8 if name=='I8' else 16,norm_methods=None,
            feat_shuffle_method='latin',batch_size=1,kv_cache=True,model_path=str(path),
            allow_auto_download=False,checkpoint_version=checkpoint['filename'],device='cuda',
            use_amp=False,use_fa3=False,random_state=SEED,n_jobs=2,verbose=False)
        typed=codec.typed(raw)
        model.fit(typed,y)
        padding_index=int(np.argmax(typed.notna().sum(axis=1).to_numpy()))
    else:
        import pyarrow as pa
        import sdm
        assert importlib.util.find_spec('cudf') is None and importlib.util.find_spec('numba') is None
        model=sdm.models.KumoTabular(task='regression',size='small',pretrained=False,device='cpu')
        weights=torch.load(path,map_location='cpu',weights_only=True)
        model.models[sdm.Task.regression].load_state_dict(weights,strict=True);del weights
        model=model.to('cuda').eval();frame=codec.typed(raw)
        stypes={c:'categorical' if c in codec.vocabulary else 'numerical' for c in frame}
        x=sdm.TableTensor.from_arrow(pa.Table.from_pandas(frame,preserve_index=False),stypes=stypes).to('cuda')
        yt=sdm.TableTensor.from_arrow(pa.table({'target':y}),stypes={'target':'numerical'}).to('cuda')
        with torch.inference_mode():
            model.fit(x=x,y=yt,num_estimators=8,estimator_batch_size=1,
                      generator=torch.Generator(device='cuda').manual_seed(SEED))
    metadata={'name':name,'training_rows':len(raw),'units':'raw INR millions',
        'model_class':type(model).__module__+'.'+type(model).__name__,
        'checkpoint_sha256':checkpoint['advertised_sha256'],'checkpoint_revision':checkpoint['model_revision'],
        'actual_source_hashes':hashes,'runtime':runtime(),'input_features':40,
        'categorical_columns':list(codec.vocabulary),'all_outer_training_rows':True,'seed':SEED,**resources()}
    assert len(codec.vocabulary)==8
    if name in ['I8','I16']:
        metadata['query_layout']='Original V11 fixed 500 rows, padded with the training feature row having the most observed fields; no labels or IDs.'
    return {'name':name,'model':model,'codec':codec,'metadata':metadata,
            'padding_raw':raw.iloc[[padding_index if name in ['I8','I16'] else 0]].drop(columns=['Annual Turnover','Registration Number'],errors='ignore').copy()}

def predict(fitted,raw):
    name,model,codec=fitted['name'],fitted['model'],fitted['codec']
    if name in ['I8','I16']:
        parts=[]
        for block,count in query_blocks(raw,fitted['padding_raw'],500):
            parts.append(np.asarray(model.predict(codec.typed(block),output_type='mean'),float)[:count])
        prediction=np.concatenate(parts)
    else:
        import pyarrow as pa
        import sdm
        frame=codec.typed(raw);stypes={c:'categorical' if c in codec.vocabulary else 'numerical' for c in frame}
        x=sdm.TableTensor.from_arrow(pa.Table.from_pandas(frame,preserve_index=False),stypes=stypes).to('cuda')
        with torch.inference_mode():result=model.predict(x)
        assert list(result.numerical_columns)==[f'q{i:03d}' for i in range(1,1000)]
        values=result.numerical.detach().cpu().numpy().astype(np.float64)
        assert values.shape==(len(raw),999)
        prediction=values.mean(axis=1)
    assert prediction.shape==(len(raw),) and np.isfinite(prediction).all()
    return np.maximum(prediction,0.)*1e6
