"""Explicit train-only LimiX-2 regression adapter; frozen pretrained weights."""
import gc,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_sources/LimiX'))
LOCAL_MLX=json.loads((ROOT/'protocol.json').read_text()).get('backend')=='local_mlx'
import torch
if LOCAL_MLX:
    # Load Torch's optional compiler checks while Triton is genuinely absent.
    # The Mac compatibility shim is for the reference model imports only.
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(ROOT/'research_sources/mlx_port'))
    import limix_mlx.shim
from inference.v2_0.predictor import LimiXPredictor
from inference.v2_0.preprocess import FilterValidFeatures,CategoricalFeatureEncoder

class TrainOnlyFilter(FilterValidFeatures):
    def fit(self,x,categorical_features,seed,y=None,**kwargs):
        if y is None:raise ValueError('Training boundary required')
        support=np.asarray(x[:len(y)])
        result=super().fit(support,categorical_features,seed,y=None)
        valid=np.asarray(self.valid_features,bool)&~np.isnan(support).all(axis=0)
        self.valid_features=valid;self.invalid_indices=~valid
        self.categorical_idx=[i for i,j in enumerate(np.flatnonzero(valid)) if j in categorical_features]
        if not valid.any():raise ValueError('No supported features')
        return self.categorical_idx

class TrainOnlyCategorical(CategoricalFeatureEncoder):
    def fit_transform(self,x,categorical_features,seed,y=None,**kwargs):
        if y is None:raise ValueError('Training boundary required')
        n=len(y);self.category_mappings=None
        train,cats=super().fit_transform(x[:n],categorical_features,seed)
        query=np.asarray(x[n:]).copy()
        if self.transformer is not None:
            query=self.transformer.transform(query)
            if self.category_mappings:
                for col,perm in self.category_mappings.items():
                    valid=~np.isnan(query[:,col])
                    query[valid,col]=perm[query[valid,col].astype(int)]
        return np.concatenate([train,query]),cats

class TrainOnlyPredictor(LimiXPredictor):
    def _member_numeric_inputs(self,frame,*,train_size,cast_float32,task_type='Regression'):
        assert task_type=='Regression'
        values=np.asarray(frame,dtype=np.float32 if cast_float32 else np.float64)
        cats=self.get_categorical_features_indices(values[:train_size])
        return [(values,cats.copy()) for _ in self.inference_pipeline_config]

def vocabulary(frame):
    return {c:sorted(frame[c].dropna().astype(str).unique().tolist()) for c in frame.select_dtypes(include=['object','string','category'])}

def encode(frame,vocab):
    result=frame.copy()
    for c,values in vocab.items():
        result[c]=result[c].map({v:i for i,v in enumerate(values)}).astype(float)
    result=result.to_numpy(dtype=np.float64)
    assert not np.isinf(result).any()
    return result

def make_predictor(checkpoint):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=json.loads((ROOT/'research_sources/LimiX/config/reg_default_noretrieval_v2.json').read_text())
    assert len(config['pipelines'])==8
    # User's one-hour amendment, frozen before any accuracy results:
    # one official quantile/SVD member and one official power/one-hot member.
    config['pipelines']=[config['pipelines'][0],config['pipelines'][4]]
    options=dict(model_path=str(checkpoint),inference_config=config,mix_precision=False,seed=10242,
        deterministic=True,test_batch_mode='fixed',test_batch_size=json.loads((ROOT/'protocol.json').read_text())['query_batch_size'],
        enable_preprocess_parallel=False,preprocess_num_jobs=1,use_data_cache=False,cache_dir=str(ROOT/'local_cache'))
    if LOCAL_MLX:
        from limix_mlx.api import LimiXMLXPredictor
        import mlx.core as mx
        from mlx.utils import tree_flatten
        mx.set_memory_limit(3*1024**3)
        mx.set_cache_limit(128*1024**2)
        class LocalPredictor(LimiXMLXPredictor):
            CacheManager=LimiXPredictor.CacheManager
            _member_numeric_inputs=TrainOnlyPredictor._member_numeric_inputs
        model=LocalPredictor(npz_path=str(ROOT/'weights/LimiX-2.npz'),**options)
        weights=mx.load(str(ROOT/'weights/LimiX-2.npz'))
        have={k for k,_ in tree_flatten(model._mlx_model.parameters())}
        extra={'_reg_borders'}
        assert not (have-set(weights)-extra),'Missing MLX parameters'
        assert not (set(weights)-have-extra),'Unused checkpoint parameters'
        for name in ['_reg_borders','reg_log_widths']:
            if name in weights:setattr(model._mlx_model,name,weights[name])
        mx.eval(model._mlx_model.parameters())
    else:
        model=TrainOnlyPredictor(device=torch.device('cuda'),**options)
    for pipeline in model.preprocess_pipelines:
        for step in pipeline:
            if type(step) is FilterValidFeatures:step.__class__=TrainOnlyFilter
            elif type(step) is CategoricalFeatureEncoder:step.__class__=TrainOnlyCategorical
    return model

def fit_state(frame,y):
    assert len(frame)==len(y) and frame.shape[1]==40
    v=vocabulary(frame);x=encode(frame,v)
    target=np.asarray(y,np.float64)/1e6
    assert np.isfinite(target).all() and (target>0).all()
    return {'x':x,'y':target,'vocabulary':v,'columns':frame.columns.tolist()}

def infer(model,state,query):
    assert query.columns.tolist()==state['columns']
    q=encode(query,state['vocabulary'])
    p=np.asarray(model.predict(state['x'].copy(),state['y'].copy(),q,task_type='Regression'),float).reshape(-1)
    assert p.shape==(len(query),) and np.isfinite(p).all()
    assert len(model.preprocess_pipelines)==2
    assert not model.svd_runtime_retry_audit and not model.polynomial_interaction_runtime_retry_audit and not model.cuda_pipeline_fallback_audit
    return np.maximum(p,0.)*1e6

def save_state(state,path):
    np.savez_compressed(path,x=state['x'],y=state['y'])
    Path(path).with_suffix('.json').write_text(json.dumps({k:state[k] for k in ['vocabulary','columns']},sort_keys=True)+'\n')

def load_state(path):
    result=json.loads(Path(path).with_suffix('.json').read_text())
    with np.load(path,allow_pickle=False) as z:result.update(x=z['x'].copy(),y=z['y'].copy())
    return result

def release(model):
    model.model=None
    if LOCAL_MLX:
        import mlx.core as mx
        model._mlx_model=None
        mx.clear_cache()
    gc.collect();torch.cuda.empty_cache()

def preprocess_audit(model,state,query):
    """Different query values and composition cannot alter fitted support outputs."""
    from inference.v2_0.target_transform_preprocess import TargetTransform
    train=state['x'];q=encode(query,state['vocabulary']);target=state['y']
    y=(target-target.mean())/target.std(ddof=1)
    evidence=[]
    for index,pipeline in enumerate(model.preprocess_pipelines):
        outs=[];widths=[]
        for test in [q,np.full_like(q,99999.),q[:1]]:
            x=np.concatenate([train,test]);cats=model.get_categorical_features_indices(train)
            for j,step in enumerate(pipeline):
                assert not isinstance(step,TargetTransform),'No target transform in frozen config'
                x,cats=step.fit_transform(x,cats,model.seeds[index*model.preprocess_num+j],y=y)
            outs.append(np.asarray(x[:len(y)]).copy());widths.append(x.shape[1])
        assert widths==[widths[0]]*3
        for output in outs[1:]:np.testing.assert_array_equal(outs[0],output)
        evidence.append({'member':index,'support_rows':len(y),'output_columns':widths[0],'support_invariant_to_query':True})
    unknown=query.copy()
    for c in state['vocabulary']:unknown[c]='__V23_UNSEEN_CATEGORY__'
    u=encode(unknown,state['vocabulary'])
    assert np.isnan(u[:,[query.columns.get_loc(c) for c in state['vocabulary']]]).all()
    return {'members':evidence,'unseen_categories_map_to_nan':True,'feature_count':40,'target_and_ID_absent':not {'Registration Number','Annual Turnover'}&set(query.columns)}
