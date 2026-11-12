"""The fixed V18 recipes through supported author APIs."""
import gc,hashlib,inspect,json,os,random,sys,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
import features
from core import ROOT,sha,digest,save

def source_paths():
    path=ROOT/'engineering_sources/exaone/src'
    if str(path) not in sys.path:sys.path.insert(0,str(path))

def deterministic():
    import torch
    random.seed(10242);np.random.seed(10242);torch.manual_seed(10242);torch.cuda.manual_seed_all(10242)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)

class Codec:
    """Frozen training-only categorical maps; numerical missingness is preserved."""
    def fit(self, raw):
        X = features.frame(raw)
        self.columns = X.columns.tolist()
        self.vocabulary = {c: {v: i for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                           for c in X.select_dtypes(include=['object','string','category'])}
        self.categorical_indices = [self.columns.index(c) for c in self.vocabulary]
        self.rows = len(X)
        return self

    def transform(self, raw):
        X = features.frame(raw)
        assert X.columns.tolist() == self.columns
        for c, values in self.vocabulary.items():
            X[c] = X[c].astype('string').map(values).astype(float)
        return X.to_numpy(dtype=np.float64)


def inner_indices(raw, fold=0, scheme='full'):
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    indices = np.arange(len(raw))
    seed = 9091 + fold
    if scheme == 'city':
        groups = features.canonical_city(raw['City']).fillna('__V10_UNKNOWN_CITY__')
        a, b = next(GroupShuffleSplit(1, test_size=.15, random_state=seed).split(raw, groups=groups))
        assert set(groups.iloc[a]).isdisjoint(groups.iloc[b])
    else:
        a, b = train_test_split(indices, test_size=.15, random_state=seed)
    return np.sort(a), np.sort(b)



class EpochLogger:
    def __init__(self):self.epochs=[];self.epoch_seconds=[];self.last=time.monotonic();self.steps=0
    def setup(self,config):pass
    def log_step(self,metrics,step):self.steps+=1
    def log_epoch(self,metrics,step):
        if 'val/MSE' in metrics:
            self.epochs.append({'epoch':int(metrics['train/epoch'])+1,'mse_millions_squared':float(metrics['val/MSE'])})
        if 'train/epoch' in metrics:
            now=time.monotonic();self.epoch_seconds.append(now-self.last);self.last=now
            path=ROOT/'results/current_fit_progress.json'
            save(path,{'epoch':int(metrics['train/epoch'])+1,'optimizer_steps':self.steps,'time':time.time()})
        deadline=os.environ.get('MLA_V18_TRAINING_DEADLINE')
        if deadline and time.time()>float(deadline):raise TimeoutError('V18 research time exhausted during fine-tuning')
    def finish(self):pass

def checkpoint_exa():
    from huggingface_hub import hf_hub_download
    record=json.loads((ROOT/'checkpoint_registry.json').read_text())['EXA8']
    target=ROOT/'model_weights'/record['filename'];target.parent.mkdir(exist_ok=True)
    if not target.exists():
        import shutil
        path=hf_hub_download(repo_id=record['repo_id'],filename=record['filename'],revision=record['revision'],token=False)
        shutil.copyfile(path,target)
    assert sha(target)==record['sha256']
    return str(target)

def original_model(path,cats):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,model_path=path,n_estimators=1,
        random_state=10242,device='cuda',fit_mode='low_memory',inference_precision=torch.float32,
        memory_saving_mode=True,categorical_features_indices=cats)

def ft_kwargs(path,cats,chunk):
    import torch
    from tabpfn.constants import ModelVersion
    return dict(device='cuda',learning_rate=3e-6,use_lr_scheduler=False,weight_decay=.01,
        n_finetune_ctx_plus_query_samples=chunk,finetune_ctx_query_split_ratio=.2,n_inference_subsample_samples=None,
        random_state=10242,early_stopping_patience=8,min_delta=0.,grad_clip_value=1.,
        n_estimators_finetune=1,n_estimators_validation=1,n_estimators_final_inference=1,
        use_activation_checkpointing=True,use_fixed_preprocessing_seed=True,save_checkpoint_interval=None,
        ce_loss_weight=0.,crps_loss_weight=1.,crls_loss_weight=0.,mse_loss_weight=1.,mse_loss_clip=None,
        mae_loss_weight=0.,eval_metric='mse',model_version=ModelVersion.V3_5,
        extra_regressor_kwargs={'model_path':path,'inference_precision':torch.float32,
            'categorical_features_indices':cats,'memory_saving_mode':True})

def fit(raw,y,name,fold=0,scheme='full'):
    source_paths();deterministic()
    (ROOT/'results').mkdir(exist_ok=True)
    import torch
    started=time.monotonic();codec=Codec().fit(raw);X=codec.transform(raw)
    target=np.asarray(y,dtype=float)/1e6
    assert len(target)==len(raw) and np.isfinite(target).all()
    result={'name':name,'codec':codec,'padding_row':X[[0]],'training_X':X,'training_y':target}
    meta={'name':name,'training_rows':len(raw),'units':'raw INR millions','all_original_training_targets_retained':True,
        'seed':10242,'training_target_sha256':hashlib.sha256(np.asarray(y,dtype=float).tobytes()).hexdigest(),
        'categorical_columns':list(codec.vocabulary),'features':len(codec.columns)}
    if name=='EXA8':
        from dataclasses import replace
        from exaonetabular import EXAONETabularRegressor
        from exaonetabular.presets import released_manifest
        path=checkpoint_exa();manifest=released_manifest('regression')
        manifest=replace(manifest,runtime=replace(manifest.runtime,ensemble_count=8,compute_dtype='float32',seed=10242,
            support_row_limit=100000,feature_limit=500,query_batch_limit=500),
            regression=replace(manifest.regression,point_estimate='mean',member_weighting='uniform',n_svd=0,
                svd_gate=False,svd_split=False),checkpoint_sha256=sha(path))
        free,_=torch.cuda.mem_get_info(0);memory=min(12<<30,free)
        model=EXAONETabularRegressor.from_pretrained(weights=path,manifest=manifest,device='cuda:0',max_vram_bytes=memory)
        mask=np.zeros(X.shape[1],bool);mask[codec.categorical_indices]=True
        model.fit(X,target,categorical_mask=mask)
        # Inspect fitted state for evidence only; inference remains public predict.
        assert len(model._fitted_state['support_y'])==len(raw)
        assert model._fitted_state['member_weights'] is None
        meta.update(actual_estimators=8,precision='float32',checkpoint_sha256=sha(path),internal_selection=False,
            support_rows=len(raw),memory_budget_bytes=memory,manifest=manifest.to_dict())
    else:
        import frozen_pfn
        config=json.loads((ROOT/'resolved_models.json').read_text());path=frozen_pfn.local_checkpoint(config['models']['P35_8']['checkpoint'])
        if name=='UNADAPTED':
            model=original_model(path,codec.categorical_indices);model.fit(X,target)
            meta.update(actual_estimators=1,precision='float32',checkpoint_sha256=sha(path),diagnostic_only=True)
        else:
            assert name in ['FT512','FT1024'];chunk=features.SPECS[name]['context_plus_query_rows']
            from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
            release=json.loads((ROOT/'release_review_registry.json').read_text())['tabpfn_9_2_0']['files']
            assert sha(inspect.getfile(FinetunedTabPFNRegressor))==release['src/tabpfn/finetuning/finetuned_regressor.py']['sha256']
            a,b=inner_indices(raw,fold,scheme);inner=Codec().fit(raw.iloc[a]);inner_X=inner.transform(raw.iloc[a]);val=inner.transform(raw.iloc[b])
            original=original_model(path,inner.categorical_indices);original.fit(inner_X,target[a])
            arch=[type(m).__module__+'.'+type(m).__name__ for m in original.models_]
            assert any('v3_5' in item or '3p5' in item.lower() for item in arch)
            initial_mse=float(np.mean((target[b]-np.asarray(original.predict(val,output_type='mean'),float))**2))
            del original;gc.collect();torch.cuda.empty_cache()
            setup_seconds=time.monotonic()-started;logger=EpochLogger()
            selector=FinetunedTabPFNRegressor(epochs=30,early_stopping=True,validation_split_ratio=None,experiment_logger=logger,
                **ft_kwargs(path,inner.categorical_indices,chunk))
            selection_started=time.monotonic()
            # Official epoch checkpoints are retained while fit is active. Interrupted fits are replayed
            # from the original checkpoint: the wrapper does not persist all early-stopping history.
            with tempfile.TemporaryDirectory(prefix='v18_active_select_',dir=ROOT/'results') as directory:
                selector.fit(inner_X,target[a],X_val=val,y_val=target[b],output_dir=Path(directory))
            selection_seconds=time.monotonic()-selection_started
            history=[{'epoch':0,'mse_millions_squared':initial_mse},*logger.epochs]
            assert any(item['epoch']>0 for item in history),'No completed inner epochs'
            selected=min(history,key=lambda item:(item['mse_millions_squared'],item['epoch']))['epoch']
            del selector;gc.collect();torch.cuda.empty_cache();deterministic()
            refit_started=time.monotonic()
            if selected==0:model=original_model(path,codec.categorical_indices);model.fit(X,target)
            else:
                model=FinetunedTabPFNRegressor(epochs=selected,early_stopping=False,validation_split_ratio=None,
                    **ft_kwargs(path,codec.categorical_indices,chunk))
                with tempfile.TemporaryDirectory(prefix='v18_active_refit_',dir=ROOT/'results') as directory:
                    model.fit(X,target,output_dir=Path(directory))
            refit_seconds=time.monotonic()-refit_started
            meta.update(inner_training_rows=len(a),inner_validation_rows=len(b),inner_seed=9091+fold,
                inner_train_sha256=digest(a.tolist()),inner_validation_sha256=digest(b.tolist()),
                inner_policy='city-disjoint' if scheme=='city' else 'random-15-percent',selected_epochs=selected,
                inner_epochs_run=max(v['epoch'] for v in history),inner_history=history,outer_validation_used_for_epoch_selection=False,
                precision='CUDA-AMP-training/float32-inference',checkpoint_sha256=sha(path),matched_original_architectures=arch,
                installed_finetune_source_sha256=sha(inspect.getfile(FinetunedTabPFNRegressor)),
                learning_rate=3e-6,use_lr_scheduler=False,setup_seconds=setup_seconds,selection_seconds=selection_seconds,
                refit_seconds=refit_seconds,inner_optimizer_steps=logger.steps,full_subset_refitted=True,actual_estimators=1,
                context_plus_query_rows=chunk,matched_unadapted_inner_mse_millions_squared=initial_mse)
    result['model']=model;meta['fit_seconds']=time.monotonic()-started;result['target_metadata']=meta
    return result

def predict(fitted,raw):
    X=fitted['codec'].transform(raw);output=[];started=time.monotonic()
    for start in range(0,len(X),500):
        block=X[start:start+500];size=len(block)
        if size<500:block=np.concatenate([block,np.repeat(fitted['padding_row'],500-size,axis=0)])
        if fitted['name']=='EXA8':p=fitted['model'].predict(block)
        else:p=fitted['model'].predict(block,output_type='mean')
        p=np.asarray(p,float);assert p.shape==(500,) and np.isfinite(p).all();output.append(p[:size])
    fitted['target_metadata']['last_query_seconds']=time.monotonic()-started
    result=np.maximum(np.concatenate(output),0.)*1e6
    assert result.shape==(len(raw),) and np.isfinite(result).all()
    return result
