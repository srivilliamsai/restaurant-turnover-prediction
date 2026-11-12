"""The three declared V17 recipes, using public author interfaces only."""
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import features
from core import ROOT, sha, digest


def source_paths():
    for path in [ROOT/'engineering_sources/xiaomi', ROOT/'engineering_sources/nori/src']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def deterministic():
    import torch
    random.seed(10242)
    np.random.seed(10242)
    torch.manual_seed(10242)
    torch.cuda.manual_seed_all(10242)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    os.environ['SYNTHEFY_DISABLE_PIPELINE_BATCHING'] = '1'
    os.environ['SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE'] = '0'
    os.environ['SYNTHEFY_DISABLE_CACHED_INFERENCE'] = '1'


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


def checkpoint(name):
    from huggingface_hub import hf_hub_download
    record = json.loads((ROOT/'checkpoint_registry.json').read_text())[name]
    target = ROOT/'model_weights'/name/Path(record['filename']).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        import shutil
        path = hf_hub_download(repo_id=record['repo_id'], filename=record['filename'],
                               revision=record['revision'], token=False)
        shutil.copyfile(path, target)
    assert sha(target) == record['sha256']
    return str(target)


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
    """Public fine-tuning logging protocol, retaining original-scale MSE history."""
    def __init__(self):
        self.epochs = []
        self.steps = 0
    def setup(self, config):
        pass
    def log_step(self, metrics, step):
        self.steps += 1
    def log_epoch(self, metrics, step):
        if 'val/MSE' in metrics:
            self.epochs.append({'epoch':int(metrics['train/epoch'])+1,
                                'mse_millions_squared':float(metrics['val/MSE'])})
    def finish(self):
        pass


def ft_kwargs(path, cats):
    import torch
    from tabpfn.constants import ModelVersion
    return dict(device='cuda', learning_rate=1e-5, weight_decay=.01,
                n_finetune_ctx_plus_query_samples=512, finetune_ctx_query_split_ratio=.2,
                n_inference_subsample_samples=None, random_state=10242,
                early_stopping_patience=8, min_delta=0., grad_clip_value=1.,
                n_estimators_finetune=1, n_estimators_validation=1,
                n_estimators_final_inference=1, use_activation_checkpointing=True,
                use_fixed_preprocessing_seed=True, save_checkpoint_interval=None,
                ce_loss_weight=0., crps_loss_weight=1., crls_loss_weight=0.,
                mse_loss_weight=1., mse_loss_clip=None, mae_loss_weight=0.,
                eval_metric='mse', model_version=ModelVersion.V3_5,
                extra_regressor_kwargs={'model_path':path, 'inference_precision':torch.float32,
                                        'categorical_features_indices':cats,
                                        'memory_saving_mode':True})


def fit(raw, y, name, fold=0, scheme='full'):
    source_paths()
    deterministic()
    import torch
    started = time.monotonic()
    codec = Codec().fit(raw)
    X = codec.transform(raw)
    target = np.asarray(y, dtype=float)/1e6
    assert len(target) == len(raw) and np.isfinite(target).all()
    result = {'name':name, 'codec':codec, 'padding_row':X[[0]], 'training_X':X,
              'training_y':target}
    metadata = {'name':name,'training_rows':len(raw),'units':'raw INR millions',
                'all_original_training_targets_retained':True,'seed':10242,
                'training_target_sha256':hashlib.sha256(np.asarray(y,dtype=float).tobytes()).hexdigest(),
                'categorical_columns':list(codec.vocabulary),'features':len(codec.columns)}
    if name == 'XIA32':
        from tabldm import TabLDMRegressor
        model = TabLDMRegressor(n_estimators=32,norm_methods=['none','power'],
            feat_shuffle_method='random',batch_size=1,kv_cache=False,model_path=checkpoint(name),
            allow_auto_download=False,device='cuda:0',use_amp=False,use_fa3=False,
            offload_mode='auto',random_state=10242,enhance_candidates=False,
            enable_high_kurtosis_target_ensemble=False)
        model.fit(X, target)
        assert model.n_samples_in_ == len(raw) and model.nnls_weights_ is None
        metadata.update(actual_estimators=32,precision='float32',internal_selection=False,
                        checkpoint_sha256=sha(checkpoint(name)))
        result['model'] = model
    elif name == 'NOR15':
        from synthefy_nori.inference.predictor import NoriPredictor
        config = json.loads((ROOT/'nori_inference.json').read_text())
        assert len(config) == 4
        assert all(not item.get('HighDimFeatureSelector',{}).get('fit_on_test',False) for item in config)
        model = NoriPredictor(device=torch.device('cuda:0'),model_path=checkpoint(name),
            inference_config=config,mix_precision=False,seed=10242,augmentations=(),
            quantile_collapse='mean',bar_point_estimator='mean',discrete_y_snap_max_unique=0,
            categorical_features_indices=codec.categorical_indices,
            memory_policy={'allow_subsample':False,'cache':False,'elements_budget':8000000})
        assert model.n_estimators == 4 and not model.mix_precision
        mean, scale = float(target.mean()), max(float(target.std()),1e-12)
        result.update(model=model, target_mean=mean,target_scale=scale,
                      training_y=(target-mean)/scale)
        metadata.update(actual_estimators=4,precision='float32',internal_selection=False,
                        public_interface='NoriPredictor',train_only_config_sha256=sha(ROOT/'nori_inference.json'),
                        checkpoint_sha256=sha(checkpoint(name)))
    else:
        assert name == 'P35FT512'
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
        import frozen_pfn
        config = json.loads((ROOT/'resolved_models.json').read_text())
        path = frozen_pfn.local_checkpoint(config['models']['P35_8']['checkpoint'])
        a,b = inner_indices(raw,fold,scheme)
        inner_codec = Codec().fit(raw.iloc[a])
        inner_X, inner_valid = inner_codec.transform(raw.iloc[a]), inner_codec.transform(raw.iloc[b])
        cats = inner_codec.categorical_indices
        original = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,
            model_path=path,n_estimators=1,random_state=10242,device='cuda',fit_mode='low_memory',
            inference_precision=torch.float32,memory_saving_mode=True,categorical_features_indices=cats)
        original.fit(inner_X,target[a])
        architectures=[type(m).__module__+'.'+type(m).__name__ for m in original.models_]
        assert any(any(tag in item.lower() for tag in ['v3_5','v3p5','3p5']) for item in architectures), architectures
        metadata['matched_original_architectures']=architectures
        import inspect
        metadata['installed_finetune_source_sha256']=sha(inspect.getfile(FinetunedTabPFNRegressor))
        init = np.asarray(original.predict(inner_valid,output_type='mean'),float)
        initial_mse = float(np.mean((target[b]-init)**2))
        del original
        gc.collect();torch.cuda.empty_cache()
        logger = EpochLogger()
        selector = FinetunedTabPFNRegressor(epochs=30,early_stopping=True,
            validation_split_ratio=None,experiment_logger=logger,**ft_kwargs(path,cats))
        with tempfile.TemporaryDirectory(prefix='v17_ft_select_') as directory:
            selector.fit(inner_X,target[a],X_val=inner_valid,y_val=target[b],output_dir=Path(directory))
        assert logger.epochs, 'No completed inner epoch metrics recorded'
        history = [{'epoch':0,'mse_millions_squared':initial_mse},*logger.epochs]
        selected = min(history,key=lambda item:(item['mse_millions_squared'],item['epoch']))['epoch']
        metadata.update(inner_training_rows=len(a),inner_validation_rows=len(b),inner_seed=9091+fold,
            inner_train_sha256=digest(a.tolist()),inner_validation_sha256=digest(b.tolist()),
            inner_policy='city-disjoint' if scheme=='city' else 'random-15-percent',
            selected_epochs=selected,inner_epochs_run=len(logger.epochs),inner_history=history,
            outer_validation_used_for_epoch_selection=False,precision='CUDA-AMP-training/float32-inference',
            checkpoint_sha256=sha(path),matched_unadapted_inner_mse_millions_squared=initial_mse)
        del selector
        gc.collect();torch.cuda.empty_cache();deterministic()
        if selected == 0:
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,
                model_path=path,n_estimators=1,random_state=10242,device='cuda',fit_mode='low_memory',
                inference_precision=torch.float32,memory_saving_mode=True,categorical_features_indices=codec.categorical_indices)
            model.fit(X,target)
        else:
            model = FinetunedTabPFNRegressor(epochs=selected,early_stopping=False,
                validation_split_ratio=None,**ft_kwargs(path,codec.categorical_indices))
            with tempfile.TemporaryDirectory(prefix='v17_ft_refit_') as directory:
                model.fit(X,target,output_dir=Path(directory))
        result['model'] = model
        metadata.update(full_subset_refitted=True,actual_estimators=1)
    metadata['fit_seconds'] = time.monotonic()-started
    result['target_metadata'] = metadata
    return result


def predict(fitted,raw):
    X = fitted['codec'].transform(raw)
    output = []
    for start in range(0,len(X),500):
        block = X[start:start+500]
        size = len(block)
        if size < 500:
            block = np.concatenate([block,np.repeat(fitted['padding_row'],500-size,axis=0)])
        if fitted['name'] == 'NOR15':
            from synthefy_nori import strict_pipeline
            with strict_pipeline():
                p = fitted['model'].predict(fitted['training_X'],fitted['training_y'],block)
            if hasattr(p,'detach'):
                p = p.detach().cpu().numpy()
            p = np.asarray(p,float)*fitted['target_scale']+fitted['target_mean']
            report = fitted['model'].memory_report_
            if report is not None:
                dropped = report.get('dropped_context_rows',0) if isinstance(report,dict) else getattr(report,'dropped_context_rows',0)
                assert not dropped,'Context rows dropped'
        else:
            p = fitted['model'].predict(block,output_type='mean')
        p = np.asarray(p,dtype=float).reshape(-1)
        assert p.shape == (500,) and np.isfinite(p).all()
        output.append(p[:size])
    return np.maximum(np.concatenate(output),0.)*1e6
