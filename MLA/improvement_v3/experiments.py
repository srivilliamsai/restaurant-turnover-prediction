"""Bounded additional experiments. All validation labels retain their original values."""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GroupKFold

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE/'improvement_v2'))
from experiment import features, load_data, rmse, fit_predict, predict_fitted, SOCIAL
OUT = BASE/'improvement_v3/results'
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads((BASE/'improvement_v2/results/locked_candidate.json').read_text())
V2_WEIGHTS = json.loads((BASE/'improvement_v2/results/final_execution.json').read_text())['weights']

# Chosen before seeing this round's scores: vary tree complexity, regularization,
# robust threshold, subsampling and ordered categorical statistics.
TRIALS = {
    'bag_h40_d5': dict(variant='base', depth=5, iterations=650, delta=40),
    'bag_h40_d6': dict(variant='base', depth=6, iterations=550, delta=40),
    'bag_h40_d4': dict(variant='base', depth=4, iterations=850, delta=40),
    'bag_h60_d5': dict(variant='base', depth=5, iterations=650, delta=60),
    'bag_h20_d5': dict(variant='base', depth=5, iterations=800, delta=20),
    'bag_mono_h40': dict(variant='base', depth=5, iterations=600, delta=40, monotone=True),
    'bag_mono_h60': dict(variant='base', depth=5, iterations=600, delta=60, monotone=True),
    'bag_enhanced_h40': dict(variant='engineered', depth=5, iterations=650, delta=40),
    'bag_drop_h40': dict(variant='engineered_drop', depth=5, iterations=650, delta=40),
    'bag_ordered_h40': dict(variant='base', depth=5, iterations=650, delta=40, ordered=True),
    'bag_regularized_h40': dict(variant='base', depth=6, iterations=850, delta=40, l2=20, random_strength=3),
    'bag_subsample_h40': dict(variant='base', depth=5, iterations=850, delta=40, subsample=.7, l2=10),
    'bag_cap120': dict(variant='base', depth=5, iterations=650, cap=120),
    'bag_raw_l2': dict(variant='base', depth=5, iterations=450, l2=20),
}

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

def split_indices(train, split_seed):
    if split_seed=='city':
        return list(GroupKFold(5).split(train,groups=features(train,'engineered')['City']))
    return list(KFold(5,shuffle=True,random_state=int(split_seed)).split(train))

def v2_oof(train, y, split_seed):
    old=BASE/'improvement_v2/results'
    file=OUT/f'v2_{split_seed}.npz'
    if file.exists(): return np.load(file)['oof']
    predictions=np.zeros(len(y))
    suffix=f'{split_seed}_fixed' if str(split_seed) in ['71','2026','303'] else 'city_canonical'
    splits=split_indices(train,split_seed)
    for name,w in V2_WEIGHTS.items():
        previous=old/f'{name}_{suffix}.npz'
        if (str(split_seed) in ['71','2026','303','city']) and previous.exists():
            component=np.load(previous)['oof']
        else:
            spec=LOCK['configs'][name]; X=features(train,spec['variant']); component=np.zeros(len(y))
            for f,(tr,va) in enumerate(splits):
                component[va],_=fit_predict(X,y,tr,va,spec,43+f,spec.get('iterations'))
        predictions+=w*component
    np.savez_compressed(file,oof=predictions)
    return predictions

def run_trial(train,y,name,split_seed=42,seeds=(1042,2042,3042)):
    spec=TRIALS[name]; X=features(train,spec['variant'])
    cache=OUT/f'{name}_{split_seed}.npz'
    if cache.exists(): return np.load(cache)['oof']
    pred=np.zeros((len(y),len(seeds)))
    for fold,(tr,va) in enumerate(split_indices(train,split_seed)):
        for j,seed in enumerate(seeds):
            fitted=cat_fit(X.iloc[tr],y[tr],spec,seed+fold)
            pred[va,j]=np.maximum(fitted.predict(X.iloc[va]),0)
    avg=pred.mean(axis=1)
    np.savez_compressed(cache,oof=avg,individual=pred,seeds=seeds)
    return avg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',default='42');p.add_argument('--models',nargs='*',default=list(TRIALS));args=p.parse_args()
    train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
    baseline=v2_oof(train,y,args.seed)
    print('V2',args.seed,rmse(y,baseline),flush=True)
    results=[]
    for name in args.models:
        start=time.time();pred=run_trial(train,y,name,args.seed)
        row=dict(name=name,split_seed=args.seed,rmse=rmse(y,pred),blend50_rmse=rmse(y,.5*baseline+.5*pred),mean=pred.mean(),seconds=time.time()-start)
        results.append(row); print(json.dumps(row),flush=True)
        pd.DataFrame(results).to_csv(OUT/f'cat_scores_{args.seed}.csv',index=False)
