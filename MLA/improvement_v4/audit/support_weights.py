"""Bounded development-only covariate support experiment; validation labels unchanged."""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
sys.path.insert(0, str(BASE / 'improvement_v2'))
from experiment import features, load_data, rmse

MISSING_COLUMNS = ['Facebook Popularity Quotient', 'Instagram Popularity Quotient', 'Resturant Tier', 'Ambience']
SEEDS = [1042, 2042]
SETTINGS = [0.25, 0.5]
MODEL = dict(iterations=850, learning_rate=.04, depth=5, l2_leaf_reg=10,
             thread_count=2, verbose=False, allow_writing_files=False,
             loss_function='Huber:delta=4e+07', bootstrap_type='Bernoulli', subsample=.7)

def main():
    train, test = load_data()
    y = train['Annual Turnover'].to_numpy(float)
    X = features(train, 'base')
    cats = X.select_dtypes(include=['object', 'string']).columns.tolist()
    unsupported = train[MISSING_COLUMNS].isna().any(axis=1).to_numpy()
    assert not test[MISSING_COLUMNS].isna().any().any()
    folds = list(KFold(5, shuffle=True, random_state=42).split(train))
    base2 = np.load(BASE/'improvement_v3/results/v2_42.npz')['oof']
    base3 = np.load(BASE/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
    base = .5 * base2 + .5 * base3
    specs = dict(missing_columns=MISSING_COLUMNS, lower_weights=SETTINGS,
                 model=MODEL, model_seeds=SEEDS, split_seed=42,
                 seed_schedule='model_seed + zero_based_fold_index',
                 description='Only training weights change; every original validation label is scored.',
                 absent_pattern_train_rows=int(unsupported.sum()),
                 absent_pattern_test_rows=int(test[MISSING_COLUMNS].isna().any(axis=1).sum()),
                 validation_train_sha256=hashlib.sha256(Path('~/Downloads/Train_dataset.csv').read_bytes()).hexdigest())
    (ROOT/'support_specs.json').write_text(json.dumps(specs, indent=2)+'\n')
    rows=[]
    for low_weight in SETTINGS:
        name=f'support_weight_{str(low_weight).replace(".","p")}'
        out=ROOT/(name+'_42.npz')
        pred=np.zeros((len(y),len(SEEDS)))
        coverage=np.zeros(len(y),int)
        train_weights=np.where(unsupported,low_weight,1.)
        start=time.time()
        for f,(tr,va) in enumerate(folds):
            assert set(tr).isdisjoint(va)
            coverage[va]+=1
            for j,seed in enumerate(SEEDS):
                model=CatBoostRegressor(**MODEL, random_seed=seed+f)
                model.fit(X.iloc[tr],y[tr],cat_features=cats,sample_weight=train_weights[tr])
                pred[va,j]=np.maximum(model.predict(X.iloc[va]),0)
        assert (coverage==1).all() and np.isfinite(pred).all()
        avg=pred.mean(axis=1)
        np.savez_compressed(out,oof=avg,individual=pred,seeds=SEEDS,missing_pattern=unsupported)
        for w in [0.,.25,.5,1.]:
            p=(1-w)*base+w*avg
            row=dict(setting=name,new_weight=w,rmse=rmse(y,p),
                     gain_vs_v3=rmse(y,base)-rmse(y,p),
                     complete_support_rmse=rmse(y[~unsupported],p[~unsupported]),
                     complete_support_gain=rmse(y[~unsupported],base[~unsupported])-rmse(y[~unsupported],p[~unsupported]),
                     missing_pattern_rmse=rmse(y[unsupported],p[unsupported]))
            rows.append(row)
            print(json.dumps(row),flush=True)
        print(name,'seconds',round(time.time()-start,2),flush=True)
        pd.DataFrame(rows).to_csv(ROOT/'support_scores.csv',index=False)

if __name__=='__main__': main()
