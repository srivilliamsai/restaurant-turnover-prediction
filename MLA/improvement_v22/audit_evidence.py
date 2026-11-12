"""V22 planning audit: saved V14 predictions and target-free domain diagnostics.

No turnover estimator is trained. Run from the project root with:
PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v22/audit_evidence.py
"""
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V21 = PROJECT / 'improvement_v21'
SC = ['Service', 'Comfortablility']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.unique(np.r_[a, b])
    return float(np.max(abs(np.searchsorted(a, grid, side='right') / len(a)
                            - np.searchsorted(b, grid, side='right') / len(b))))


def domain_frame(raw):
    result = raw.drop(columns=['Registration Number', 'Annual Turnover'], errors='ignore').copy()
    date = pd.to_datetime(result.pop('Opening Day of Restaurant'), dayfirst=True, errors='coerce')
    for name, values in [('year', date.dt.year), ('month', date.dt.month), ('day', date.dt.day)]:
        result['opening_' + name] = values
    for column in result.select_dtypes(include=['object', 'string']):
        result[column] = result[column].fillna('__missing__').astype(str).str.strip().str.lower()
    assert not {'Annual Turnover', 'Registration Number'} & set(result)
    return result


def estimator(frame, family):
    categorical = frame.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric = [c for c in frame if c not in categorical]
    pre = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                                keep_empty_features=True), StandardScaler()), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ], sparse_threshold=0)
    if family == 'logistic':
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=2201)
    else:
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
                    max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=10,
                    early_stopping=False, random_state=2201)
    return make_pipeline(pre, model)


def main():
    started = time.monotonic()
    registry = read(V21 / 'baseline_registry.json')
    inputs = {name: sha(PROJECT / name) for name in registry['inputs']}
    assert inputs == registry['inputs']
    train = pd.read_csv(PROJECT / 'Train_dataset.csv').rename(columns={'Endorsed By': 'Endoresed By'})
    test = pd.read_csv(PROJECT / 'Test_dataset.csv')
    y = train['Annual Turnover'].to_numpy(float)
    ids = train['Registration Number'].tolist()
    assert len(train) == 3493 and len(test) == 500 and digest(ids) == registry['row_ids_sha256']
    assert train['Registration Number'].is_unique and test['Registration Number'].is_unique
    assert set(ids).isdisjoint(test['Registration Number']) and np.isfinite(y).all()
    refs = {}
    portal = read(V21 / 'reference/v14_portal_trial.json')
    for ext, key in [('csv', 'csv_sha256'), ('ipynb', 'notebook_sha256')]:
        path = V21 / f'reference/scored_v14.{ext}'
        assert sha(path) == portal[key]
        refs[str(path.relative_to(PROJECT))] = sha(path)

    X = train.drop(columns=['Registration Number', 'Annual Turnover'])
    Z = test.drop(columns='Registration Number')
    assert X.columns.tolist() == Z.columns.tolist()
    stats = {}
    for c in X.select_dtypes(include='number'):
        a, b = X[c].dropna().to_numpy(float), Z[c].dropna().to_numpy(float)
        stats[c] = dict(train_mean=float(a.mean()), test_mean=float(b.mean()),
                        train_missing=int(X[c].isna().sum()), test_missing=int(Z[c].isna().sum()),
                        KS=ks(a, b), train_observed=int(len(a)), test_observed=int(len(b)))
    sources = read(V21 / 'provenance/v14_source_frozen.json')['sources']
    controls, witnesses, control_metrics = {}, [], {}
    for seed in [42, 71]:
        entry = registry['schemes'][str(seed)]
        path = V21 / entry['file']
        assert sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as data:
            v9 = data['v9'].copy()
            assert np.max(abs(v9 - .5*data['v7'] - .5*data['t9'])) < .01
        p35, counts = np.zeros(len(y)), np.zeros(len(y), int)
        for fold, group in enumerate(np.array_split(np.random.RandomState(seed).permutation(len(y)), 5)):
            b = np.sort(group)
            a = np.setdiff1d(np.arange(len(y)), b)
            path = V21 / f'provenance/v14_cache/P35_8_{seed}_fold{fold}.npz'
            meta = read(path.with_suffix('.json'))
            ctx = meta['context']
            assert sha(path) == meta['sha256']
            assert ctx['inputs'] == inputs and ctx['sources'] == sources and ctx['name'] == 'P35_8'
            assert ctx['train_indices'] == digest(a.tolist()) and ctx['valid_indices'] == digest(b.tolist())
            assert ctx['training_targets'] == hashlib.sha256(y[a].tobytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                values = data['prediction'].copy()
            assert values.shape == b.shape and np.isfinite(values).all()
            p35[b] = values
            counts[b] += 1
            witnesses.append(dict(file=str(path.relative_to(PROJECT)), sha256=meta['sha256']))
        assert (counts == 1).all()
        pred = controls[seed] = .25*v9 + .75*p35
        se = (y-pred)**2
        order = np.argsort(-se)
        control_metrics[str(seed)] = dict(rmse_INR=rmse(y, pred),
            tail_gt80M_rows=int((y > 80e6).sum()),
            tail_gt80M_squared_error_share=float(se[y > 80e6].sum()/se.sum()),
            largest_five_squared_error_share=float(se[order[:5]].sum()/se.sum()),
            largest_five_ids=train.iloc[order[:5]]['Registration Number'].tolist())
    squared = (y-controls[42])**2
    rng = np.random.default_rng(2202)
    draws = np.array([np.sqrt(squared[rng.choice(len(y), 500, replace=False)].mean()) for _ in range(10000)])
    simulation = dict(draws=10000, sample_size=500, seed=2202,
        RMSE_INR_quantiles=dict(zip(['p01','p05','p50','p95','p99'], np.quantile(draws,[.01,.05,.5,.95,.99]).tolist())),
        fraction_at_or_below_historical_portal=float(np.mean(draws <= 11736110.7877709)),
        limitation='Illustrative sampling of fixed assignment-42 OOF residuals only. Not a hidden-test estimate, rank forecast, confidence interval, or identification of the portal sampling mechanism.')

    pooled = pd.concat([domain_frame(train), domain_frame(test)], ignore_index=True)
    labels = np.r_[np.zeros(len(train),int), np.ones(len(test),int)]
    masks = pooled.isna().astype(float).add_suffix('_missing')
    masks = masks.loc[:, masks.nunique() > 1]
    views = {'ALL': pooled, 'NO_SC': pooled.drop(columns=SC), 'SC_ONLY': pooled[SC], 'MISSINGNESS_ONLY': masks}
    split = list(StratifiedKFold(5, shuffle=True, random_state=2201).split(pooled, labels))
    domain, pred_arrays = {}, {}
    for view, frame in views.items():
        domain[view] = {}
        for family in ['logistic', 'hist_gradient_boosting']:
            pred = np.zeros(len(pooled))
            folds = []
            for a, b in split:
                model = estimator(frame, family)
                model.fit(frame.iloc[a], labels[a])
                pred[b] = model.predict_proba(frame.iloc[b])[:,1]
                folds.append(float(roc_auc_score(labels[b], pred[b])))
            domain[view][family] = dict(OOF_AUC=float(roc_auc_score(labels,pred)), fold_AUC=folds)
            pred_arrays[view+'_'+family] = pred
            print(view, family, domain[view][family]['OOF_AUC'], flush=True)
    prediction_path = ROOT / 'domain_predictions.npz'
    np.savez_compressed(prediction_path, domain=labels, **pred_arrays)
    history = []
    for version in ['15_retry1','16','17','18','19','20','21']:
        directory = PROJECT / f'improvement_v{version}'
        frame = pd.read_csv(directory / 'results/screening.csv')
        score_column = 'mean_relative_gain'
        assert score_column in frame
        best = frame.sort_values(score_column, ascending=False).iloc[0]
        review = directory / 'delivery/independent_review.json'
        history.append(dict(version=version, comparisons=len(frame), best=best.to_dict(),
                    screening_sha256=sha(directory/'results/screening.csv'),
                    independent_review_sha256=sha(review),
                    review_status=read(review).get('status'),
                    qualification_floor_fraction=.002))
    report = dict(status='PASS', checked_at_IST=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        scope='Planning evidence. No turnover-model fits, new checkpoints, GPU allocations or portal actions.',
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__),
        script_sha256=sha(__file__), input_sha256=inputs, scored_reference_sha256=refs,
        training_rows=len(train), test_rows=len(test), predictor_count=len(X.columns),
        exact_train_predictor_duplicates=int(X.duplicated().sum()),
        exact_test_predictor_duplicates=int(Z.duplicated().sum()),
        exact_train_test_predictor_matches=int(pd.util.hash_pandas_object(Z,index=False).isin(pd.util.hash_pandas_object(X,index=False)).sum()),
        numeric_shift=stats, V14_control_metrics=control_metrics,
        verified_control_fold_outputs=witnesses, sampling_sensitivity=simulation,
        domain_classifier=dict(protocol='Five stratified held-out domain folds, seed 2201; fold-fitted imputation/scaling/vocabulary. Domain labels are train/test membership. IDs and turnover excluded. Fixed recipes, no tuning.',
            views=domain, prediction_file=prediction_path.name, prediction_sha256=sha(prediction_path),
            limitation='Domain separability measures observed feature differences, not target error, label-preserving corruption, or the validity of feature removal. These diagnostics cannot validate covariate-shift assumptions.'),
        historical_screening=history,
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'evidence_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved evidence_audit.json; seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
