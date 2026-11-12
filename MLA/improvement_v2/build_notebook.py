"""Build a self-contained Colab notebook from the evaluated implementation."""
from pathlib import Path
import ast, json, textwrap, uuid

ROOT=Path(__file__).resolve().parent
NAME='submission'
source=(ROOT/'experiment.py').read_text()
nodes=ast.parse(source).body
functions={n.name:ast.get_source_segment(source,n) for n in nodes if isinstance(n,ast.FunctionDef)}
constants=[]
for n in nodes:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['TARGET','ID','EVENTS','SOCIAL','CORE','SPECS'] for t in n.targets):
        constants.append(ast.get_source_segment(source,n))
cells=[]
def md(s): cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=textwrap.dedent(s).strip().splitlines(keepends=True)))
def code(s): cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=textwrap.dedent(s).strip().splitlines(keepends=True),execution_count=None,outputs=[]))

md('''
# Restaurant Annual Turnover Prediction — Improved Submission

**Student register number:** submission

**Student name:** submission

**Team name:** submission

Predict annual turnover in INR for 500 restaurants. Lower **RMSE** is better.

This notebook contains executable code, saved outputs, findings, model comparisons,
and submission checks. It compares 38 settings and blends the useful models using
predictions made for rows held out from each training fold.

**Google Colab:** upload this notebook, select a CPU runtime, and run all cells.
Upload the four supplied CSV files when prompted. The final cell downloads
`submission.csv`. Download the executed notebook through
Colab's File menu before submitting it to the assignment. Run time depends on the CPU.
''')
md('''
## 1. Objective and evaluation

This is a regression problem. The generic classification/accuracy FAQ does not
describe this target. We score **sqrt(mean((actual - predicted)^2))** in original INR.

The previous submitted CSV received **12,204,615.6438253 RMSE, rank 3**, in the user's
11 October 2026 screenshot. The earlier leader scored **11,946,659.0702364**. Those
are portal-reported scores. The new CSV's hidden-test score is unknown until uploaded.

Development uses shuffled five-fold validation (seed 42) to compare settings and
choose weights. Three further shuffled assignments and a City GroupKFold stress
test evaluate fixed models and weights. These checks reuse the labeled dataset;
they are robustness estimates, not an independent test set. Seeds 71 and 2026
were also examined during the earlier 30-setting search. Seed 303 was reserved
until the final candidate was fixed. No hidden targets or leaderboard labels
were used to train models or choose weights.
''')
code('''
import importlib.util, subprocess, sys, warnings, json, hashlib, copy
from pathlib import Path

packages = {'numpy':'numpy','pandas':'pandas','matplotlib':'matplotlib',
            'sklearn':'scikit-learn','scipy':'scipy','catboost':'catboost',
            'lightgbm':'lightgbm','xgboost':'xgboost'}
missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, catboost, lightgbm, xgboost
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
warnings.filterwarnings('ignore', category=FutureWarning)

SEED = 42
FILE_STEM = 'submission'
OUTPUT_DIR = Path.cwd()
print('Python:',sys.version.split()[0])
print('Versions:', {k:v.__version__ for k,v in {'numpy':np,'pandas':pd,'sklearn':sklearn,'catboost':catboost,'lightgbm':lightgbm,'xgboost':xgboost}.items()})
''')
md('''
## 2. Load and audit the inputs

Restaurant registration numbers identify prediction rows. They are different from
the student's register number in the filename and are excluded from model features.
The train/test spelling difference in the endorsement column is aligned explicitly.
''')
code('''
required = ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
DATA_DIR = Path.cwd()
if not all((DATA_DIR/f).exists() for f in required):
    local = Path('~/Downloads')
    if all((local/f).exists() for f in required):
        DATA_DIR = local
    else:
        try:
            from google.colab import files
        except ImportError:
            raise FileNotFoundError('Place all four CSV files beside this notebook, or set DATA_DIR.')
        files.upload()
        DATA_DIR = Path.cwd()
assert all((DATA_DIR/f).exists() for f in required), 'Upload all four input CSV files.'
train=pd.read_csv(DATA_DIR/required[0]).rename(columns={'Endorsed By':'Endoresed By'})
test=pd.read_csv(DATA_DIR/required[1])
dictionary=pd.read_csv(DATA_DIR/required[2])
sample=pd.read_csv(DATA_DIR/required[3])
y=train['Annual Turnover'].to_numpy(dtype=float)
assert np.isfinite(y).all() and (y>=0).all()
assert 'Annual Turnover' not in test
assert len(test)==500 and test['Registration Number'].is_unique
assert train['Registration Number'].is_unique
assert set(train['Registration Number']).isdisjoint(test['Registration Number'])
assert train.drop(columns='Annual Turnover').columns.tolist()==test.columns.tolist()
assert sample.columns.tolist()==['Registration Number','Annual Turnover']
assert sample['Registration Number'].equals(test['Registration Number'])
print('Train:',train.shape,'Test:',test.shape)
print('All input schema, target, identifier and row-order checks passed.')
print('Dictionary variables:',len(dictionary))
print('Sample predictions are placeholders:',sample['Annual Turnover'].unique().tolist())
input_hashes={f:hashlib.sha256((DATA_DIR/f).read_bytes()).hexdigest() for f in required}
print('Input SHA256 hashes:')
print(json.dumps(input_hashes,indent=2))
''')
md('''
## 3. Findings that guide the experiments

Turnover has a long upper tail. A few extreme restaurants contribute much of the
squared error, so we compare standard RMSE training with Huber losses and a
training-only turnover cap. **Every validation target remains unchanged**, including
extreme values. No rows are removed from the RMSE score.

Service and comfort ratings have different distributions in train and test. Their
removal is evaluated as an experiment; their values are not shifted arbitrarily.
Event-rating blanks mean an event is not offered according to the data dictionary,
so the features retain missingness and event availability.
''')
code('''
print(pd.Series(y,name='Annual Turnover').describe(percentiles=[.5,.9,.95,.99]).round(0).to_string())
errors=(y-y.mean())**2
for threshold in [60e6,100e6,200e6]:
    mask=y>threshold
    print(f'Above INR {threshold/1e6:g}m: {mask.sum()} rows; {100*errors[mask].sum()/errors.sum():.1f}% of mean-baseline squared error')
print('Service/comfort train versus test means:')
print(pd.DataFrame({'Train':train[['Service','Comfortablility']].mean(),'Test':test[['Service','Comfortablility']].mean()}).round(3).to_string())
numeric=train.select_dtypes('number').drop(columns='Registration Number')
print('Largest absolute rank correlations (associations, not causation):')
print(numeric.corr(method='spearman')['Annual Turnover'].drop('Annual Turnover').sort_values(key=abs,ascending=False).head(8).round(3).to_string())
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
ax[0].hist(y/1e6,bins=45,color='#2563eb',edgecolor='white')
ax[0].set(xlabel='Annual turnover (INR millions)',ylabel='Restaurants',title='Full target distribution',yscale='log')
ax[1].hist(y[y<=100e6]/1e6,bins=30,color='#0f766e',edgecolor='white')
ax[1].set(xlabel='Annual turnover (INR millions)',ylabel='Restaurants',title='Target at or below INR 100m (EDA only)')
fig.tight_layout(); plt.show()
''')
md('''
## 4. Features and model functions

The base features include normalized categories, opening year/month, cuisine parts,
and event indicators. Enhanced variants add opening-date details, social means and
differences, missingness counts, and rating interactions. One variant combines a
small explicit set of city aliases. All feature engineering is deterministic per row.

CatBoost handles categories directly. Other algorithms use imputation and one-hot
encoding fitted **within each training fold**, with unseen categories handled safely.
The spline model adds smooth numeric effects; SVR supplies a different model shape.

The monotonic CatBoost candidate assumes higher social popularity, hygiene and Zomato
ratings should not decrease predicted turnover, and a higher tier number should
not increase it when other features are fixed. This is a tested modeling assumption.
Its feature importance is predictive association, not evidence of causation.
''')
code('\n\n'.join(constants+[functions[n] for n in ['features','rmse','sklearn_model','fit_predict','predict_fitted']]))
code('''
feature_data={variant:features(train,variant) for variant in {s['variant'] for s in SPECS.values()}}
for variant,X in feature_data.items():
    assert 'Annual Turnover' not in X and 'Registration Number' not in X
    assert X.columns.equals(features(test,variant).columns)
    changed=train.copy(); changed['Annual Turnover']=0; changed['Registration Number']=-1
    pd.testing.assert_frame_equal(X,features(changed,variant))
assert np.isclose(rmse([0,3],[0,0]),np.sqrt(4.5))
print('RMSE formula check, target/ID exclusion checks and train/test feature alignment passed.')
print('Base feature count:',feature_data['base'].shape[1])
''')
md('''
## 5. Development comparison: 38 settings

Settings include CatBoost, LightGBM, XGBoost, Extra Trees, histogram boosting, Ridge,
spline Ridge and SVR. Early stopping and model selection use these development
folds, so their scores are selection estimates. The later checks use fixed tree
counts and fixed ensemble weights. The rank-3 baseline is reproduced as
0.55 × raw CatBoost (302 trees) + 0.45 × Huber CatBoost (768 trees).
''')
code('''
dev_folds=list(KFold(5,shuffle=True,random_state=42).split(train))
coverage=np.zeros(len(train),dtype=int)
for tr,va in dev_folds:
    assert set(tr).isdisjoint(va)
    coverage[va]+=1
assert (coverage==1).all()
dev_predictions={}; dev_trees={}; dev_rows=[]
for name,spec in SPECS.items():
    X=feature_data[spec['variant']]; pred=np.zeros(len(y)); trees=[]
    for f,(tr,va) in enumerate(dev_folds,1):
        pred[va],n=fit_predict(X,y,tr,va,spec,42+f)
        trees.append(n)
    dev_predictions[name]=pred; dev_trees[name]=trees
    dev_rows.append({'Model':name,'Development RMSE':rmse(y,pred)})
    print(f'{name:25s} RMSE {rmse(y,pred):,.0f}')
development_table=pd.DataFrame(dev_rows).sort_values('Development RMSE')
old_dev=.55*dev_predictions['raw_baseline']+.45*dev_predictions['huber_baseline']
print(f'Rank-3 model, fixed configuration on development folds: {rmse(y,old_dev):,.0f}')
print(development_table.head(10).to_string(index=False,float_format=lambda v:f'{v:,.0f}'))
''')
md('''
## 6. Freeze the ensemble before further checks

Fit nonnegative weights that sum to one using development out-of-fold predictions.
The objective is squared error in INR millions plus `2 × sum(weight²)`, which
discourages placing all weight on one model. Drop weights below 5% and renormalize.
Tree counts are the median selected across development folds. No later validation
score is used to adjust the weights. The published test score from the old model
is a reference only and does not enter this calculation.
''')
code('''
names=list(SPECS)
matrix=np.column_stack([dev_predictions[n] for n in names])/1e6
opt=minimize(lambda w:np.mean((matrix@w-y/1e6)**2)+2*np.sum(w*w),np.ones(len(names))/len(names),
             bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
             method='SLSQP',options={'maxiter':1000,'ftol':1e-10})
assert opt.success,opt.message
weights=np.where(opt.x>=.05,opt.x,0); weights/=weights.sum()
ensemble_weights={n:float(w) for n,w in zip(names,weights) if w>0}
fixed_specs={}
for name in ensemble_weights:
    spec=copy.deepcopy(SPECS[name])
    if spec['kind'] in ['cat','lgb','xgb']: spec['iterations']=int(np.median(dev_trees[name]))
    fixed_specs[name]=spec
new_dev=sum(w*dev_predictions[n] for n,w in ensemble_weights.items())
print(pd.Series(ensemble_weights,name='Weight').sort_values(ascending=False).round(4).to_string())
print(f'Development ensemble RMSE: {rmse(y,new_dev):,.0f}')
''')
md('''
## 7. Repeated validation and an unseen-city stress test

Compare both models on identical folds, without using the validation targets for
early stopping. Use three shuffled five-fold assignments and five city groups.
City grouping prevents a city's rows from appearing in both training and validation
within a fold, making it a harder check for restaurants in unseen cities.

These are repeat checks on the same labeled dataset. They cannot establish the new
hidden-test RMSE, and agreement across them cannot guarantee first place.
''')
code('''
validation_schemes={f'Shuffled 5-fold seed {seed}':list(KFold(5,shuffle=True,random_state=seed).split(train)) for seed in [71,2026,303]}
city_groups=feature_data['engineered']['City']  # Keep the explicit city aliases in one group.
validation_schemes['City GroupKFold']=list(GroupKFold(5).split(train,y,city_groups))
validation_specs={'raw_baseline':SPECS['raw_baseline'],'huber_baseline':SPECS['huber_baseline'],**fixed_specs}
validation_rows=[]; validation_predictions={}
for label,folds in validation_schemes.items():
    pp={}
    coverage=np.zeros(len(y),dtype=int)
    for tr,va in folds:
        assert set(tr).isdisjoint(va)
        if label=='City GroupKFold': assert set(city_groups.iloc[tr]).isdisjoint(city_groups.iloc[va])
        coverage[va]+=1
    assert (coverage==1).all()
    for name,spec in validation_specs.items():
        pred=np.zeros(len(y)); X=feature_data[spec['variant']]
        for f,(tr,va) in enumerate(folds,1):
            pred[va],_=fit_predict(X,y,tr,va,spec,42+f,spec.get('iterations'))
        pp[name]=pred
    old=.55*pp['raw_baseline']+.45*pp['huber_baseline']
    new=sum(w*pp[n] for n,w in ensemble_weights.items())
    validation_predictions[label]={'old':old,'new':new}
    row={'Validation':label,'Rank-3 model RMSE':rmse(y,old),'New candidate RMSE':rmse(y,new),'Reduction':rmse(y,old)-rmse(y,new)}
    validation_rows.append(row)
    print(f'{label}: old {row["Rank-3 model RMSE"]:,.0f}; new {row["New candidate RMSE"]:,.0f}; reduction {row["Reduction"]:,.0f}')
validation_table=pd.DataFrame(validation_rows)
random_table=validation_table[validation_table['Validation'].str.startswith('Shuffled')]
print('Mean across three shuffled assignments:')
print(random_table.drop(columns='Validation').mean().round(0).to_string())
''')
md('''
## 8. Interpret the improvement and its uncertainty

The full-data RMSE above is the primary measure. The extra diagnostic for turnover
at or below INR 100 million helps explain sensitivity to the extreme tail; it is
not a replacement score and does not exclude those rows from model selection.

The bootstrap below resamples restaurant rows from the repeated validation errors.
Its interval describes sampling uncertainty conditional on the fitted models;
it does not cover every source of uncertainty, including future distribution changes.
''')
code('''
random_predictions=[v for k,v in validation_predictions.items() if k.startswith('Shuffled')]
old_sq=np.mean([(y-v['old'])**2 for v in random_predictions],axis=0)
new_sq=np.mean([(y-v['new'])**2 for v in random_predictions],axis=0)
rng=np.random.default_rng(412)
gains=[]
for _ in range(1000):
    ix=rng.integers(0,len(y),len(y))
    gains.append(np.sqrt(old_sq[ix].mean())-np.sqrt(new_sq[ix].mean()))
gain_ci=np.quantile(gains,[.025,.975])
print(f'Conditional 95% bootstrap interval for RMSE reduction: [{gain_ci[0]:,.0f}, {gain_ci[1]:,.0f}]')
body=y<=100e6
print(f'Diagnostic RMSE for targets <= INR 100m: old {np.sqrt(old_sq[body].mean()):,.0f}; new {np.sqrt(new_sq[body].mean()):,.0f}')
print('All validation rows were included in the primary results.')
print('Previous portal-reported RMSE: 12,204,615.64; new portal score: not yet available.')
fig,ax=plt.subplots(figsize=(10,4))
x=np.arange(len(validation_table)); width=.36
ax.bar(x-width/2,validation_table['Rank-3 model RMSE']/1e6,width,label='Rank-3 model',color='#94a3b8')
ax.bar(x+width/2,validation_table['New candidate RMSE']/1e6,width,label='New candidate',color='#2563eb')
ax.set_xticks(x,['Shuffle 71','Shuffle 2026','Shuffle 303','Unseen cities'])
ax.set_ylabel('RMSE (INR millions)'); ax.set_title('Validation with fixed model settings; lower is better')
ax.legend(); fig.tight_layout(); plt.show()
''')
md('''
## 9. Train on all labeled data and predict the test restaurants

Each selected model is refitted on all training rows using its frozen settings.
The final prediction is the weighted average. Nonnegative predictions are enforced.
Test targets are unavailable and are never imputed or used for fitting.
''')
code('''
full_predictions={}; fitted_models={}; test_features={}
tr_idx=np.arange(len(train)); te_idx=np.arange(len(train),len(train)+len(test))
combined_target=np.r_[y,np.full(len(test),np.nan)]
for name,spec in fixed_specs.items():
    variant=spec['variant']; tx=features(test,variant)
    both=pd.concat([feature_data[variant],tx],ignore_index=True)
    pred,trees,fitted=fit_predict(both,combined_target,tr_idx,te_idx,spec,42,spec.get('iterations'),return_fitted=True)
    full_predictions[name]=pred; fitted_models[name]=fitted; test_features[name]=tx
    print(f'Fitted {name}; trees={trees}; weight={ensemble_weights[name]:.4f}')
test_prediction=sum(w*full_predictions[n] for n,w in ensemble_weights.items())

# Reproduce the old final model for an auditable comparison of prediction files.
old_parts=[]
for name in ['raw_baseline','huber_baseline']:
    spec=SPECS[name]; both=pd.concat([feature_data['base'],features(test)],ignore_index=True)
    p,_=fit_predict(both,combined_target,tr_idx,te_idx,spec,42,spec['iterations'])
    old_parts.append(p)
old_test_prediction=.55*old_parts[0]+.45*old_parts[1]
archived=Path('rank3_backup')/(FILE_STEM+'.csv')
if archived.exists():
    archived_df=pd.read_csv(archived)
    assert archived_df['Registration Number'].equals(test['Registration Number'])
    assert np.allclose(old_test_prediction,archived_df['Annual Turnover'],rtol=1e-10,atol=.01)
    print('Old model reproduces the archived rank-3 CSV within INR 0.01 tolerance.')
print(f'Mean absolute prediction change: INR {np.mean(np.abs(test_prediction-old_test_prediction)):,.0f}')
importance_name=next(n for n in ensemble_weights if fixed_specs[n]['kind']=='cat')
importance_model=fitted_models[importance_name]['model']
importance=pd.Series(importance_model.feature_importances_,index=importance_model.feature_names_).sort_values(ascending=False)
print('Predictive feature importance for',importance_name,':')
print(importance.head(10).round(2).to_string())
''')
md('''
## 10. Check prediction behavior and export the required format

Checks cover ID/target leakage in feature preparation, disjoint validation folds,
unseen categorical values, missing values, row permutation, and CSV round-trip.
They cannot check prediction accuracy on the test rows without the hidden targets.
''')
code('''
# Predictions must follow restaurant rows even after their order is changed.
perm=np.random.default_rng(82).permutation(len(test))
permuted=sum(ensemble_weights[n]*predict_fitted(fitted_models[n],features(test.iloc[perm],fixed_specs[n]['variant'])) for n in ensemble_weights)
assert np.allclose(permuted,test_prediction[perm],rtol=1e-10,atol=.01)

# Small edge case: an unseen city, missing cuisine and missing numeric ratings.
edge=test.iloc[:2].copy()
edge['City']='new_unseen_city_for_validation'
edge['Cuisine']=pd.NA
edge['Facebook Popularity Quotient']=np.nan
edge['Live Music Rating']=np.nan
edge_prediction=sum(ensemble_weights[n]*predict_fitted(fitted_models[n],features(edge,fixed_specs[n]['variant'])) for n in ensemble_weights)
assert np.isfinite(edge_prediction).all() and (edge_prediction>=0).all()

submission=pd.DataFrame({'Registration Number':test['Registration Number'].to_numpy(),'Annual Turnover':test_prediction})
assert submission.shape==(500,2)
assert submission.columns.tolist()==['Registration Number','Annual Turnover']
assert submission['Registration Number'].equals(sample['Registration Number'])
assert submission['Registration Number'].is_unique
assert np.isfinite(submission['Annual Turnover']).all() and (submission['Annual Turnover']>=0).all()
output_path=OUTPUT_DIR/(FILE_STEM+'.csv')
submission.to_csv(output_path,index=False)
reloaded=pd.read_csv(output_path)
assert reloaded.shape==(500,2)
assert reloaded['Registration Number'].equals(test['Registration Number'])
assert np.allclose(reloaded['Annual Turnover'],test_prediction,rtol=1e-12,atol=.001)
assert len(output_path.read_text().splitlines())==501
print('PASS: row-order invariance, unseen/missing feature cases, finite nonnegative predictions, IDs, headers, 500 rows and CSV round-trip.')
print('Saved:',output_path.name)
print('SHA256:',hashlib.sha256(output_path.read_bytes()).hexdigest())
print('Prediction range (INR):',round(test_prediction.min()),'to',round(test_prediction.max()))
print(submission.head().to_string(index=False))
''')
md('''
## 11. Findings and submission instructions

1. Extreme turnover values explain much of the training RMSE. Huber training and
   training-only capping help manage their influence; validation retains every target.
2. The selected ensemble combines useful models with different error patterns.
   LightGBM and XGBoost were evaluated but did not receive retained ensemble weight
   in this run. More algorithms do not automatically improve the result.
3. Validation scores and portal scores use the same stated RMSE metric on different
   data. This notebook's validation values are not forecasts of the new portal score.
4. An improvement across repeated splits supports trying the candidate, but the
   gain may be small and the hidden-test ranking can still worsen. The previous
   rank-3 notebook and CSV were retained as a backup.

Upload the generated CSV to **assignment M.Tech. MLA Hackathon -Restaurant Turnover
Prediction**. Upload this executed `.ipynb`, including its outputs, to **LLJ_Oct_2026**
when your model choice is final. The instructions say the assignment allows one
submission, while the hackathon allows multiple CSV submissions. Filenames do not
change the portal registration name; follow the faculty's registration format there.
Review the code and explain these choices in your own words.
''')
code('''
try:
    from google.colab import files
    files.download(str(output_path))
except ImportError:
    print('The prediction CSV is ready in the notebook folder.')
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':NAME+'.ipynb','provenance':[]}},'cells':cells}
path=ROOT/(NAME+'.ipynb')
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print(path)
