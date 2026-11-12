from pathlib import Path
import ast, json, textwrap, hashlib
import nbformat as nb

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
STEM='submission'
v2=json.loads((BASE/'improvement_v2/results/locked_candidate.json').read_text())
weights=json.loads((BASE/'improvement_v2/results/final_execution.json').read_text())['weights']
lock=json.loads((ROOT/'results/candidate_lock.json').read_text())
contract={'v2_configs':v2['configs'],'v2_weights':weights,'v3':lock}
digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
hashes={f:hashlib.sha256((Path('~/Downloads')/f).read_bytes()).hexdigest() for f in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']}
manifest={'input_hashes':hashes,'configuration_sha256':digest,
          'files':{f'comparison_{s}.npz':hashlib.sha256((ROOT/'results'/f'comparison_{s}.npz').read_bytes()).hexdigest() for s in [71,2026,303,909,'city']}}
(ROOT/'results/cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
source=(BASE/'improvement_v2/experiment.py').read_text()
tree=ast.parse(source)
funcs={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
constants=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['TARGET','ID','EVENTS','SOCIAL','CORE'] for t in n.targets)]
newsource=(ROOT/'experiments.py').read_text()
newfuncs={n.name:ast.get_source_segment(newsource,n) for n in ast.parse(newsource).body if isinstance(n,ast.FunctionDef)}
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))

md('''
# Restaurant Annual Turnover Prediction — October 12 Improvement

**Student register number:** submission

**Student name:** submission

**Team name:** submission

Predict annual turnover in INR for the 500 test restaurants. The submission has
exactly `Registration Number` and `Annual Turnover` as its two columns.

**Colab instructions:** upload this notebook, use a CPU runtime, and Run all.
Upload the four supplied CSV inputs when prompted. The final cell downloads the
prediction CSV. Save the executed `.ipynb` through Colab's File menu.
Validation fits many small models and can take several minutes on a Colab CPU.
''')
md('''
## 1. Objective, actual leaderboard result, and model selection

This is **regression**. The metric is `sqrt(mean((actual - prediction)**2))`
on the original turnover values in INR. Lower is better. Every validation
restaurant, including extreme turnover values, is included.

The October 12 screenshot shows **one submission**, RMSE **12,204,615.6438253**,
and **rank 7**. The same submission was ranked 3 on October 11. Its score did not
change; competing entries moved ahead. The leader shown scores **11,946,659.0702364**.
The earlier improved model (v2) and this new candidate have no portal result in the
provided screenshots. A local score cannot establish their hidden-test ranking.

The earlier round compared 38 model settings. This round tested 42 further settings:
14 CatBoost configurations averaged over three seeds, 20 spline/linear/kernel
settings, and 8 Random Forest settings. A coarse blend grid on seed-42 development
predictions selected **50% of the v2 ensemble + 50% of a three-seed subsampled
CatBoost ensemble**. Random Forest worsened development RMSE; smoother models
provided only a small gain and were not retained.

All settings below are frozen before the confirmation checks. Seeds 71, 2026 and
303 were already used in earlier work. Seed 909's split was reserved during this
round's selection. All splits reuse the same labeled restaurants, whose labels
also informed development; these are robustness checks, **not an untouched external
test or a nested estimate of the entire model-selection process**.

Huber losses and a training-only cap are regularization choices. Validation targets
are never capped, transformed, removed, or replaced. No hidden test targets are used.
''')
code('''
import os, sys, json, hashlib, importlib.util, subprocess, warnings, copy
from pathlib import Path
packages={'numpy':'numpy==2.5.4','pandas':'pandas==3.0.6','sklearn':'scikit-learn==1.9.1',
          'catboost':'catboost==1.2.10','scipy':'scipy','matplotlib':'matplotlib'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, catboost, scipy
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
warnings.filterwarnings('ignore',category=FutureWarning)
FILE_STEM='submission'
OUTPUT_DIR=Path.cwd()
print('Python:',sys.version.split()[0])
print('Versions:',{k:v.__version__ for k,v in {'numpy':np,'pandas':pd,'sklearn':sklearn,'catboost':catboost,'scipy':scipy}.items()})
print('Tested versions: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, catboost 1.2.10.')
''')
md('''
## 2. Read the source files and enforce their schema

The dictionary identifies `Registration Number` as the **restaurant ID**. Student
details appear in the notebook and filenames, not in the 500 prediction rows.
The endorsement-column spelling difference in train and test is normalized.
''')
code('''
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
DATA_DIR=Path.cwd()
if not all((DATA_DIR/f).exists() for f in required):
    local=Path('~/Downloads')
    if all((local/f).exists() for f in required): DATA_DIR=local
    else:
        try: from google.colab import files
        except ImportError: raise FileNotFoundError('Place the four source CSVs beside the notebook or set DATA_DIR.')
        files.upload(); DATA_DIR=Path.cwd()
assert all((DATA_DIR/f).exists() for f in required)
train=pd.read_csv(DATA_DIR/required[0]).rename(columns={'Endorsed By':'Endoresed By'})
test=pd.read_csv(DATA_DIR/required[1]); dictionary=pd.read_csv(DATA_DIR/required[2]); sample=pd.read_csv(DATA_DIR/required[3])
y=train['Annual Turnover'].to_numpy(float)
assert np.isfinite(y).all() and (y>=0).all()
assert 'Annual Turnover' not in test
assert train.drop(columns='Annual Turnover').columns.tolist()==test.columns.tolist()
assert len(test)==500 and test['Registration Number'].is_unique and train['Registration Number'].is_unique
assert set(train['Registration Number']).isdisjoint(test['Registration Number'])
assert sample.columns.tolist()==['Registration Number','Annual Turnover']
assert sample['Registration Number'].equals(test['Registration Number'])
input_hashes={f:hashlib.sha256((DATA_DIR/f).read_bytes()).hexdigest() for f in required}
print('Train:',train.shape,'Test:',test.shape)
print('PASS: schema, target values, restaurant IDs, sample headers and row order.')
print('Source SHA256 hashes:',json.dumps(input_hashes,indent=2))
''')
md('''
## 3. Data findings and their implications

A small upper tail has a large effect on squared error. It motivates robust training
experiments but does not justify deleting difficult validation rows. Service and
comfort ratings also differ between train and test; a feature-removal variant was
evaluated in v2. These are possible reasons local RMSE differs from portal RMSE;
the cause cannot be verified without the hidden targets.
''')
code('''
print(pd.Series(y).describe(percentiles=[.5,.9,.95,.99]).round(0).to_string())
sq=(y-y.mean())**2
tail=y>100e6
print(f'Above INR 100m: {tail.sum()} restaurants; {100*sq[tail].sum()/sq.sum():.1f}% of mean-baseline squared error.')
print(pd.DataFrame({'Train':train[['Service','Comfortablility']].mean(),'Test':test[['Service','Comfortablility']].mean()}).round(3).to_string())
fig,ax=plt.subplots(figsize=(8,3.5))
ax.hist(y/1e6,bins=45,color='#2563eb',edgecolor='white')
ax.set(xlabel='Annual turnover (INR millions)',ylabel='Restaurants (log scale)',yscale='log',title='All training targets are retained for evaluation')
fig.tight_layout();plt.show()
''')
md('''
## 4. Reproducible feature and model functions

Dates, cuisines, event availability, and missingness are handled deterministically.
The target and restaurant ID are excluded. Imputation, encoding and scaling are
fitted only on each training fold. CatBoost uses categorical values directly.

The new component uses Huber delta INR 40m, depth 5, 850 trees, learning rate 0.04,
L2 regularization 10, and Bernoulli row subsampling 0.7. Three seeds reduce variation
between training runs. The earlier ten-model mixture remains half of the final
prediction. This adds diversity without choosing predictions using hidden labels.
''')
code('\n\n'.join(constants+[funcs[n] for n in ['features','rmse','sklearn_model','fit_predict','predict_fitted']]+[newfuncs['cat_fit']]))
code('CONTRACT=json.loads('+repr(json.dumps(contract))+')\n'+'''
V2_CONFIGS=CONTRACT['v2_configs']; V2_WEIGHTS=CONTRACT['v2_weights']; V3=CONTRACT['v3']
OLD_CONFIGS={
    'old_raw':dict(kind='cat',variant='base',depth=6,iterations=302,loss='RMSE'),
    'old_huber':dict(kind='cat',variant='base',depth=6,iterations=768,loss='Huber:delta=12000000')}
assert np.isclose(sum(V2_WEIGHTS.values()),1)
assert np.isclose(V3['v2_weight']+sum(V3['additional_weights'].values()),1)
variants={s['variant'] for s in V2_CONFIGS.values()}|{s['variant'] for s in V3['specs'].values()}
feature_data={v:features(train,v) for v in variants}
for v,X in feature_data.items():
    assert 'Annual Turnover' not in X and 'Registration Number' not in X
    assert X.columns.equals(features(test,v).columns)
    changed=train.copy();changed['Annual Turnover']=0;changed['Registration Number']=-1
    pd.testing.assert_frame_equal(X,features(changed,v))
assert np.isclose(rmse([0,3],[0,0]),np.sqrt(4.5))
print('PASS: original-scale RMSE toy example, weight sums, target/ID exclusion and feature alignment.')
print('V3 weights:',V3['v2_weight'],V3['additional_weights'])
print('Development-only comparison:',V3['development'])
''')
md('''
## 5. Confirm fixed settings on repeated folds and unseen cities

Every row receives exactly one held-out prediction in each five-fold assignment.
The city stress test puts canonical city aliases in the same group, keeping city
groups disjoint between training and validation.

The local executed notebook can reuse the actual fold predictions from this round
using an explicit cache environment variable. Input hashes, configuration hash and
prediction-file hashes must match before reuse. The output identifies this reuse.
**Colab has no cache and executes the complete validation training below.**
''')
code('''
def get_splits(scheme):
    if scheme=='city':
        groups=features(train,'engineered')['City']
        splits=list(GroupKFold(5).split(train,groups=groups))
        for tr,va in splits: assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
    else: splits=list(KFold(5,shuffle=True,random_state=int(scheme)).split(train))
    counts=np.zeros(len(y),int)
    for tr,va in splits:
        assert set(tr).isdisjoint(va); counts[va]+=1
    assert (counts==1).all()
    return splits

def compute_validation(scheme):
    splits=get_splits(scheme); results={k:np.zeros(len(y)) for k in ['old','v2','candidate']}
    for name,spec in {**OLD_CONFIGS,**V2_CONFIGS}.items():
        X=feature_data[spec['variant']]; component=np.zeros(len(y))
        for f,(tr,va) in enumerate(splits):
            component[va],_=fit_predict(X,y,tr,va,spec,43+f,spec.get('iterations'))
        if name in OLD_CONFIGS: results['old']+=(.55 if name=='old_raw' else .45)*component
        else: results['v2']+=V2_WEIGHTS[name]*component
    results['candidate']=V3['v2_weight']*results['v2']
    for name,w in V3['additional_weights'].items():
        spec=V3['specs'][name];X=feature_data[spec['variant']]; component=np.zeros(len(y))
        for f,(tr,va) in enumerate(splits):
            p=[]
            for seed in V3['model_seeds']:
                model=cat_fit(X.iloc[tr],y[tr],spec,seed+f)
                p.append(np.maximum(model.predict(X.iloc[va]),0))
            component[va]=np.mean(p,axis=0)
        results['candidate']+=w*component
    return results

cache_value=os.environ.get('MLA_VALIDATION_CACHE')
cache_root=Path(cache_value) if cache_value else None
manifest=None
if cache_root is not None:
    manifest=json.loads((cache_root/'cache_manifest.json').read_text())
    assert manifest['input_hashes']==input_hashes, 'Cache inputs changed'
    assert manifest['configuration_sha256']==hashlib.sha256(json.dumps(CONTRACT,sort_keys=True).encode()).hexdigest()
validation_predictions={};rows=[]
for scheme in [71,2026,303,909,'city']:
    get_splits(scheme)  # Check coverage and separation even when reusing predictions.
    filename=f'comparison_{scheme}.npz'
    if cache_root is not None:
        assert hashlib.sha256((cache_root/filename).read_bytes()).hexdigest()==manifest['files'][filename]
        cached=np.load(cache_root/filename)
        predictions={k:cached[k] for k in ['old','v2','candidate']}
        print('Verified executed-fold cache:',filename)
    else:
        print('Training validation models:',scheme)
        predictions=compute_validation(scheme)
    assert all(p.shape==y.shape and np.isfinite(p).all() and (p>=0).all() for p in predictions.values())
    validation_predictions[str(scheme)]=predictions
    rows.append({'Validation':str(scheme),'Submitted model RMSE':rmse(y,predictions['old']),
                 'Previous improved RMSE':rmse(y,predictions['v2']),'Current candidate RMSE':rmse(y,predictions['candidate'])})
validation_table=pd.DataFrame(rows)
validation_table['Gain vs previous improved']=validation_table['Previous improved RMSE']-validation_table['Current candidate RMSE']
print(validation_table.to_string(index=False,float_format=lambda x:f'{x:,.2f}'))
random_table=validation_table[validation_table['Validation']!='city']
means=random_table.drop(columns='Validation').mean()
print('Mean across four shuffled assignments:')
print(means.to_string(float_format=lambda x:f'{x:,.2f}'))
''')
md('''
## 6. Interpret validation and uncertainty

Lower RMSE across several fixed split assignments supports trying the candidate.
The size of the improvement should be reported alongside uncertainty. The bootstrap
below resamples rows of already-computed errors; its interval is conditional on these
models and does not account for every source of model-selection or distribution risk.
The city result is a separate stress test, not part of the random-split average.
In this executed run the conditional interval includes zero, so the incremental
gain over v2 is not decisive under this resampling check.
''')
code('''
random_results=[validation_predictions[str(s)] for s in [71,2026,303,909]]
old_sq=np.mean([(y-p['old'])**2 for p in random_results],axis=0)
v2_sq=np.mean([(y-p['v2'])**2 for p in random_results],axis=0)
new_sq=np.mean([(y-p['candidate'])**2 for p in random_results],axis=0)
rng=np.random.default_rng(812);gains=[]
for _ in range(1000):
    ix=rng.integers(0,len(y),len(y));gains.append(np.sqrt(v2_sq[ix].mean())-np.sqrt(new_sq[ix].mean()))
gain_ci=np.quantile(gains,[.025,.975])
print('Conditional 95% interval for improvement vs v2:',np.round(gain_ci,2).tolist())
print('All targets included in the RMSE above; no hidden-test score is available.')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
x=np.arange(len(validation_table)); labels=['71','2026','303','909','City']
for column,color in [('Submitted model RMSE','#64748b'),('Previous improved RMSE','#2563eb'),('Current candidate RMSE','#0f766e')]:
    axes[0].plot(x,validation_table[column]/1e6,'o-',label=column.replace(' RMSE',''),color=color,linewidth=1.5)
axes[0].set_xticks(x,labels)
axes[0].set(xlabel='Validation assignment',ylabel='RMSE (INR millions)',title='Absolute validation RMSE')
axes[0].legend(fontsize=8,loc='upper left');axes[0].grid(axis='y',alpha=.2)
g=validation_table['Gain vs previous improved'].to_numpy()/1000
axes[1].bar(x,g,color=['#0f766e' if v>=0 else '#b45309' for v in g])
axes[1].axhline(0,color='#334155',linewidth=1)
axes[1].set_xticks(x,labels)
axes[1].set(xlabel='Validation assignment',ylabel='RMSE reduction vs v2 (INR thousands)',title='Positive = improved; City stress test worsened')
fig.tight_layout();plt.show()
''')
md('''
## 7. Fit the frozen models on all training rows

Training uses only the supplied train targets. The test data contributes no target
values. Predictors remain in original INR units; the final mixture is nonnegative.
''')
code('''
full_parts={};fitted_v2={};fitted_new={}
tr_idx=np.arange(len(train));te_idx=np.arange(len(train),len(train)+len(test))
combined_y=np.r_[y,np.full(len(test),np.nan)]
for name,spec in V2_CONFIGS.items():
    X=pd.concat([feature_data[spec['variant']],features(test,spec['variant'])],ignore_index=True)
    pred,trees,fitted=fit_predict(X,combined_y,tr_idx,te_idx,spec,42,spec.get('iterations'),return_fitted=True)
    full_parts[name]=pred;fitted_v2[name]=fitted
    print('Fitted v2 component:',name)
v2_test=sum(V2_WEIGHTS[n]*p for n,p in full_parts.items())
test_prediction=V3['v2_weight']*v2_test
for name,w in V3['additional_weights'].items():
    spec=V3['specs'][name];models=[];parts=[]
    for seed in V3['model_seeds']:
        model=cat_fit(feature_data[spec['variant']],y,spec,seed)
        models.append(model);parts.append(np.maximum(model.predict(features(test,spec['variant'])),0))
    fitted_new[name]=models;test_prediction+=w*np.mean(parts,axis=0)
    print('Fitted new component:',name,'seeds:',V3['model_seeds'])

def predict_final(raw):
    prev=sum(V2_WEIGHTS[n]*predict_fitted(m,features(raw,V2_CONFIGS[n]['variant'])) for n,m in fitted_v2.items())
    pred=V3['v2_weight']*prev
    for name,w in V3['additional_weights'].items():
        X=features(raw,V3['specs'][name]['variant'])
        pred+=w*np.mean([np.maximum(m.predict(X),0) for m in fitted_new[name]],axis=0)
    return pred

assert np.allclose(predict_final(test),test_prediction,rtol=1e-12,atol=.01)
print('Mean absolute change from v2 predictions (INR):',float(np.abs(test_prediction-v2_test).mean()))
''')
md('''
## 8. Behavior tests, exact submission format, and export

The CSV contains 500 prediction records plus one header row. No index column is
written. These checks establish format and prediction behavior; they do not measure
accuracy on the test restaurants because their true turnovers are unavailable.
''')
code('''
perm=np.random.default_rng(91).permutation(len(test))
assert np.allclose(predict_final(test.iloc[perm]),test_prediction[perm],rtol=1e-10,atol=.01)
edge=test.iloc[:2].copy();edge['City']='unseen_validation_city';edge['Cuisine']=pd.NA
edge['Facebook Popularity Quotient']=np.nan;edge['Live Music Rating']=np.nan
edge_pred=predict_final(edge)
assert np.isfinite(edge_pred).all() and (edge_pred>=0).all()

def check_submission(frame):
    assert frame.shape==(500,2)
    assert frame.columns.tolist()==['Registration Number','Annual Turnover']
    assert frame['Registration Number'].equals(test['Registration Number'])
    assert frame['Registration Number'].is_unique
    assert np.isfinite(frame['Annual Turnover']).all() and (frame['Annual Turnover']>=0).all()

submission=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':test_prediction})
check_submission(submission)
# Deliberately wrong row order, count, header and values must be rejected.
bad_cases=[submission.iloc[:-1],submission.iloc[::-1].reset_index(drop=True),
           submission.rename(columns={'Annual Turnover':'Turnover'}),submission.assign(index=0)]
for invalid in [np.nan,np.inf,-1]:
    b=submission.copy();b.loc[0,'Annual Turnover']=invalid;bad_cases.append(b)
for bad in bad_cases:
    rejected=False
    try:check_submission(bad)
    except AssertionError:rejected=True
    assert rejected,'Invalid submission was not rejected'
output_path=OUTPUT_DIR/(FILE_STEM+'.csv')
submission.to_csv(output_path,index=False)
roundtrip=pd.read_csv(output_path);check_submission(roundtrip)
assert np.allclose(roundtrip['Annual Turnover'],test_prediction,rtol=1e-12,atol=.001)
assert len(output_path.read_text().splitlines())==501
print('PASS: row permutation, unseen/missing values, seven intentionally invalid examples, exact headers, IDs,500 rows, finite values and CSV round-trip.')
print('Saved:',output_path.name)
print('SHA256:',hashlib.sha256(output_path.read_bytes()).hexdigest())
print('Turnover range:',float(test_prediction.min()),'to',float(test_prediction.max()))
print(submission.head().to_string(index=False))
''')
md('''
## 9. Inferences and submission

- The previous submitted model and improved candidates are compared on identical
  validation rows, using the same original-scale RMSE formula.
- Robust training and blending reduce sensitivity to extreme targets and training
  randomness. The resulting gains are modest; adding algorithms did not always help.
- The current candidate combines the existing mixture with three subsampled models.
  Its apparent local improvement is evidence for a submission, not proof of rank 1.
- The city-isolation stress test worsened relative to v2. This is a real limitation:
  generalization to entirely unseen cities may be weaker even though all four random
  splits improved. Both improved candidates still beat the original model there.
- The conditional bootstrap interval includes zero. Treat the small gain over v2
  as uncertain and this file as a candidate to evaluate on the portal.
- A new portal score below **11,946,659.0702364** would beat the leader in the supplied
  screenshot, assuming that leader score remains unchanged. The new result is unknown.

Upload the generated **CSV** under **assignment M.Tech. MLA Hackathon -Restaurant Turnover
Prediction**. Keep the observed portal score with that exact file version.
Upload the executed **notebook** with its code, outputs and inferences to
**LLJ_Oct_2026** only when the model choice is final; that assignment permits one
submission according to the provided instructions. The hackathon permits repeated
CSV submissions. The portal registration format is a separate faculty requirement;
renaming these files does not change the registration shown as submission.
''')
code('''
try:
    from google.colab import files
    files.download(str(output_path))
except ImportError:
    print('Prediction CSV is ready in the notebook folder.')
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':STEM+'.ipynb','provenance':[]}})
nb.validate(book);nb.write(book,ROOT/(STEM+'.ipynb'))
print('Built notebook:',ROOT/(STEM+'.ipynb'))
