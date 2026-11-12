from pathlib import Path
import ast,json,hashlib,textwrap
import nbformat as nb
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
STEM='submission'
def definitions(path):
    source=path.read_text();tree=ast.parse(source)
    return {n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}, {t.id:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
old,constants=definitions(BASE/'improvement_v2/experiment.py')
v3,_=definitions(BASE/'improvement_v3/experiments.py')
v4,_=definitions(BASE/'improvement_v4/pipeline.py')
new,_=definitions(ROOT/'new_models.py');pipe,_=definitions(ROOT/'pipeline.py')
contract=json.loads((ROOT/'results/candidate_lock.json').read_text());portal=json.loads((ROOT/'portal_snapshot.json').read_text())
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
filenames=[f'comparison_{s}.npz' for s in [18421,18422,2026,303,'city']]+['full_prediction_parts.npz']
manifest={'inputs':{n:hashlib.sha256((Path('~/Downloads')/n).read_bytes()).hexdigest() for n in required},'contract':hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),'files':{n:hashlib.sha256((ROOT/'results'/n).read_bytes()).hexdigest() for n in filenames}}
(ROOT/'results/cache_manifest.json').write_text(json.dumps(manifest,indent=2))
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# Restaurant Annual Turnover Prediction — October 18 model comparison

**Student register number:** submission

**Student name:** submission

**Team name:** submission

In Google Colab, upload this notebook, select a CPU runtime and run all cells.
Upload the four official input CSVs when prompted. The final cell downloads the
500-row prediction file. Save the executed notebook from Colab's File menu.
Repeated validation may take several minutes; progress is printed per split.
''')
md('''
## 1. Objective and known portal result

Predict continuous annual turnover in INR. Score predictions using
`sqrt(mean((actual - prediction)**2))`. Lower RMSE is better.

The October 18 screenshots show **rank 18**, two submissions, and unchanged RMSE
**12,094,600.3216516**. Other entries moved ahead. The leader shown is
**submission**, RMSE **11,917,847.2693885**.
The required reduction is more than **176,753.05 (1.461%)** if that leading score
stays unchanged. This notebook's new CSV has **no observed portal score**.

This round screened 22 training settings on five-fold seed 42, tested six
shortlisted settings on seed 71, and selected a coarse blend using those two
development assignments. Settings were then frozen. The confirmation uses new
split assignments 18421/18422, previously used assignments 2026/303, and a separate
city-group stress test. All assignments reuse labels involved in earlier work:
these are robustness checks, not an independent external test or nested estimate.
No hidden labels or test-score-derived prediction corrections are used.
''')
code('''
import os,sys,json,hashlib,importlib.util,subprocess,warnings
from pathlib import Path
packages={'numpy':'numpy','pandas':'pandas','sklearn':'scikit-learn','catboost':'catboost','scipy':'scipy','matplotlib':'matplotlib'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn,catboost,scipy
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold,GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
warnings.filterwarnings('ignore',category=FutureWarning)
FILE_STEM='submission';OUTPUT_DIR=Path.cwd()
print('Runtime:',sys.version.split()[0])
print('Library versions:',{k:v.__version__ for k,v in {'numpy':np,'pandas':pd,'sklearn':sklearn,'catboost':catboost,'scipy':scipy}.items()})
''')
md('''
## 2. Load inputs and check alignment

The row-level `Registration Number` is the restaurant ID. Student details belong
in the filename and notebook cover. Normalize the endorsement spelling difference
between train and test. Preserve the original test order and all validation targets.
''')
code('''
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
DATA_DIR=Path.cwd()
if not all((DATA_DIR/n).exists() for n in required):
    local=Path('~/Downloads')
    if all((local/n).exists() for n in required):DATA_DIR=local
    else:
        from google.colab import files
        files.upload()
assert all((DATA_DIR/n).exists() for n in required)
train=pd.read_csv(DATA_DIR/required[0]).rename(columns={'Endorsed By':'Endoresed By'})
test=pd.read_csv(DATA_DIR/required[1]);dictionary=pd.read_csv(DATA_DIR/required[2]);sample=pd.read_csv(DATA_DIR/required[3])
y=train['Annual Turnover'].to_numpy(float)
assert len(test)==500 and np.isfinite(y).all() and (y>0).all()
assert train.drop(columns='Annual Turnover').columns.tolist()==test.columns.tolist()
assert train['Registration Number'].is_unique and test['Registration Number'].is_unique
assert set(train['Registration Number']).isdisjoint(test['Registration Number'])
assert sample.columns.tolist()==['Registration Number','Annual Turnover']
assert sample['Registration Number'].equals(test['Registration Number'])
input_hashes={n:hashlib.sha256((DATA_DIR/n).read_bytes()).hexdigest() for n in required}
print('PASS: input schemas, positive targets, unique restaurant IDs, disjoint train/test and sample row order.')
print('Train shape:',train.shape,'Test shape:',test.shape)
print('Input SHA256:',json.dumps(input_hashes,indent=2))
''')
md('''
## 3. Modeling hypotheses

Extreme training turnovers have a large effect on squared error. This round compares
square-root/log/power target models, training-only mean corrections after inversion,
quantile objectives, training-only caps, feature selection inside each training fold,
and new feature combinations. Training changes never alter validation labels.
The correction, when selected, is the training mean of original target minus
inverse-transformed fitted prediction. It is estimated from training predictions;
it can be biased and must be judged by held-out original-scale RMSE.

No algorithm is assumed to improve results automatically. The complete selected
configuration and signed comparison results are printed below.
''')
code('''
print(pd.Series(y).describe(percentiles=[.5,.9,.95,.99]).to_string())
tail=y>100e6;squared=(y-y.mean())**2
print('Rows above INR100m:',tail.sum(),'Share of mean-baseline squared error:',100*squared[tail].sum()/squared.sum())
fig,ax=plt.subplots(figsize=(8,3.4));ax.hist(y/1e6,bins=45,color='#2563eb',edgecolor='white')
ax.set(xlabel='Annual turnover (INR millions)',ylabel='Restaurants, log scale',yscale='log',title='All original targets remain in validation')
fig.tight_layout();plt.show()
''')
code('\n\n'.join([constants[n] for n in ['TARGET','ID','EVENTS','SOCIAL','CORE']]+[old[n] for n in ['features','rmse','sklearn_model','fit_predict','predict_fitted']]+[v3['cat_fit']]+[v4[n] for n in ['get_splits','compute_v3','check_submission']]+[new[n] for n in ['v5_features','inverse_target','fit_v5','predict_v5']]+[pipe[n] for n in ['evaluate','full_fit','predict_final']]))
code('CONTRACT=json.loads('+repr(json.dumps(contract))+')\n'+'''
assert np.isclose(CONTRACT['v3_weight']+sum(c['weight'] for c in CONTRACT['components'].values()),1)
assert np.isclose(sum(CONTRACT['v2_weights'].values()),1)
assert np.isclose(rmse([0,3],[0,0]),np.sqrt(4.5))
changed=train.copy();changed['Annual Turnover']=0;changed['Registration Number']=-1
for variant in ['base','clean','rich']:
    X=v5_features(train,variant)
    assert X.columns.equals(v5_features(test,variant).columns)
    pd.testing.assert_frame_equal(X,v5_features(changed,variant))
    assert 'Annual Turnover' not in X and 'Registration Number' not in X
print('PASS: RMSE toy example, weight sums, target/ID exclusion and feature invariance.')
print('Frozen blend:',CONTRACT['v3_weight'],json.dumps(CONTRACT['components'],indent=2))
print('Development selection:',json.dumps(CONTRACT['development'],indent=2))
''')
md('''
## 4. Same-fold comparisons and uncertainty

The local executed copy may use explicitly requested actual fold predictions with
verified input/configuration/file hashes. It recomputes every metric. Colab normally
has no cache and trains all folds. Each split scores every original target exactly
once. Known city aliases are grouped together for the separate city stress check;
other spelling variants may remain, so this is an approximate geographic check.
''')
code('''
cache_value=os.environ.get('MLA_VALIDATION_CACHE');cache_root=Path(cache_value) if cache_value else None
manifest=None
if cache_root is not None:
    manifest=json.loads((cache_root/'cache_manifest.json').read_text())
    assert manifest['inputs']==input_hashes
    assert manifest['contract']==hashlib.sha256(json.dumps(CONTRACT,sort_keys=True).encode()).hexdigest()
validation_predictions={};rows=[]
for scheme in [18421,18422,2026,303,'city']:
    get_splits(train,scheme)
    if cache_root is None:
        print('Training validation assignment:',scheme,flush=True)
        predictions=evaluate(train,y,scheme,CONTRACT)
    else:
        name=f'comparison_{scheme}.npz'
        assert hashlib.sha256((cache_root/name).read_bytes()).hexdigest()==manifest['files'][name]
        predictions=dict(np.load(cache_root/name));print('Verified actual fold cache:',name)
    for key in ['previous','candidate']:
        assert predictions[key].shape==y.shape and np.isfinite(predictions[key]).all() and (predictions[key]>=0).all()
    validation_predictions[str(scheme)]=predictions
    rows.append({'Validation':str(scheme),'Previous RMSE':rmse(y,predictions['previous']),'Candidate RMSE':rmse(y,predictions['candidate'])})
validation_table=pd.DataFrame(rows);validation_table['Reduction']=validation_table['Previous RMSE']-validation_table['Candidate RMSE']
print(validation_table.to_string(index=False,float_format=lambda x:f'{x:,.2f}'))
random_table=validation_table[validation_table.Validation!='city']
means=random_table.drop(columns='Validation').mean()
fresh_means=validation_table[validation_table.Validation.isin(['18421','18422'])].drop(columns='Validation').mean()
print('Mean four shuffled assignments:',means.to_dict())
print('Mean two new assignments:',fresh_means.to_dict())
supported=bool(means['Reduction']>0 and fresh_means['Reduction']>0)
print('Recommendation:', 'Reasonable candidate to try on the portal; ranking remains unconfirmed.' if supported else 'Keep confirmed v3 as preferred; this candidate is experimental.')
''')
code('''
schemes=[18421,18422,2026,303]
previous_sq=np.mean([(y-validation_predictions[str(s)]['previous'])**2 for s in schemes],axis=0)
candidate_sq=np.mean([(y-validation_predictions[str(s)]['candidate'])**2 for s in schemes],axis=0)
rng=np.random.default_rng(812);gains=[]
for _ in range(2000):
    ix=rng.integers(0,len(y),len(y));gains.append(np.sqrt(previous_sq[ix].mean())-np.sqrt(candidate_sq[ix].mean()))
gain_ci=np.quantile(gains,[.025,.975]);print('Conditional row-bootstrap95% interval for reduction:',gain_ci.tolist())
print('Interval includes zero:',bool(gain_ci[0]<=0<=gain_ci[1]))
print('This conditional interval excludes full model-selection and distribution uncertainty. Repeated splits reuse restaurants.')
fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(5);labels=['18421*','18422*','2026','303','City']
for column,color in [('Previous RMSE','#2563eb'),('Candidate RMSE','#0f766e')]:
    axes[0].plot(x[:4],validation_table[column].iloc[:4]/1e6,'o-',color=color,label=column)
    axes[0].scatter(x[-1],validation_table[column].iloc[-1]/1e6,color=color,marker='D')
axes[0].set(ylabel='RMSE (INR millions)',title='Same rows, original turnover values');axes[0].legend()
reduction=validation_table.Reduction.to_numpy()/1000
axes[1].bar(x,reduction,color=['#0f766e' if r>0 else '#b45309' for r in reduction]);axes[1].axhline(0,color='#334155')
axes[1].set(ylabel='RMSE reduction (INR thousands)',title='Positive = better; negative = worse')
for ax in axes:
    ax.set_xticks(x,labels);ax.set_xlabel('*New split; city excluded from random averages');ax.axvline(3.5,linestyle='--',color='#64748b')
fig.tight_layout();plt.show()
''')
md('''
## 5. Train all selected models and test the export

Full fitting uses all supplied training rows. The optional independent full-fit
reference is only checked after refitting; predictions are not substituted from it.
Format and behavior checks do not measure accuracy on hidden test targets.
''')
code('''
previous_test,test_prediction,fitted=full_fit(train,test,y,CONTRACT)
assert np.allclose(predict_final(test,fitted,CONTRACT),test_prediction,rtol=1e-12,atol=.01)
if cache_root is not None:
    name='full_prediction_parts.npz'
    assert hashlib.sha256((cache_root/name).read_bytes()).hexdigest()==manifest['files'][name]
    expected=np.load(cache_root/name)
    assert np.allclose(previous_test,expected['previous'],rtol=1e-12,atol=.01)
    assert np.allclose(test_prediction,expected['candidate'],rtol=1e-12,atol=.01)
perm=np.random.default_rng(91).permutation(len(test))
assert np.allclose(predict_final(test.iloc[perm],fitted,CONTRACT),test_prediction[perm],rtol=1e-10,atol=.01)
edge=test.iloc[:2].copy();edge['City']='unseen_city';edge['Cuisine']=pd.NA;edge['Facebook Popularity Quotient']=np.nan
assert np.isfinite(predict_final(edge,fitted,CONTRACT)).all()
submission=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':test_prediction});check_submission(submission,test)
bad=[submission.iloc[:-1],submission.iloc[::-1].reset_index(drop=True),submission.rename(columns={'Annual Turnover':'Turnover'}),submission.assign(index=0),submission.assign(**{'Registration Number':'submission'})]
for value in [np.nan,np.inf,-1]:
    one=submission.copy();one.loc[0,'Annual Turnover']=value;bad.append(one)
for one in bad:
    rejected=False
    try:check_submission(one,test)
    except (AssertionError,TypeError,ValueError):rejected=True
    assert rejected
output_path=OUTPUT_DIR/(FILE_STEM+'.csv');submission.to_csv(output_path,index=False)
roundtrip=pd.read_csv(output_path);check_submission(roundtrip,test)
assert np.allclose(roundtrip['Annual Turnover'],test_prediction,rtol=1e-12,atol=.001)
assert len(output_path.read_text().splitlines())==501
print('PASS: full-fit reproduction, row permutation, missing/unseen values, eight invalid examples, CSV round-trip.')
print('Saved:',output_path.name,'Rows:',len(submission),'Columns:',submission.columns.tolist())
print('SHA256:',hashlib.sha256(output_path.read_bytes()).hexdigest())
print(submission.head().to_string(index=False))
print('Mean absolute change from confirmed v3:',float(np.abs(test_prediction-previous_test).mean()))
''')
md('''
## 6. Inferences and submission

Use the signed validation results and uncertainty above to judge this candidate.
Local RMSE differs from the portal RMSE because the evaluated restaurants differ.
Do not subtract a local improvement from the prior portal score to predict ranking.
The new CSV's portal RMSE is unknown until it is evaluated there. First place would
require a score below **11,917,847.2693885**, if the shown leader is unchanged.

If choosing to try this candidate, upload the CSV to the hackathon module. Record
the returned score with its exact filename/version/hash. Preserve the previously
successful CSV. Submit the executed notebook matching the final chosen model to
**LLJ_Oct_2026** when ready; the provided instructions allow one notebook submission.
''')
code('''
try:
    from google.colab import files
    files.download(str(output_path))
except ImportError:print('CSV is ready in the notebook folder.')
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':STEM+'.ipynb','provenance':[]}})
for cell in book.cells:
    if cell.cell_type=='code':compile(cell.source,'notebook_cell','exec')
nb.validate(book);nb.write(book,ROOT/(STEM+'.ipynb'));print('Notebook built')
