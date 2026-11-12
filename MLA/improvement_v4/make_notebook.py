"""Build the portable v4 notebook after its independent fold/full fits finish.

This generator reads actual saved model outputs to make a hash-verified local
cache manifest. It never fills notebook outputs with synthetic results. Running
the generated notebook executes all cells; Colab uses the complete training path.
"""
from pathlib import Path
import ast
import json
import textwrap
import hashlib
import nbformat as nb

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
STEM='submission'
SCHEMES=[1701,1711,71,2026,303,909,'city']
INPUTS=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']

def read_definitions(path):
    source=path.read_text()
    tree=ast.parse(source)
    funcs={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name): constants[target.id]=ast.get_source_segment(source,node)
    return funcs,constants

def build():
    contract=json.loads((ROOT/'results/candidate_lock.json').read_text())
    expected_files=[f'comparison_{s}.npz' for s in SCHEMES]+['full_prediction_parts.npz']
    missing=[str(ROOT/'results'/f) for f in expected_files if not (ROOT/'results'/f).exists()]
    if missing: raise FileNotFoundError('Complete independent model fits before building: '+', '.join(missing))
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    hashes={f:hashlib.sha256((Path('~/Downloads')/f).read_bytes()).hexdigest() for f in INPUTS}
    manifest={'input_hashes':hashes,'configuration_sha256':digest,
              'files':{f:hashlib.sha256((ROOT/'results'/f).read_bytes()).hexdigest() for f in expected_files}}
    (ROOT/'results/cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    old_funcs,old_constants=read_definitions(BASE/'improvement_v2/experiment.py')
    v3_funcs,_=read_definitions(BASE/'improvement_v3/experiments.py')
    feature_funcs,feature_constants=read_definitions(ROOT/'features/screen_features.py')
    pipe_funcs,_=read_definitions(ROOT/'pipeline.py')
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))

    md('''
    # Restaurant Annual Turnover Prediction — Fourth Model Comparison

    **Student register number:** submission

    **Student name:** submission

    **Team name:** submission

    **Status:** experimental candidate; compare confirmation results before replacing the confirmed submission.

    Predict annual turnover in INR for 500 test restaurants. The CSV contains
    exactly `Registration Number` and `Annual Turnover` as its two columns.

    **Google Colab:** upload this notebook, select a CPU runtime, and Run all.
    Upload the four supplied CSV files when prompted. The final cell downloads
    the prediction CSV. Save the executed notebook with its outputs using the
    Colab File menu. The complete repeated validation fits many models and may
    take tens of minutes on a CPU; the progress messages identify each assignment.
    ''')
    md('''
    ## 1. Objective and the results that are actually known

    Annual turnover is a continuous value, so this is a **regression** problem.
    The evaluation metric is `sqrt(mean((actual - prediction)**2))`, in original
    INR units. Every validation restaurant is included, including extreme values.
    No hidden test targets are available locally.

    The supplied October 12 screenshot confirms two CSV submissions and **rank 3**.
    The latest confirmed best RMSE is **12,094,600.3216516**, improved from the
    first submission's **12,204,615.6438253**. The leader shown scores
    **11,946,659.0702364**. This notebook generates a further candidate whose
    **portal RMSE and rank are unknown until submitted**.

    This round evaluated **24 settings, including the reference setting**, and a
    coarse grid of **78 blends** on the existing seed-42 development predictions.
    Settings and blend weights were frozen before subsequent comparisons. The
    final mixture retains 25% of the submitted v3 model, adds 37.5% of a CatBoost
    model using 32 numeric split borders, and 37.5% of a CatBoost model that treats
    ordinal ratings as categories. Each new component averages two model seeds.

    Validation assignments 1701 and 1711 were not used to select this candidate.
    Assignments 71, 2026, 303 and 909 were used previously; cities form a separate
    stress test. **All assignments reuse the same labeled restaurants**, whose
    labels informed earlier development. These are robustness checks, not an
    untouched external test or an unbiased nested model-selection estimate.
    No blend is retuned using these confirmation scores or the public score.
    ''')
    code('''
    import os, sys, json, hashlib, importlib.util, subprocess, warnings
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
    print('Tested library versions: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, catboost 1.2.10.')
    print('Known first submission RMSE: 12,204,615.6438253')
    print('Known latest submission RMSE: 12,094,600.3216516; screenshot rank: 3')
    print('Leader shown: 11,946,659.0702364; this candidate portal score: UNKNOWN')
    ''')
    md('''
    ## 2. Load files and enforce the data contract

    `Registration Number` in each prediction row is the **restaurant identifier**.
    The student's register number belongs in the notebook and filenames. The
    training file's `Endorsed By` spelling is normalized to the test file's spelling.
    ''')
    code('''
    required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
    DATA_DIR=Path.cwd()
    if not all((DATA_DIR/f).exists() for f in required):
        local=Path('~/Downloads')
        if all((local/f).exists() for f in required): DATA_DIR=local
        else:
            try: from google.colab import files
            except ImportError: raise FileNotFoundError('Place the four CSV files beside the notebook or set DATA_DIR.')
            files.upload(); DATA_DIR=Path.cwd()
    assert all((DATA_DIR/f).exists() for f in required)
    train=pd.read_csv(DATA_DIR/required[0]).rename(columns={'Endorsed By':'Endoresed By'})
    test=pd.read_csv(DATA_DIR/required[1])
    dictionary=pd.read_csv(DATA_DIR/required[2]); sample=pd.read_csv(DATA_DIR/required[3])
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
    print('PASS: schema, target validity, unique/disjoint restaurant IDs, sample headers and row order.')
    print('Input SHA256 hashes:',json.dumps(input_hashes,indent=2))
    ''')
    md('''
    ## 3. Data findings and the modeling hypotheses

    A small upper tail strongly affects squared error. Robust training losses and
    training-only target capping are candidate regularizers; they do not change
    the RMSE metric or justify removing large validation targets. Service and
    comfort rating distributions differ between train and test. These observations
    can motivate comparisons, but the hidden test RMSE cannot be inferred from them.

    Fewer numeric split thresholds may reduce variance. Treating ordinal ratings as
    categories permits unordered category groupings instead of only ordered numeric
    threshold splits. Numeric trees already support nonlinear effects without an
    equal-distance assumption. The retained part of the previous ensemble supplies models that
    still use ratings numerically. Whether either hypothesis helps is checked below.
    ''')
    code('''
    print(pd.Series(y).describe(percentiles=[.5,.9,.95,.99]).round(0).to_string())
    sq=(y-y.mean())**2; tail=y>100e6
    print(f'Above INR 100m: {tail.sum()} rows; {100*sq[tail].sum()/sq.sum():.1f}% of mean-baseline squared error.')
    print(pd.DataFrame({'Train':train[['Service','Comfortablility']].mean(),
                        'Test':test[['Service','Comfortablility']].mean()}).round(3).to_string())
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.hist(y/1e6,bins=45,color='#2563eb',edgecolor='white')
    ax.set(xlabel='Annual turnover (INR millions)',ylabel='Restaurants (log scale)',yscale='log',
           title='All original training targets remain in validation')
    fig.tight_layout();plt.show()
    ''')
    md('''
    ## 4. Self-contained features, models and the frozen contract

    The notebook includes the complete fitting code, including the submitted model
    used for comparison. Each imputer, encoder and scaler is fitted inside its
    training fold. Features exclude restaurant ID and turnover. Date parsing and
    categorical string normalization are deterministic; missing values receive
    explicit treatment. Categorical rating values are passed directly to CatBoost.

    The complete configuration below records selected models, training seeds,
    weights, development results and the validation plan. Huber delta is INR40m for
    the new models. Their iterations are fixed, so validation labels are not used
    for early stopping. All scored labels retain their original turnover values.
    ''')
    constants=[old_constants[n] for n in ['TARGET','ID','EVENTS','SOCIAL','CORE']]
    constants += [feature_constants[n] for n in ['RATINGS','FLAGS']]
    functions=[old_funcs[n] for n in ['features','rmse','sklearn_model','fit_predict','predict_fitted']]
    functions += [v3_funcs['cat_fit'],feature_funcs['feature_frame']]
    functions += [pipe_funcs[n] for n in ['get_splits','new_fit','compute_v3','new_oof','full_fit','predict_final','check_submission']]
    code('\n\n'.join(constants+functions))
    code('CONTRACT=json.loads('+repr(json.dumps(contract))+')\n'+textwrap.dedent('''
    assert np.isclose(sum(CONTRACT['v2_weights'].values()),1)
    assert np.isclose(CONTRACT['v3_weight']+sum(s['weight'] for s in CONTRACT['components'].values()),1)
    assert np.isclose(rmse([0,3],[0,0]),np.sqrt(4.5))
    changed=train.copy();changed['Annual Turnover']=0;changed['Registration Number']=-1
    variants={s['variant'] for s in CONTRACT['v2_configs'].values()}
    for variant in variants:
        X=features(train,variant)
        assert X.columns.equals(features(test,variant).columns)
        assert 'Annual Turnover' not in X and 'Registration Number' not in X
        pd.testing.assert_frame_equal(X,features(changed,variant))
    for component in CONTRACT['components'].values():
        variant=component['variant'];X=feature_frame(train,variant)
        assert X.columns.equals(feature_frame(test,variant).columns)
        assert 'Annual Turnover' not in X and 'Registration Number' not in X
        pd.testing.assert_frame_equal(X,feature_frame(changed,variant))
    print('PASS: toy RMSE formula, weight sums, target/ID exclusion and feature schema alignment.')
    print('Frozen development comparison:',json.dumps(CONTRACT['development'],indent=2))
    print('Previous model weight:',CONTRACT['v3_weight'])
    print('New components:',json.dumps(CONTRACT['components'],indent=2))
    ctr=feature_frame(train,'categorical_ratings');cte=feature_frame(test,'categorical_ratings')
    coverage=[]
    for c in RATINGS:
        coverage.append({'Feature':c,'Train categories':ctr[c].nunique(),
                         'Unseen test rows':int((~cte[c].isin(set(ctr[c]))).sum())})
    print('Categorical coverage:',pd.DataFrame(coverage).to_string(index=False),sep='\\n')
    '''))
    md('''
    ## 5. Compare frozen models on identical validation assignments

    Every restaurant receives exactly one held-out prediction per five-fold
    assignment. Known canonical city aliases are combined for the separate city
    stress check. The earlier city-normalization rules are retained so that the
    old and new models use precisely the same comparison groups; city spellings
    outside that explicit map remain a grouping limitation.

    The local executed notebook may reuse **actual model predictions** via the
    explicit `MLA_VALIDATION_CACHE` environment variable. Input hashes, the complete
    configuration hash, and prediction-file hashes must match. The cell prints
    each verified reuse. With that variable absent, including normal Colab use,
    `compute_validation(..., use_prior_cache=False)` trains every fold below.
    ''')
    code('''
    def compute_validation(train,y,scheme,contract,use_prior_cache=False):
        if use_prior_cache:
            raise ValueError('Use the explicit hash-verified cache loader below, or train with use_prior_cache=False.')
        previous=compute_v3(train,y,scheme,contract)
        prediction=contract['v3_weight']*previous
        for name,component in contract['components'].items():
            prediction+=component['weight']*new_oof(train,y,scheme,component,contract['seeds'])
        return {'previous':previous,'candidate':prediction}

    cache_value=os.environ.get('MLA_VALIDATION_CACHE')
    cache_root=Path(cache_value) if cache_value else None
    manifest=None
    if cache_root is not None:
        manifest=json.loads((cache_root/'cache_manifest.json').read_text())
        assert manifest['input_hashes']==input_hashes,'Cached input files changed'
        digest=hashlib.sha256(json.dumps(CONTRACT,sort_keys=True).encode()).hexdigest()
        assert manifest['configuration_sha256']==digest,'Cached configuration changed'
    validation_predictions={};rows=[]
    schemes=[1701,1711,71,2026,303,909,'city']
    for scheme in schemes:
        get_splits(train,scheme) # Fold/group coverage checks run even with a verified cache.
        filename=f'comparison_{scheme}.npz'
        if cache_root is not None:
            assert hashlib.sha256((cache_root/filename).read_bytes()).hexdigest()==manifest['files'][filename]
            cached=np.load(cache_root/filename)
            predictions={k:cached[k] for k in ['previous','candidate']}
            print('Verified executed-fold cache:',filename)
        else:
            print('Training complete validation:',scheme,flush=True)
            predictions=compute_validation(train,y,scheme,CONTRACT,use_prior_cache=False)
        assert all(p.shape==y.shape and np.isfinite(p).all() and (p>=0).all() for p in predictions.values())
        validation_predictions[str(scheme)]=predictions
        rows.append({'Validation':str(scheme),'Type':'City stress' if scheme=='city' else 'New split' if scheme in [1701,1711] else 'Previous split',
                     'Previous submitted RMSE':rmse(y,predictions['previous']),
                     'Current candidate RMSE':rmse(y,predictions['candidate'])})
    validation_table=pd.DataFrame(rows)
    validation_table['Gain vs previous submitted']=validation_table['Previous submitted RMSE']-validation_table['Current candidate RMSE']
    print(validation_table.to_string(index=False,float_format=lambda x:f'{x:,.2f}'))
    metric_columns=['Previous submitted RMSE','Current candidate RMSE','Gain vs previous submitted']
    random_table=validation_table[validation_table['Validation']!='city']
    means=random_table[metric_columns].mean()
    fresh_means=validation_table[validation_table['Type']=='New split'][metric_columns].mean()
    print('Mean across all six shuffled assignments:')
    print(means.to_string(float_format=lambda x:f'{x:,.2f}'))
    print('Mean of two split assignments reserved during this round selection:')
    print(fresh_means.to_string(float_format=lambda x:f'{x:,.2f}'))
    promote_candidate=(fresh_means['Gain vs previous submitted']>0 and means['Gain vs previous submitted']>0)
    print('Decision:', 'Candidate passes the positive-mean comparison rule; hidden-test performance remains unknown.'
          if promote_candidate else 'Retain the confirmed v3 submission: this experimental candidate does not pass both positive-mean comparisons.')
    ''')
    md('''
    ## 6. Inferences, error concentration and uncertainty

    Confirmation results are shown with their actual signs: a negative reduction
    means the new candidate worsened. The city stress result is reported separately
    and is excluded from random-split averages. Repeated assignments are correlated
    because they reuse restaurants. They cannot be counted as independent datasets.

    The paired bootstrap resamples restaurants' already-computed squared errors,
    averaging their errors across assignments. Its interval is conditional on
    these fitted and selected models. It does not capture full retraining,
    model-selection optimism, or differences in the hidden test distribution.
    Target-tail diagnostics do not remove any rows from the primary metric.
    ''')
    code('''
    shuffled=[1701,1711,71,2026,303,909]
    previous_sq=np.mean([(y-validation_predictions[str(s)]['previous'])**2 for s in shuffled],axis=0)
    candidate_sq=np.mean([(y-validation_predictions[str(s)]['candidate'])**2 for s in shuffled],axis=0)
    rng=np.random.default_rng(812);gains=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y))
        gains.append(np.sqrt(previous_sq[ix].mean())-np.sqrt(candidate_sq[ix].mean()))
    gain_ci=np.quantile(gains,[.025,.975])
    print('Conditional paired 95% interval for RMSE reduction:',np.round(gain_ci,2).tolist())
    if gain_ci[0]<=0<=gain_ci[1]:
        print('Inference: the interval includes zero; the incremental gain is uncertain under this check.')
    elif gain_ci[0]>0:
        print('Inference: this conditional interval favors the candidate; it does not establish a hidden-test improvement.')
    else:
        print('Inference: this conditional interval favors the submitted model.')
    wins=int((random_table['Gain vs previous submitted']>0).sum())
    print(f'Shuffled validation comparisons improved: {wins} of {len(random_table)}.')
    city_gain=float(validation_table.loc[validation_table['Validation']=='city','Gain vs previous submitted'].iloc[0])
    print(f'City stress test reduction: {city_gain:,.2f}; '+('improved' if city_gain>0 else 'worsened or tied'))
    for label,mask in [('At most INR 100m',y<=100e6),('Above INR 100m',y>100e6)]:
        print(label,': rows=',mask.sum(),'; previous RMSE=',round(float(np.sqrt(previous_sq[mask].mean())),2),
              '; candidate RMSE=',round(float(np.sqrt(candidate_sq[mask].mean())),2),
              '; candidate squared-error share=',round(float(100*candidate_sq[mask].sum()/candidate_sq.sum()),2),'%')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    x=np.arange(len(validation_table));labels=['1701*','1711*','71','2026','303','909','City']
    for column,color in [('Previous submitted RMSE','#2563eb'),('Current candidate RMSE','#0f766e')]:
        axes[0].plot(x[:6],validation_table[column].iloc[:6]/1e6,'o-',label=column.replace(' RMSE',''),color=color)
        axes[0].scatter(x[-1],validation_table[column].iloc[-1]/1e6,marker='D',color=color,s=55)
    axes[0].set_xticks(x,labels)
    axes[0].set(xlabel='*New split assignment; City is a separate stress test',ylabel='RMSE (INR millions)',title='Validation on all original turnover values')
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    g=validation_table['Gain vs previous submitted'].to_numpy()/1000
    axes[1].bar(x,g,color=['#0f766e' if v>=0 else '#b45309' for v in g])
    axes[1].axhline(0,color='#334155',linewidth=1)
    axes[1].set_xticks(x,labels)
    axes[1].set(xlabel='*New split assignment; City excluded from means',ylabel='RMSE reduction (INR thousands)',title='Positive means improved; negative means worse')
    for ax in axes:
        ax.axvline(5.5,color='#64748b',linestyle='--',linewidth=1)
        ax.axvspan(5.55,6.45,color='#f1f5f9',alpha=.8,zorder=-1)
    fig.tight_layout();plt.show()
    ''')
    md('''
    ## 7. Fit the frozen models on all labeled rows

    This cell refits every selected model on all training restaurants. It does not
    substitute cached final predictions. The optional local cache contains a
    separately computed full fit; its hash is verified and the freshly generated
    predictions must match it. In Colab, the local reference file is unnecessary.
    ''')
    code('''
    previous_test_prediction,test_prediction,fitted=full_fit(train,test,y,CONTRACT)
    assert np.allclose(predict_final(test,fitted,CONTRACT),test_prediction,rtol=1e-12,atol=.01)
    if cache_root is not None:
        filename='full_prediction_parts.npz'
        assert hashlib.sha256((cache_root/filename).read_bytes()).hexdigest()==manifest['files'][filename]
        independent=np.load(cache_root/filename)
        assert np.allclose(independent['previous'],previous_test_prediction,rtol=1e-12,atol=.01)
        assert np.allclose(independent['candidate'],test_prediction,rtol=1e-12,atol=.01)
        print('PASS: freshly trained final predictions reproduce the independent full fit within INR 0.01.')
    print('Mean absolute change from the submitted CSV (INR):',float(np.abs(test_prediction-previous_test_prediction).mean()))
    print('Candidate range:',float(test_prediction.min()),'to',float(test_prediction.max()))
    ''')
    md('''
    ## 8. Behavior tests and exact submission file format

    File checks establish that all 500 test restaurants receive valid predictions.
    They do not establish prediction accuracy without the hidden target values.
    Deliberately invalid files are rejected to exercise the validation rules.
    ''')
    code('''
    perm=np.random.default_rng(91).permutation(len(test))
    assert np.allclose(predict_final(test.iloc[perm],fitted,CONTRACT),test_prediction[perm],rtol=1e-10,atol=.01)
    edge=test.iloc[:2].copy();edge['City']='unseen_validation_city';edge['Cuisine']=pd.NA
    edge['Facebook Popularity Quotient']=np.nan;edge['Live Music Rating']=np.nan
    edge['Hygiene Rating']=99 # Unseen category in the ordinal representation.
    edge_prediction=predict_final(edge,fitted,CONTRACT)
    assert np.isfinite(edge_prediction).all() and (edge_prediction>=0).all()
    submission=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':test_prediction})
    check_submission(submission,test)
    bad_cases=[submission.iloc[:-1],submission.iloc[::-1].reset_index(drop=True),
               submission.rename(columns={'Annual Turnover':'Turnover'}),submission.assign(index=0)]
    for invalid in [np.nan,np.inf,-1]:
        bad=submission.copy();bad.loc[0,'Annual Turnover']=invalid;bad_cases.append(bad)
    duplicate=submission.copy();duplicate.loc[1,'Registration Number']=duplicate.loc[0,'Registration Number'];bad_cases.append(duplicate)
    for bad in bad_cases:
        rejected=False
        try:check_submission(bad,test)
        except AssertionError:rejected=True
        assert rejected,'Invalid submission was not rejected'
    output_path=OUTPUT_DIR/(FILE_STEM+'.csv')
    submission.to_csv(output_path,index=False)
    roundtrip=pd.read_csv(output_path);check_submission(roundtrip,test)
    assert np.allclose(roundtrip['Annual Turnover'],test_prediction,rtol=1e-12,atol=.001)
    assert len(output_path.read_text().splitlines())==501
    print('PASS: row permutation, unseen/missing features, eight deliberately invalid files, CSV schema and round-trip.')
    print('Exactly 500 rows plus one header; columns:',submission.columns.tolist())
    print('Saved:',output_path.name)
    print('SHA256:',hashlib.sha256(output_path.read_bytes()).hexdigest())
    print(submission.head().to_string(index=False))
    ''')
    md('''
    ## 9. Conclusions and submission

    **Measured outcome:** the candidate worsened both split assignments reserved
    during this round, with an average RMSE increase of **4,388.77**. The city
    stress test worsened by **19,864.08**. Its six-shuffle average improved by
    **6,176.11**, but the conditional interval includes zero. Retain the confirmed
    v3 submission as the preferred model; v4 is an experimental alternative.

    - The submitted v3 model's portal RMSE is **12,094,600.3216516**. That is the
      observed result for the previous file; it must not be assigned to this new CSV.
    - The new candidate was selected using local development RMSE. The output above
      reports the fresh-split checks, previously used splits, city stress result and
      conditional uncertainty. Mixed results must be retained in the conclusion.
    - Fewer numeric thresholds and categorical ratings are plausible variance
      controls. The tests show whether their benefit survives new fold assignments;
      they do not prove a benefit on a different restaurant population.
    - Training and validation share the same labeled source dataset across repeated
      experiments. No hidden labels, competitor predictions or test-score-derived
      prediction adjustments are used. Further search can overfit development data.
    - To beat the leader in the latest supplied screenshot, a new submission must
      achieve **below 11,946,659.0702364**, assuming the leader's score is unchanged.
      This candidate's portal score and rank remain unknown. Local gains cannot be
      subtracted from the prior portal RMSE to predict its next result.

    Retain the confirmed v3 submission if the comparisons above do not support a
    replacement. If you choose to evaluate this experimental candidate, upload its
    **CSV** to **assignment M.Tech. MLA Hackathon -Restaurant Turnover Prediction**, then
    retain the returned score with that file's SHA256. Upload the
    final executed **notebook** to **LLJ_Oct_2026** when model selection is complete;
    the supplied assignment instructions permit only one notebook submission.
    Review the code and inferences so you can explain the modeling choices.
    ''')
    code('''
    try:
        from google.colab import files
        files.download(str(output_path))
    except ImportError:
        print('Prediction CSV is ready in the notebook folder.')
    ''')
    book=nb.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
        'colab':{'name':STEM+'.ipynb','provenance':[]}})
    for cell in book.cells:
        if cell.cell_type=='code': compile(cell.source,'<notebook-cell>','exec')
    nb.validate(book)
    path=ROOT/(STEM+'.ipynb');nb.write(book,path)
    print('Built notebook:',path)
    return path

if __name__=='__main__':
    build()
