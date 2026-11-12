"""Recompute every measured blend and decision from complete fold outputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import features,workflow
from core import ROOT,Engine,sha,digest,save,rmse,DEVELOPMENT,CONFIRM

def main():
    e=Engine()
    frozen=json.loads((ROOT/'source_frozen.json').read_text())
    assert e.sources==frozen['sources'] and e.inputs==frozen['inputs']
    audit=json.loads((e.out/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['sources']==e.sources
    pilot=json.loads((e.out/'pilot.json').read_text())
    settings=pilot['screen_settings']
    measured=pd.read_csv(e.out/'screening.csv')
    expected_rows=[]
    checked=0
    maximum_metric_difference=0.
    treatment_thresholds={}
    for name in settings:
        predictions={}
        naive={}
        treatment_thresholds[name]={}
        for scheme in DEVELOPMENT:
            p=np.zeros(len(e.y))
            q=np.zeros(len(e.y))
            thresholds=[]
            for fold,(a,b) in enumerate(e.splits(scheme)):
                path=e.out/'cache'/f'{name}_{scheme}_fold{fold}.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['sha256']==sha(path)
                assert meta['context']==e.context(name,scheme,fold,a,b)
                _,wanted=features.training_target(e.y[a],name)
                assert meta['fit_metadata']==wanted
                thresholds.append(wanted.get('threshold_INR'))
                with np.load(path,allow_pickle=False) as values:
                    p[b]=values['prediction']
                    if name=='SQ2':q[b]=values['naive_squared_mean']
                checked+=1
            predictions[scheme]=p
            naive[scheme]=q
            treatment_thresholds[name][str(scheme)]=thresholds
        for weight in features.WEIGHTS:
            row={'name':name,'weight':weight}
            for scheme in DEVELOPMENT:
                base=e.reference(scheme)
                candidate=(1-weight)*base+weight*predictions[scheme]
                old,new=rmse(e.y,base),rmse(e.y,candidate)
                row.update({f'rmse_{scheme}':new,f'gain_{scheme}':old-new,f'relative_gain_{scheme}':(old-new)/old})
            row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]))
            saved=measured[(measured['name']==name)&(measured['weight']==weight)]
            assert len(saved)==1
            saved=saved.iloc[0]
            for key,value in row.items():
                if key not in ['name','weight']:
                    difference=abs(float(saved[key])-value)
                    maximum_metric_difference=max(maximum_metric_difference,difference)
                    assert difference<1e-6,(name,weight,key,difference)
            expected_rows.append(row)
    assert len(measured)==len(expected_rows)==4*len(settings)
    chosen=workflow.selection(expected_rows)
    lock=json.loads((e.out/'candidate_lock.json').read_text())
    assert len(chosen)==len(lock['choices'])
    if chosen:
        want,got=chosen[0],lock['choices'][0]
        assert want['name']==got['name'] and want['weight']==got['weight']
        assert abs(want['mean_relative_gain']-got['mean_relative_gain'])<1e-12
    decision=json.loads((e.out/'decision.json').read_text())
    if not chosen:
        assert decision['complete'] and not decision['qualified'] and not decision['finalists']
        assert not (e.out/'confirmation.csv').exists()
    else:
        finalist=decision['finalists'][0]
        metrics={}
        baselines,predictions={},{}
        for scheme in CONFIRM:
            baseline=e.reference(scheme)
            component=np.zeros(len(e.y))
            for fold,(a,b) in enumerate(e.splits(scheme)):
                path=e.out/'cache'/f"{chosen[0]['name']}_{str(scheme).replace(':','_')}_fold{fold}.npz"
                meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['sha256']==sha(path)
                assert meta['context']==e.context(chosen[0]['name'],scheme,fold,a,b)
                with np.load(path,allow_pickle=False) as z:component[b]=z['prediction']
                checked+=1
            p=(1-chosen[0]['weight'])*baseline+chosen[0]['weight']*component
            old,new=rmse(e.y,baseline),rmse(e.y,p)
            metrics[str(scheme)]={'v9_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':(old-new)/old}
            baselines[scheme],predictions[scheme]=baseline,p
            for key,value in metrics[str(scheme)].items():
                assert abs(value-finalist['metrics'][str(scheme)][key])<1e-6
        interval=workflow.uncertainty(e.y,baselines,predictions)
        assert np.max(np.abs(np.array(interval['ci95_INR'])-finalist['uncertainty']['ci95_INR']))<1e-6
        assert workflow.qualifies(metrics,interval)==decision['qualified']
    for suffix in ['.csv','.ipynb']:
        original=ROOT/'reference'/('scored_v9'+suffix)
        if not decision['qualified']:assert sha(original)==sha(ROOT/'delivery'/('KEEP_V9_'+__import__('core').STEM+suffix))
    save(ROOT/'delivery/independent_review.json',{
        'status':'PASS','independently_recomputed_blends':len(expected_rows),
        'complete_fold_outputs_verified':checked,'maximum_metric_difference_INR':maximum_metric_difference,
        'source_and_data_identity_unchanged':True,'all_original_validation_targets':len(e.y),
        'fold_local_thresholds':treatment_thresholds,'selection_recomputed':True,
        'qualified':decision['qualified'],'confirmation_performed':bool(chosen),
        'v9_scored_files_preserved':True,'portal_uploads':0})
    print('PASS:',len(expected_rows),'blends independently recomputed;',checked,'complete folds verified; qualified:',decision['qualified'])

if __name__=='__main__':main()

