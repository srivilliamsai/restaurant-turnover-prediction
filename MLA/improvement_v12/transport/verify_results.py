"""Independent cached-array reconstruction; missing folds can never trigger fitting."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import Engine,sha,save,rmse,CONFIRM,DEVELOPMENT
import workflow,features

def prediction(name,w,v7,t9,c):
    base=.5*v7+.5*t9
    return np.maximum(.5*v7+.5*((1-w)*t9+w*c) if name.startswith('T') else (1-w)*base+w*c,0.)

def main():
    e=Engine(ROOT.parent,ROOT/'colab_results/results');pilot=json.loads((e.out/'pilot.json').read_text());lock=json.loads((e.out/'candidate_lock.json').read_text());decision=json.loads((e.out/'decision.json').read_text())
    assert e.sources==pilot['sources']==lock['sources'] and e.inputs==pilot['inputs']==lock['inputs']
    assert sha(ROOT/'resolved_models.json')==lock['model_contract_sha256']==pilot['model_contract_sha256']
    def cached(name,scheme):
        registry=json.loads((ROOT/'baseline_registry.json').read_text())
        for atom in features.SPECS[name]['atoms']:
            if atom=='BASE' and str(scheme) in registry['schemes']:continue
            for f,_ in enumerate(e.splits(scheme)):
                p=e.cache.root/f'{atom}_{str(scheme).replace(":","_")}_fold{f}.npz'
                assert p.exists() and p.with_suffix('.json').exists(),'Missing cached atom: review cannot train'
        return e.component(name,scheme)
    rows=[];maximum=0.;refs={s:e.reference_parts(s) for s in DEVELOPMENT}
    for name in pilot['screen_settings']:
        cp={s:cached(name,s) for s in DEVELOPMENT}
        for w in [.25,.5,.75,1.]:
            row={'name':name,'family':features.SPECS[name]['family'],'weight':w,'inference_cost':features.SPECS[name]['cost']}
            for scheme in DEVELOPMENT:
                v7,t9=refs[scheme];old=rmse(e.y,.5*v7+.5*t9);new=rmse(e.y,prediction(name,w,v7,t9,cp[scheme]))
                row[f'rmse_{scheme}']=new;row[f'gain_{scheme}']=old-new;row[f'relative_gain_{scheme}']=(old-new)/old
            row['mean_rmse']=float(np.mean([row[f'rmse_{s}'] for s in DEVELOPMENT]));row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]));rows.append(row)
    if rows:
        recorded=pd.read_csv(e.out/'screening.csv');assert len(recorded)==len(rows)
        for row in rows:
            old=recorded[(recorded.name==row['name'])&(recorded.weight==row['weight'])].iloc[0]
            for field in ['rmse_42','rmse_71','gain_42','gain_71','mean_rmse','mean_relative_gain']:maximum=max(maximum,abs(float(old[field])-row[field]))
    rank=lambda r:(r['mean_rmse'],r['weight'],r['inference_cost'],r['name'])
    eligible=[r for r in rows if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002];choices=[]
    for family in ['T','R']:
        family_rows=sorted([r for r in eligible if r['family']==family],key=rank)
        if family_rows:choices.append(family_rows[0])
    choices.sort(key=rank);assert len(choices)==len(lock['choices'])
    for a,b in zip(choices,lock['choices']):assert (a['name'],a['weight'])==(b['name'],b['weight'])
    qualified=[]
    for choice,finalist in zip(choices,decision['finalists']):
        bases={};predictions={};gains=[];relative=[]
        for scheme in CONFIRM:
            v7,t9=e.reference_parts(scheme);base=.5*v7+.5*t9;c=cached(choice['name'],scheme);p=prediction(choice['name'],choice['weight'],v7,t9,c)
            old,new=rmse(e.y,base),rmse(e.y,p);m=finalist['metrics'][str(scheme)]
            maximum=max(maximum,abs(m['v9_rmse']-old),abs(m['candidate_rmse']-new));bases[scheme]=base;predictions[scheme]=p;gains.append(old-new);relative.append(1-new/old)
        interval=workflow.uncertainty(e.y,bases,predictions);assert np.allclose(interval['ci95_INR'],finalist['uncertainty']['ci95_INR'],rtol=0,atol=.01)
        passing=all(g>0 for g in gains[:3]) and np.mean(relative[:3])>=.002 and interval['ci95_INR'][0]>0 and all(g>=-.01 for g in relative[3:])
        assert bool(passing)==finalist['qualified']
        if passing:qualified.append(choice)
    assert len(decision['finalists'])==len(choices) and bool(qualified)==decision['qualified']
    assert [(c['name'],c['weight']) for c in qualified]==[(c['name'],c['weight']) for c in decision['choices']]
    assert maximum<.01
    save(e.out/'independent_review.json',{'status':'PASS','configurations_recomputed':len(rows),'frozen_selection_matches':True,'qualification_recomputed':True,'qualified':bool(qualified),'maximum_metric_difference_INR':maximum,'inputs':e.inputs,'sources':e.sources})
    print('Independent V12 selection, both formulas, metrics and qualification PASS.',flush=True)

if __name__=='__main__':main()
