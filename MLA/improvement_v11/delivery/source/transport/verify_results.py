"""Independent metric reconstruction; never fit missing models during review."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import Engine,sha,save,rmse,CONFIRM,DEVELOPMENT
import workflow

e=Engine(ROOT.parent,ROOT/'colab_results/results');pilot=json.loads((e.out/'pilot.json').read_text());lock=json.loads((e.out/'candidate_lock.json').read_text());decision=json.loads((e.out/'decision.json').read_text())
def cached(name,scheme):
    for f,(a,b) in enumerate(e.splits(scheme)):
        p=e.cache.root/f'{name}_{str(scheme).replace(":","_")}_fold{f}.npz';assert p.exists() and p.with_suffix('.json').exists(),'Missing checkpoint, review cannot fit it'
    return e.component(name,scheme)
rows=[];maximum=0.;complementarity=[]
bases={s:e.reference(s) for s in DEVELOPMENT}
for name in pilot['screen_settings']:
    cp={s:cached(name,s) for s in DEVELOPMENT}
    for s in DEVELOPMENT:
        be=e.y-bases[s];ce=e.y-cp[s]
        correlation=float(np.corrcoef(be,ce)[0,1]) if np.std(be)>0 and np.std(ce)>0 else None
        complementarity.append({'setting':name,'scheme':s,'component_rmse':rmse(e.y,cp[s]),'v9_rmse':rmse(e.y,bases[s]),'error_correlation':correlation})
    for w in [.25,.5,.75,1.]:
        row={'name':name,'weight':w,'estimators':int(name[1:])}
        for s in DEVELOPMENT:
            new=rmse(e.y,(1-w)*bases[s]+w*cp[s]);old=rmse(e.y,bases[s]);row[f'rmse_{s}']=new;row[f'gain_{s}']=old-new;row[f'relative_gain_{s}']=(old-new)/old
        row['mean_rmse']=float(np.mean([row[f'rmse_{s}'] for s in DEVELOPMENT]));row['mean_relative_gain']=float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]));rows.append(row)
if rows:
    recorded=pd.read_csv(e.out/'screening.csv')
    for row in rows:
        old=recorded[(recorded.name==row['name'])&(recorded.weight==row['weight'])].iloc[0]
        for field in ['rmse_42','rmse_71','gain_42','gain_71','mean_rmse','mean_relative_gain']:maximum=max(maximum,abs(float(old[field])-row[field]))
eligible=[r for r in rows if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002];eligible.sort(key=lambda r:(r['mean_rmse'],r['weight'],r['estimators'],r['name']))
choice=eligible[0] if eligible else None
assert (choice is None)==(lock['choice'] is None)
if choice:
    for key in ['name','weight','estimators']:assert choice[key]==lock['choice'][key]
    bases={};predictions={};gains=[];relative=[]
    for scheme in CONFIRM:
        if scheme in CONFIRM[:3]:cached('BASE',scheme)
        base=e.reference(scheme);c=cached(choice['name'],scheme);p=(1-choice['weight'])*base+choice['weight']*c
        b,n=rmse(e.y,base),rmse(e.y,p);m=decision['metrics'][str(scheme)];maximum=max(maximum,abs(m['v9_rmse']-b),abs(m['candidate_rmse']-n));bases[scheme]=base;predictions[scheme]=p;gains.append(b-n);relative.append(1-n/b)
    interval=workflow.uncertainty(e.y,bases,predictions)
    assert np.allclose(interval['ci95_INR'],decision['uncertainty']['ci95_INR'],rtol=0,atol=.01)
    qualified=all(g>0 for g in gains[:3]) and np.mean(relative[:3])>=.002 and interval['ci95_INR'][0]>0 and all(g>=-.01 for g in relative[3:])
else:qualified=False
assert bool(qualified)==decision['qualified'] and maximum<.01
save(e.out/'model_complementarity.json',complementarity)
save(e.out/'independent_review.json',{'status':'PASS','configurations_recomputed':len(rows),'frozen_selection_matches':True,'qualification_recomputed':True,'qualified':bool(qualified),'maximum_metric_difference_INR':maximum,'inputs':e.inputs,'sources':e.sources})
print('Independent V11 selection, metrics and qualification PASS.',flush=True)
