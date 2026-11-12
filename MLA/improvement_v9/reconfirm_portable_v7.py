"""Reconfirm the frozen V9 choice against fresh V7 folds on its original platform."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import modeling
import workflow
from engine import Engine, rmse, sha, save

root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
source=run/'partial_backup/results'
out=run/'portability_confirmation'
e=Engine(root.parent,out)
lock=json.loads((source/'candidate_lock.json').read_text())
assert sha(source/'candidate_lock.json')=='0be8a79c77c85b6b5c29a3efcb238e2d8d9d147e1ae2025e8e218a5246ad9ba5'
original=json.loads((source/'decision.json').read_text())
assert len(original['trials'])==1
choice=original['trials'][0]
assert (choice['name'],choice['mode'],choice['weight'])==('tabpfn_raw_v7','ensemble16',.5)
pilot=json.loads((source/'pilot.json').read_text())
execution=pilot['families']['tabpfn']['execution_contract']
baselines={};predictions={};metrics={};segments=[];folds=[]
started=time.monotonic()
for scheme in [48211,48212,48213,'strat:27421','city']:
    e.cache.stage='confirmation_portability_verification'
    b=e.baseline(scheme)
    candidate=np.zeros(len(e.y))
    for number,(a,indices) in enumerate(e.splits(scheme)):
        extra={'scheme':str(scheme),'fold':number,'mode':choice['mode'],'spec':modeling.SPECS[choice['name']]}
        stem=f"{choice['name']}_{choice['mode']}_{str(scheme).replace(':','_')}_fold{number}"
        path=source/'cache'/(stem+'.npz')
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==metadata['sha256']
        assert metadata['context']==e.context(choice['name'],a,indices,extra,execution)
        values=np.load(path,allow_pickle=False)['prediction']
        assert values.shape==(len(indices),)
        candidate[indices]=values
    p=(1-choice['weight'])*b+choice['weight']*candidate
    old,new=rmse(e.y,b),rmse(e.y,p)
    metrics[str(scheme)]={'v7_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_change':new/old-1}
    baselines[scheme]=b;predictions[scheme]=p
    np.savez_compressed(out/(str(scheme).replace(':','_')+'.npz'),baseline=b,component=candidate,prediction=p)
    total=float(np.sum((e.y-b)**2-(e.y-p)**2))
    for label,mask in [('<=40M',e.y<=40e6),('40-80M',(e.y>40e6)&(e.y<=80e6)),('>80M',e.y>80e6)]:
        reduction=float(np.sum((e.y[mask]-b[mask])**2-(e.y[mask]-p[mask])**2))
        segments.append({'scheme':str(scheme),'segment':label,'rows':int(mask.sum()),'v7_rmse':rmse(e.y[mask],b[mask]),'candidate_rmse':rmse(e.y[mask],p[mask]),'sse_reduction':reduction,'fraction_of_total_reduction':reduction/total})
    for number,(_,indices) in enumerate(e.splits(scheme)):
        folds.append({'scheme':str(scheme),'fold':number,'rows':len(indices),'v7_rmse':rmse(e.y[indices],b[indices]),'candidate_rmse':rmse(e.y[indices],p[indices])})
    save(out/'progress.json',{'status':'running','completed_assignments':list(metrics),'metrics':metrics})
    print('CORRECTED_CONFIRMATION',str(scheme),json.dumps(metrics[str(scheme)]),flush=True)
qualified=all(metrics[str(s)]['gain']>0 for s in [48211,48212,48213]) and all(metrics[s]['relative_change']<=.01 for s in ['strat:27421','city'])
interval=workflow.uncertainty(e.y,baselines,predictions)
record={'status':'completed','qualified':bool(qualified),'selection_unchanged':True,
        'choice':{k:choice[k] for k in ['name','mode','weight','family']},
        'lock_sha256':sha(source/'candidate_lock.json'),'inputs':e.inputs,'sources':e.sources,
        'baseline_platform':{'python':platform.python_version(),'machine':platform.machine(),'system':platform.system(),'runtime':e.runtime},
        'tabpfn_execution_contract':execution,'metrics':metrics,'uncertainty':interval,
        'mean_random_gain':float(np.mean([metrics[str(s)]['gain'] for s in [48211,48212,48213]])),
        'seconds':time.monotonic()-started,'portal_uploads':0}
pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
save(out/'decision.json',record)
save(out/'progress.json',{'status':'completed','qualified':bool(qualified),'metrics':metrics})
print('CORRECTED_DECISION',json.dumps(record),flush=True)
