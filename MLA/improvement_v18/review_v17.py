"""Reconstruct V17 evidence from hashes and NumPy; never fit a model or allocate a GPU."""
import csv,hashlib,json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V17=PROJECT/'improvement_v17'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def main():
 with (PROJECT/'Train_dataset.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 y=np.asarray([float(r['Annual Turnover']) for r in rows])
 ids=[int(r['Registration Number']) for r in rows]
 registry=read(V17/'baseline_registry.json')
 frozen=read(V17/'source_frozen.json')
 old=read(V17/'provenance/v14_source_frozen.json')
 inputs={n:sha(PROJECT/n) for n in registry['inputs']}
 assert inputs==registry['inputs']==frozen['inputs']==old['inputs']
 assert digest(ids)==registry['row_ids_sha256']
 for n,h in frozen['sources'].items():assert sha(V17/n)==h
 for n,h in old['sources'].items():assert sha(V17/'provenance/v14_sources'/n)==h
 portal=read(V17/'reference/v14_portal_trial.json')
 for suffix,key in [('.csv','csv_sha256'),('.ipynb','notebook_sha256')]:
  assert sha(V17/'reference'/('scored_v14'+suffix))==portal[key]
 with (V17/'results/screening.csv').open(newline='') as f:screen=list(csv.DictReader(f))
 results={};verified=[];max_difference=0.
 for seed in [42,71]:
  item=registry['schemes'][str(seed)]
  assert sha(V17/item['file'])==item['sha256']
  with np.load(V17/item['file'],allow_pickle=False) as z:
   v9=z['v9'].copy();assert np.max(np.abs(v9-.5*z['v7']-.5*z['t9']))<.01
  p35=np.zeros(len(y));nori=np.zeros(len(y));seen=np.zeros(len(y),int)
  for fold,indices in enumerate(np.array_split(np.random.RandomState(seed).permutation(len(y)),5)):
   b=np.sort(indices);a=np.setdiff1d(np.arange(len(y)),b);seen[b]+=1
   for directory,name,out,sources in [('provenance/v14_cache','P35_8',p35,old['sources']),('results/cache','NOR15',nori,frozen['sources'])]:
    path=V17/directory/f'{name}_{seed}_fold{fold}.npz';meta=read(path.with_suffix('.json'));c=meta['context']
    assert sha(path)==meta['sha256'] and c['inputs']==inputs and c['sources']==sources
    assert c['name']==name and c['train_indices']==digest(a.tolist()) and c['valid_indices']==digest(b.tolist())
    assert c['training_targets']==hashlib.sha256(y[a].tobytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:
     q=z['prediction'];assert q.shape==b.shape and np.isfinite(q).all();out[b]=q
    verified.append({'path':str(path.relative_to(PROJECT)),'sha256':meta['sha256']})
  assert (seen==1).all()
  base=.25*v9+.75*p35;old_rmse=rmse(y,base);comparisons=[]
  top=np.argsort((y-base)**2)[-5:];mask=np.ones(len(y),bool);mask[top]=False
  for w in [.1,.25,.5,.75,1.]:
   q=(1-w)*base+w*nori;value=rmse(y,q)
   saved=next(r for r in screen if float(r['weight'])==w)
   d=abs(value-float(saved[f'rmse_{seed}']));max_difference=max(max_difference,d);assert d<1e-6
   comparisons.append({'weight':w,'rmse_INR':value,'gain_INR':old_rmse-value,'relative_gain':(old_rmse-value)/old_rmse,
    'gain_excluding_five_largest_baseline_errors_INR':rmse(y[mask],base[mask])-rmse(y[mask],q[mask])})
  sse=(y-base)**2
  results[str(seed)]={'v14_rmse_INR':old_rmse,'nori_alone_rmse_INR':rmse(y,nori),
   'residual_correlation':float(np.corrcoef(y-base,y-nori)[0,1]),'comparisons':comparisons,
   'tail_gt80M_SSE_share':float(sse[y>8e7].sum()/sse.sum()),'largest_five_SSE_share':float(sse[top].sum()/sse.sum())}
 best=max(range(5),key=lambda j:np.mean([results[str(s)]['comparisons'][j]['relative_gain'] for s in [42,71]]))
 gains=[results[str(s)]['comparisons'][best]['relative_gain'] for s in [42,71]]
 assert np.mean(gains)<.002 and not read(V17/'results/decision.json')['qualified']
 pilot=read(V17/'results/pilot.json');f=pilot['settings']['P35FT512'];t=f['fold_seconds']
 evidence={'status':'PASS','checked_at_IST':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
  'scope':'Previously completed outputs only; no V18 accuracy testing, model download, fit or GPU allocation.',
  'inputs':inputs,'training_rows':len(y),'all_original_targets_retained':True,
  'verified_fold_outputs':verified,'maximum_metric_difference_INR':max_difference,
  'v14_portal_rmse_INR':portal['actual_portal_rmse_INR'],'v14_csv_sha256':portal['csv_sha256'],'v14_notebook_sha256':portal['notebook_sha256'],
  'v17_best_new_model_weight':results['42']['comparisons'][best]['weight'],
  'v17_best_mean_relative_gain':float(np.mean(gains)),'v17_qualified':False,
  'development':results,'fine_tuning_pilot':{k:f[k] for k in ['status','engineering_seconds','fold_seconds','full_fit_predict_seconds','peak_gpu_allocated_bytes','fresh_repeat_difference_INR','save_reload_difference_INR','behavior','target_metadata']},
  'fine_tuning_accuracy_tested_in_V17':False,
  'fine_tuning_historical_forecast_hours':{'development_10_plus30pct':t*10*1.3/3600,'confirmation_25_plus30pct':t*25*1.3/3600,'both':t*35*1.3/3600,
   'delivery_including_controls':1.3*(2*f['full_fit_predict_seconds']+1200)/3600},
  'xiaomi_exclusion':pilot['settings']['XIA32']['reason'],
  'source_review_registry_sha256':sha(ROOT/'source_review_registry.json'),'script_sha256':sha(__file__)}
 (ROOT/'review_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'PASS','verified_fold_outputs':len(verified),'best_gain_percent':100*float(np.mean(gains)),
  'historical_FT_confirmation_hours':t*25*1.3/3600,'evidence':str(ROOT/'review_evidence.json')},indent=2))
if __name__=='__main__':main()
