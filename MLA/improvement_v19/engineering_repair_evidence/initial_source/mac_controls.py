"""Frozen V7 controls fitted only on the original Mac, never selected by scores."""
import argparse,hashlib,importlib.metadata,json,platform,time
from pathlib import Path
import numpy as np
from core import ROOT,Engine,sha,digest,save,log,CONFIRM
NAMES=['baseline.py','legacy_models.py','legacy_reference.py','v5_contract.json']
def context(e,scheme,fold,a,b):
 return {'inputs':e.inputs,'sources':{n:sha(ROOT/n) for n in NAMES},'platform':{'system':'Darwin','machine':'arm64'},
  'runtime':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','catboost','interpret-core']},
  'scheme':str(scheme),'fold':fold,'train_indices':digest(a.tolist()),'valid_indices':digest(b.tolist()),
  'train_ids':digest(e.train.iloc[a]['Registration Number'].tolist()),'valid_ids':digest(e.train.iloc[b]['Registration Number'].tolist()),
  'training_targets':hashlib.sha256(e.y[a].tobytes()).hexdigest()}
def read_fold(e,scheme,fold,a,b):
 path=ROOT/'mac_controls'/f'v7_{scheme}_fold{fold}.npz';meta=json.loads(path.with_suffix('.json').read_text())
 c=meta['context'];expected=context(e,scheme,fold,a,b)
 assert c==expected and sha(path)==meta['sha256'],'Invalid original-Mac control'
 with np.load(path,allow_pickle=False) as z:p=z['prediction'].copy()
 assert p.shape==b.shape and np.isfinite(p).all()
 return p
def compute(e,scheme,fold,pilot=False):
 assert platform.system()=='Darwin' and platform.machine()=='arm64'
 import baseline
 a,b=e.splits(scheme)[fold];ctx=context(e,scheme,fold,a,b)
 path=ROOT/'mac_controls'/f'v7_{scheme}_fold{fold}.npz';path.parent.mkdir(exist_ok=True)
 if path.exists():return read_fold(e,scheme,fold,a,b)
 assert e.budget.remaining('pilot' if pilot else 'confirm')>0
 started=time.monotonic();contract=json.loads((ROOT/'v5_contract.json').read_text())
 pred=baseline.fold_prediction(e.train.iloc[a],e.y[a],e.train.iloc[b],contract,fold,log)
 temp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(temp,prediction=pred);temp.replace(path)
 save(path.with_suffix('.json'),{'context':ctx,'sha256':sha(path),'seconds':time.monotonic()-started})
 return pred
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','confirm']);args=p.parse_args();e=Engine()
 if args.mode=='pilot':
  a,b=e.splits(42)[0];started=time.monotonic();pred=compute(e,42,0,True)
  witness=e.registry['schemes']['42']
  with np.load(ROOT/witness['file'],allow_pickle=False) as z:difference=float(np.max(np.abs(pred-z['v7'][b])))
  assert difference<.01,f'Original-Mac V7 reproduction differs INR {difference}'
  save(ROOT/'transport_run/mac_pilot.json',{'status':'PASS','seconds':time.monotonic()-started,'maximum_difference_INR':difference})
 else:
  lock=json.loads((ROOT/'colab_results/results/candidate_lock.json').read_text());assert lock['complete'] and lock['choices']
  assert lock['sources']==e.sources and lock['inputs']==e.inputs
  for scheme in CONFIRM[:3]:
   for fold in range(5):compute(e,scheme,fold);log({'mac_control':scheme,'fold_complete':fold+1})
  save(ROOT/'transport_run/mac_complete.json',{'status':'PASS','folds':15})
if __name__=='__main__':main()
