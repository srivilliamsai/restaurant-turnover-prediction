"""GPU control engineering timing and existing-fold reproduction only."""
import json,time
import numpy as np
from core import ROOT,Engine,save
import modeling

def main():
 e=Engine(ROOT/'data');a,b=e.splits(42)[0];config=modeling.resolve();parts=e.reference_parts(42);records={}
 for name,key in [('BASE','t9'),('P35_8','p35'),('UNADAPTED',None)]:
  started=time.monotonic();fit=modeling.fit(e.train.iloc[a],e.y[a],name)
  try:p=modeling.predict(fit,e.train.iloc[b])
  finally:modeling.release(fit)
  elapsed=time.monotonic()-started
  d=float(np.max(np.abs(p-parts[key][b]))) if key else None
  if d is not None:assert d<.01,(name,d)
  records[name]={'status':'PASS','seconds':elapsed,'maximum_existing_fold_difference_INR':d}
 save(e.out/'control_pilot.json',records)
 print('GPU control pilot PASS',flush=True)
if __name__=='__main__':main()
