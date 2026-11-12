"""Real-model verification of both P settings, without adding selection settings."""
import os
import sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path('/content/mla_v10')
sys.path.insert(0,str(root))
import numpy as np
import features
import modeling
from core import Engine,save
import json
allowance=json.loads((root/'results/compute_budget.json').read_text())['maximum_hours']
e=Engine(root/'data',root/'results',hours=allowance)
a,b=e.splits(42)[0]
results={}
with e.cache.charge('export'):
    for name in ['P','C+S+P']:
        fitted=modeling.fit(e.train.iloc[a],e.y[a],name,print)
        valid=e.train.iloc[b].iloc[:12].copy()
        prediction=modeling.predict(fitted,valid)
        changed=valid.copy();changed[features.REMOVED]=1e30
        changed['Annual Turnover']=-1e30;changed['Registration Number']=-1
        difference=float(np.max(np.abs(modeling.predict(fitted,changed)-prediction)))
        assert difference<.01
        results[name]={'status':'PASS','maximum_prediction_difference_INR':difference,'training_rows':len(a)}
save(e.out/'pruned_model_verification.json',{'status':'PASS','settings':results,'sources':e.sources,
    'additional_parameter_settings':0,'purpose':'Required real-model input-invariance verification only.'})
print('PASS: both real P models ignore the removed raw inputs and validation labels/IDs.',flush=True)
