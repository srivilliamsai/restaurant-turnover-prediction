"""Fresh local V7 reproduction with per-component comparisons to Colab."""
import hashlib
import json
import pickle
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
import legacy_models as lm
import legacy_reference as ref

root=Path(__file__).resolve().parent
target=root/'colab_cli_runs/20261026T160148Z/export_diagnostics'
train,test,y=lm.load_data(root.parent)
contract=json.loads((root/'v5_contract.json').read_text())
print('LOCAL_FULL_REPRODUCTION_START',flush=True)
p,fit=baseline.fit_full(train,test,y,contract,lambda x:print(x,flush=True))
expected=pd.read_csv(root/'scored_v7_backup/submission.csv')['Annual Turnover'].to_numpy(float)
np.savez_compressed(target/'local_full_reproduction.npz',prediction=p,expected=expected)
with (target/'local_v7_fit.pkl').open('wb') as f:pickle.dump(fit,f)
predictions={}
for name,fitted in fit['legacy']['v2'].items():predictions['v2__'+name]=ref.predict_fitted(fitted,ref.features(test,contract['v2_configs'][name]['variant']))
for i,fitted in enumerate(fit['legacy']['v3']):predictions['v3__'+str(i)]=np.maximum(fitted.predict(ref.features(test,'base')),0)
for name,models in fit['legacy']['new'].items():
    for i,fitted in enumerate(models):predictions['new__'+name+'__'+str(i)]=ref.predict_v5(fitted,test)
b=ref.predict_final(test,fit['legacy'],contract)
predictions['v5']=b
for i,fitted in enumerate(fit['models']):predictions['residual__'+str(i)]=lm.predict_component(fitted,test,b)
predictions['v7']=lm.predict_submission(fit,test)
np.savez_compressed(target/'local_components.npz',**predictions)
cloud=np.load(target/'cloud_components.npz',allow_pickle=False)
rows=[]
for name,value in predictions.items():
    diff=value-cloud[name]
    rows.append({'component':name,'maximum_difference_INR':float(np.abs(diff).max()),'rms_difference_INR':float(np.sqrt(np.mean(diff**2)))})
report={'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
        'maximum_difference_from_scored_V7_INR':float(np.abs(p-expected).max()),
        'feature_frame_sha256':hashlib.sha256(pd.util.hash_pandas_object(ref.features(train),index=True).values.tobytes()).hexdigest(),
        'target_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
        'components':rows,'v3_params':fit['legacy']['v3'][0].get_all_params(),
        'residual_params':fit['models'][0]['model'].get_all_params()}
(target/'local_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'maximum_difference_from_scored_V7_INR':report['maximum_difference_from_scored_V7_INR'],'components':rows},indent=2),flush=True)
