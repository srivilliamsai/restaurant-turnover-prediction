"""Run the frozen V9 ensemble with an exact, independently reproduced V7 checkpoint."""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
import modeling
from engine import Engine, check_behavior, validate_csv

work=Path(__file__).resolve().parent
manifest=json.loads((work/'portable_manifest.json').read_text())
data=Path(os.environ['V9_PORTABLE_DATA_DIR'])
output=Path(os.environ['V9_PORTABLE_OUTPUT_DIR'])
output.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
for name,expected in manifest['source_hashes'].items():
    assert hashlib.sha256((work/name).read_bytes()).hexdigest()==expected,name
e=Engine(data,work/'fresh_results')
assert e.inputs==manifest['inputs']
model_path=work/'verified_v7.pkl'
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==manifest['v7_checkpoint_sha256']
# This checkpoint was created by our separate fresh fit on the original platform,
# reproduced against the scored file, and verified by hash before loading.
with model_path.open('rb') as f:fitted_base=pickle.load(f)
b=baseline.predict(fitted_base,e.test)
expected_base=pd.read_csv(work/'scored_v7.csv')['Annual Turnover'].to_numpy(float)
baseline_difference=float(np.max(np.abs(b-expected_base)))
assert baseline_difference<.01
base_checks=check_behavior(lambda raw:baseline.predict(fitted_base,raw),e.test,b)
print('PASS: verified V7 checkpoint reproduces scored predictions:',baseline_difference,flush=True)
choice=manifest['choice']
assert (choice['name'],choice['mode'],choice['weight'])==('tabpfn_raw_v7','ensemble16',.5)
contract=modeling.execution_contract()
expected_contract=manifest['tabpfn_execution_contract']
assert contract==expected_contract, 'Use the exact recorded free T4 / Torch / CUDA environment.'
fitted=modeling.fit_model(e.train,e.y,modeling.SPECS[choice['name']],choice['mode'],print)
c=modeling.predict_model(fitted,e.test)
p=(1-choice['weight'])*b+choice['weight']*c
expected=pd.read_csv(work/'expected.csv')['Annual Turnover'].to_numpy(float)
difference=float(np.max(np.abs(p-expected)))
assert difference<.01, f'Fresh final prediction difference exceeds INR 0.01: {difference}'
component_checks=check_behavior(lambda raw:modeling.predict_model(fitted,raw),e.test,c)
frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p})
validate_csv(frame,e.test)
path=output/'submission.csv'
frame.to_csv(path,index=False)
validate_csv(pd.read_csv(path),e.test)
assert len(path.read_text().splitlines())==501
assert np.max(np.abs(pd.read_csv(path)['Annual Turnover'].to_numpy()-p))<.01
record={'status':'PASS','inputs':e.inputs,'sources':e.sources,
        'choice':choice,'execution_contract':contract,
        'v7_checkpoint_sha256':manifest['v7_checkpoint_sha256'],
        'baseline_fresh_training_platform':manifest['baseline_fresh_training_platform'],
        'baseline_fresh_reproduction_difference_INR':manifest['baseline_fresh_reproduction_difference_INR'],
        'baseline_portable_inference_difference_INR':baseline_difference,
        'maximum_final_difference_INR':difference,
        'v7_freshly_retrained_in_this_Colab_process':False,
        'tabpfn_freshly_fitted_in_this_process':True,
        'baseline_behavior':base_checks,'tabpfn_behavior':component_checks,
        'csv_rows':len(frame),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'seconds':time.monotonic()-started,'portal_uploads':0}
(output/'fresh_execution_checks.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: fresh standalone execution reproduces final CSV within INR 0.01:',difference,flush=True)
print('CSV_SHA256',record['csv_sha256'],flush=True)
print('CSV_PATH',path,flush=True)
