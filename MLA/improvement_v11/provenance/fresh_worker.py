"""Fresh standalone inference: verified own V7 state, freshly trained TabPFN models."""
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
import features
import modeling
import verify
from core import ROOT,Engine,sha,save,validate_csv,STEM


def main():
    started=time.monotonic()
    package=json.loads((ROOT/'standalone_manifest.json').read_text())
    for name,expected in package['files'].items():assert sha(ROOT/name)==expected,name
    data=Path(os.environ['MLA_DATA_DIR']);out=Path(os.environ['MLA_V10_OUTPUT_DIR'])
    out.mkdir(parents=True,exist_ok=True)
    e=Engine(data,out/'fresh_context')
    assert e.inputs==package['inputs']
    path=ROOT/'reference/verified_v7.pkl';assert sha(path)==package['v7_checkpoint_sha256']
    with path.open('rb') as f:fitted_base=pickle.load(f)
    b=baseline.predict(fitted_base,e.test)
    expected_b=pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float)
    difference_b=float(np.max(np.abs(b-expected_b)));assert difference_b<.01
    t9_fit=modeling.fit(e.train,e.y,'BASE',print);t=modeling.predict(t9_fit,e.test)
    v9=.5*b+.5*t
    expected_v9=pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float)
    difference_v9=float(np.max(np.abs(v9-expected_v9)));assert difference_v9<.01
    choice=package['choice']
    p=v9;candidate_checks=None
    if choice is not None:
        fitted=modeling.fit(e.train,e.y,choice['name'],print)
        c=modeling.predict(fitted,e.test);w=choice['replacement_weight']
        p=.5*b+.5*((1-w)*t+w*c)
        candidate_checks=verify.inference_checks(lambda raw:modeling.predict(fitted,raw),e.test,features.SPECS[choice['name']]['prune'])
    expected=pd.read_csv(ROOT/'reference/expected.csv')['Annual Turnover'].to_numpy(float)
    difference=float(np.max(np.abs(p-expected)));assert difference<.01,f'Fresh prediction mismatch: INR {difference}'
    frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p})
    validate_csv(frame,e.test);verify.csv_checks(e,p)
    destination=out/(STEM+'.csv' if choice is not None else 'V9_Reproduced.csv')
    frame.to_csv(destination,index=False)
    validate_csv(pd.read_csv(destination),e.test)
    assert len(destination.read_text().splitlines())==501
    assert np.max(np.abs(pd.read_csv(destination)['Annual Turnover'].to_numpy(float)-p))<.01
    record={'status':'PASS','inputs':e.inputs,'choice':choice,'v7_portable_difference_INR':difference_b,
        'fresh_v9_difference_INR':difference_v9,'maximum_final_difference_INR':difference,
        'v7_freshly_retrained_in_colab':False,'tabpfn_components_freshly_fitted':True,
        'candidate_behavior':candidate_checks,'execution_contract':modeling.execution_contract(),
        'csv_sha256':sha(destination),'seconds':time.monotonic()-started,'portal_uploads':0}
    save(out/'fresh_execution_checks.json',record)
    print('PASS: fresh standalone CSV reproduction within INR 0.01:',difference,flush=True)
    print('CSV_SHA256',record['csv_sha256'],flush=True)


if __name__=='__main__':main()
