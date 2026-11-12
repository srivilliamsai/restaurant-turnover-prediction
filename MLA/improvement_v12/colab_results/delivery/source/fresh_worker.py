"""Fresh V12 inference; expected CSV is opened only after predictions exist."""
import json,os,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline,modeling,verify
from core import ROOT,STEM,Engine,sha,save,validate_csv

def main():
    start=time.monotonic();package=json.loads((ROOT/'standalone_manifest.json').read_text())
    for name,want in package['files'].items():assert sha(ROOT/name)==want,name
    out=Path(os.environ['MLA_V12_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    e=Engine(Path(os.environ['MLA_DATA_DIR']),out/'fresh_context');assert e.inputs==package['inputs']
    path=ROOT/'reference/verified_v7.pkl';assert sha(path)==package['v7_checkpoint_sha256']
    with path.open('rb') as f:original=pickle.load(f)
    b=baseline.predict(original,e.test);db=float(np.max(np.abs(b-pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float))));assert db<.01
    fit=modeling.fit_atomic(e.train,e.y,'BASE',print);t=modeling.predict_atomic(fit,e.test);modeling.release(fit)
    v9=.5*b+.5*t;dv=float(np.max(np.abs(v9-pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float))));assert dv<.01
    choice=package['choice'];fit=modeling.fit(e.train,e.y,choice['name'],print)
    behavior=verify.inference_checks(lambda raw:np.maximum(modeling.predict(fit,raw),0.),e.test)
    c=modeling.predict(fit,e.test);modeling.release(fit);p=modeling.combine(b,t,c,choice['name'],choice['weight'])
    expected=pd.read_csv(ROOT/'reference/expected.csv')['Annual Turnover'].to_numpy(float)
    difference=float(np.max(np.abs(p-expected)));assert difference<.01
    checks=verify.csv_checks(e,p);frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p});validate_csv(frame,e.test)
    path=out/(STEM+'.csv');frame.to_csv(path,index=False);validate_csv(pd.read_csv(path),e.test);assert len(path.read_text().splitlines())==501
    save(out/'fresh_execution_checks.json',{'status':'PASS','inputs':e.inputs,'choice':choice,'fresh_v9_difference_INR':dv,'v7_portable_difference_INR':db,
        'maximum_final_difference_INR':difference,'candidate_behavior':behavior,'csv_checks':checks,
        'execution_contract':json.loads((ROOT/'resolved_models.json').read_text()),'csv_sha256':sha(path),'seconds':time.monotonic()-start,
        'v7_freshly_retrained_in_colab':False,'new_components_freshly_fitted':True,'portal_uploads':0})
    print('PASS: fresh V12 notebook reproduces CSV within INR 0.01; maximum difference',difference,flush=True)

if __name__=='__main__':main()
