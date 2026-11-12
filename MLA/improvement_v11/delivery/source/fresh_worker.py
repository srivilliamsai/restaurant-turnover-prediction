"""Fresh selected-model inference; old V7 state remains explicitly documented."""
import json,os,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline,modeling,verify
from core import ROOT,STEM,Engine,sha,save,validate_csv

def main():
    start=time.monotonic();package=json.loads((ROOT/'standalone_manifest.json').read_text())
    for n,want in package['files'].items():assert sha(ROOT/n)==want,n
    out=Path(os.environ['MLA_V11_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    e=Engine(Path(os.environ['MLA_DATA_DIR']),out/'fresh_context');assert e.inputs==package['inputs']
    path=ROOT/'reference/verified_v7.pkl';assert sha(path)==package['v7_checkpoint_sha256']
    with path.open('rb') as f:original=pickle.load(f)
    b=baseline.predict(original,e.test);old_b=pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'].to_numpy(float)
    db=float(np.max(np.abs(b-old_b)));assert db<.01
    fit=modeling.fit(e.train,e.y,'BASE',print);t=modeling.predict(fit,e.test);modeling.release(fit);base=.5*b+.5*t
    dv=float(np.max(np.abs(base-pd.read_csv(ROOT/'reference/scored_v9.csv')['Annual Turnover'].to_numpy(float))));assert dv<.01
    choice=package['choice'];fit=modeling.fit(e.train,e.y,choice['name'],print);c=modeling.predict(fit,e.test)
    behavior=verify.inference_checks(lambda raw:modeling.predict(fit,raw),e.test);modeling.release(fit)
    p=(1-choice['weight'])*base+choice['weight']*c
    expected=pd.read_csv(ROOT/'reference/expected.csv')['Annual Turnover'].to_numpy(float)
    difference=float(np.max(np.abs(p-expected)));assert difference<.01
    verify.csv_checks(e,p);frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p});validate_csv(frame,e.test)
    path=out/(STEM+'.csv');frame.to_csv(path,index=False);validate_csv(pd.read_csv(path),e.test)
    assert len(path.read_text().splitlines())==501
    save(out/'fresh_execution_checks.json',{'status':'PASS','inputs':e.inputs,'choice':choice,'fresh_v9_difference_INR':dv,'v7_portable_difference_INR':db,'maximum_final_difference_INR':difference,'candidate_behavior':behavior,'execution_contract':json.loads((ROOT/'resolved_models.json').read_text()),'csv_sha256':sha(path),'seconds':time.monotonic()-start,'v7_freshly_retrained_in_colab':False,'new_components_freshly_fitted':True,'portal_uploads':0})
    print('PASS: fresh V11 notebook reproduction within INR 0.01; maximum difference',difference,flush=True)

if __name__=='__main__':main()
