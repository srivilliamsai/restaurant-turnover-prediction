"""One-time audited adoption of the first run launched before cache guards existed.

Only run after that initial process finishes. This does not select models or train.
"""
from pathlib import Path
import argparse,ast,json,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=ROOT.parent
sys.path.insert(0,str(ROOT))
from pipeline import (load_data,load_contract,file_sha256,json_sha256,
                      current_input_hashes,validation_cache_context,checked_prior_v3,rmse)

def functions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirm-first-run-finished',action='store_true',required=True)
    parser.parse_args()
    initial=json.loads((ROOT/'results/initial_run_provenance.json').read_text())
    evidence=json.loads((HERE/'initial_cache_guard_audit.json').read_text())
    for file,digest in evidence['files'].items():assert file_sha256(BASE/file)==digest,'Source/config changed: '+file
    assert current_input_hashes()==evidence['inputs'],'Data changed during initial run'
    for name in ['pipeline.py','confirm_candidate.py']:
        assert file_sha256(HERE/('initial_'+name+'.txt'))==initial['hashes']['improvement_v4/'+name]
    contract_path=ROOT/'results/candidate_lock.json'
    assert file_sha256(contract_path)==initial['hashes']['improvement_v4/results/candidate_lock.json'],'Frozen candidate changed'
    old=(HERE/'initial_pipeline.py.txt').read_text();new=(ROOT/'pipeline.py').read_text()
    old=old.replace("v3=np.load(previous)['candidate']",'v3=checked_prior_v3(train,y,scheme,contract)')
    a,b=functions(old),functions(new)
    for name,node in a.items():assert b[name]==node,'Non-guard model code changed: '+name
    old_confirm=(HERE/'initial_confirm_candidate.py.txt').read_text()
    expected=old_confirm.replace('p=dict(np.load(file))','p=read_validation_cache(file,train,y,scheme,contract)').replace('np.savez_compressed(file,**p)','write_validation_cache(file,p,train,y,scheme,contract)')
    assert ast.dump(ast.parse(expected),include_attributes=False)==ast.dump(ast.parse((ROOT/'confirm_candidate.py').read_text()),include_attributes=False),'Confirmation changed beyond cache guards'
    contract=load_contract();train,_=load_data();y=train['Annual Turnover'].to_numpy(float)
    table=pd.read_csv(ROOT/'results/confirmation.csv',dtype={'validation':str})
    schemes=['1701','1711','71','2026','303','909','city']
    assert table['validation'].tolist()==schemes,'Initial confirmation run is incomplete'
    adopted=[]
    for scheme in schemes:
        path=ROOT/'results'/f'comparison_{scheme}.npz';meta_path=path.with_suffix('.meta.json')
        assert path.exists() and not meta_path.exists(),'Missing cache or already stamped: '+str(path)
        with np.load(path,allow_pickle=False) as data:p={k:data[k].copy() for k in data.files}
        assert set(p)=={'previous','candidate'}
        assert all(a.shape==y.shape and np.isfinite(a).all() and (a>=0).all() for a in p.values())
        row=table.loc[table['validation']==scheme].iloc[0]
        assert np.isclose(rmse(y,p['previous']),row.v3_rmse,rtol=0,atol=1e-6)
        assert np.isclose(rmse(y,p['candidate']),row.v4_rmse,rtol=0,atol=1e-6)
        if scheme in ['71','2026','303','909','city']:
            assert np.array_equal(p['previous'],checked_prior_v3(train,y,scheme,contract))
        metadata={'context':validation_cache_context(train,y,scheme,contract),
                  'file_sha256':file_sha256(path),'origin':'Audited first run, launched before guard-only patch',
                  'initial_provenance_sha256':file_sha256(ROOT/'results/initial_run_provenance.json'),
                  'audit_evidence_sha256':file_sha256(HERE/'initial_cache_guard_audit.json'),
                  'proof':'Input and frozen contract hashes unchanged; original existing pipeline function ASTs identical except verified prior-cache read; confirm script differs only in cache read/write guards; existing-v3 arrays verified against its manifest.'}
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n');adopted.append(path.name)
    print(json.dumps({'adopted':adopted,'parameters_changed':False},indent=2))

if __name__=='__main__':main()
