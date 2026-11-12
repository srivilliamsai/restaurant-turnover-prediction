"""Three new V7 random controls, strictly on the verified original Mac platform."""
import json,platform,importlib.metadata,time
import numpy as np
import baseline
from core import ROOT,Engine,sha,digest,save,log

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    e=Engine(out_dir=ROOT/'mac_results');e.cache.stage='confirm'
    lock_path=ROOT/'colab_results/results/candidate_lock.json';lock=json.loads(lock_path.read_text());assert bool(lock['choices']) and lock['sources']==e.sources
    expected={'numpy':'2.5.4','pandas':'3.0.6','scikit-learn':'1.9.1','scipy':'1.18.1','catboost':'1.2.10','interpret-core':'0.7.8'}
    assert {n:importlib.metadata.version(n) for n in expected}==expected
    contract=json.loads((ROOT/'v5_contract.json').read_text());sources={n:sha(ROOT/n) for n in ['baseline.py','legacy_models.py','legacy_reference.py','v5_contract.json']}
    runtime={'system':platform.system(),'machine':platform.machine(),'python':platform.python_version(),'dependencies':expected}
    for scheme in [71501,71502,71503]:
        p=np.zeros(len(e.y))
        for f,(a,b) in enumerate(e.splits(scheme)):
            context={'scheme':scheme,'fold':f,'inputs':e.inputs,'sources':sources,'platform':runtime,'train_indices':digest(a.tolist()),'validation_indices':digest(b.tolist()),'target_hash':digest(e.y[a].tolist())}
            p[b]=e.cache.get(f'mac_v7_{scheme}_fold{f}',context,lambda a=a,b=b,f=f:baseline.fold_prediction(e.train.iloc[a],e.y[a],e.train.iloc[b],contract,f,log))
            log({'MAC_CONTROL':scheme,'fold_complete':f+1,'folds':5})
        path=e.out/f'reference_v7_{scheme}.npz';temp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(temp,prediction=p);temp.replace(path)
        save(path.with_suffix('.json'),{'sha256':sha(path),'inputs':e.inputs,'sources':sources,'platform':runtime,'fold_sha256':digest([b.tolist() for _,b in e.splits(scheme)]),'row_ids_sha256':e.row_hash,'lock_sha256':sha(lock_path)})
    save(e.out/'mac_compute_receipt.json',{'seconds':e.cache.state()['seconds'],'ledger_sha256':sha(e.cache.ledger),'lock_sha256':sha(lock_path),'purpose':'15 original-Mac outer control fits, charge once'})

if __name__=='__main__':main()
