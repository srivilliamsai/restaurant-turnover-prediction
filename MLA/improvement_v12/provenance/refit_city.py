"""Refit frozen V7 on the revised city folds on its original Mac platform only."""
import json
import platform
from pathlib import Path
import numpy as np
import baseline
from core import ROOT,Engine,sha,digest,save,log


def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    e=Engine();e.cache.stage='confirm'
    lock=json.loads((ROOT/'colab_results/results/candidate_lock.json').read_text())
    assert lock['complete'] and lock['choice'] is not None, 'No frozen finalist needs a city baseline.'
    contract=json.loads((ROOT/'v5_contract.json').read_text())
    sources={n:sha(ROOT/n) for n in ['baseline.py','legacy_models.py','legacy_reference.py','v5_contract.json']}
    runtime={'system':platform.system(),'machine':platform.machine(),'python':platform.python_version()}
    p=np.zeros(len(e.y))
    for f,(a,b) in enumerate(e.splits('city')):
        context={'inputs':e.inputs,'sources':sources,'platform':runtime,'fold':f,
                 'train_indices':digest(a.tolist()),'valid_indices':digest(b.tolist()),
                 'contract_sha256':sha(ROOT/'v5_contract.json'),'targets':digest(e.y[a].tolist())}
        p[b]=e.cache.get(f'original_mac_v7_city_fold{f}',context,
            lambda a=a,b=b,f=f:baseline.fold_prediction(e.train.iloc[a],e.y[a],e.train.iloc[b],contract,f,log))
        log({'MAC_CITY_BASELINE_FOLD':f+1,'of':5})
    destination=e.out/'reference_city_v7.npz';np.savez_compressed(destination,prediction=p)
    save(destination.with_suffix('.json'),{'inputs':e.inputs,'sources':sources,'platform':runtime,
        'row_ids_sha256':e.row_hash,'fold_sha256':digest([b.tolist() for _,b in e.splits('city')]),
        'sha256':sha(destination),'lock_sha256':sha(ROOT/'colab_results/results/candidate_lock.json')})
    log('PASS: revised-city V7 comparison fitted on original Mac platform.')


if __name__=='__main__':main()
