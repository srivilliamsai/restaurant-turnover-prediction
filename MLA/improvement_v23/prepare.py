"""Reconstruct original controls before V23 fitting; no estimator is trained."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'improvement_v22'))
from core import Engine,sha,digest
import legacy_models

def main():
    e=Engine(out=ROOT/'audit_scratch')
    data={'y':e.y}; partitions={}; metrics={}; witnesses={}
    for scheme in [42,71,48211,48212,48213,'strat:27421','city']:
        key=str(scheme).replace(':','_')
        parts=e.reference_parts(scheme)
        assert np.max(abs(parts['v9']-.5*parts['v7']-.5*parts['t9']))<.01
        data['v14_'+key]=parts['v14']
        partitions[str(scheme)]=[[a.tolist(),b.tolist()] for a,b in e.splits(scheme)]
        metrics[str(scheme)]=float(np.sqrt(np.mean((e.y-parts['v14'])**2)))
        for fold in range(5):
            p=e.root if hasattr(e,'root') else ROOT.parent/'improvement_v22'
            p=p/'provenance/v14_cache'/f'P35_8_{key}_fold{fold}.npz'
            witnesses[str(p.relative_to(ROOT.parent))]={'prediction_sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json'))}
    train=legacy_models.features(e.train);test=legacy_models.features(e.test)
    assert train.shape==(3493,40) and test.shape==(500,40)
    for name,frame in [('feature_train',train),('feature_test',test)]:frame.to_csv(ROOT/(name+'.csv'),index=False)
    np.savez_compressed(ROOT/'controls.npz',**data)
    (ROOT/'partitions.json').write_text(json.dumps(partitions)+'\n')
    portal=json.loads((ROOT.parent/'improvement_v22/reference/v14_portal_trial.json').read_text())
    import shutil
    fallback=ROOT/'delivery';fallback.mkdir(exist_ok=True)
    for ext,k in [('csv','csv_sha256'),('ipynb','notebook_sha256')]:
        p=ROOT.parent/f'improvement_v22/reference/scored_v14.{ext}'
        assert sha(p)==portal[k]
        shutil.copyfile(p,fallback/f'KEEP_V14_submission.{ext}')
    sources={name:sha(ROOT.parent/'improvement_v22'/name) for name in ['core.py','legacy_models.py','legacy_reference.py','baseline_registry.json','reference_manifest.json','city_map.json']}
    manifest={'inputs':e.inputs,'row_ids_sha256':e.row_hash,'all_labels_retained':True,'original_P35_contexts_verified':35,'witnesses':witnesses,'source_sha256':sources,'control_RMSE_INR':metrics,'V22_confirmation':json.loads((ROOT.parent/'improvement_v22_rank_check/colab_results/results/decision.json').read_text()),'files_sha256':{n:sha(ROOT/n) for n in ['controls.npz','partitions.json','feature_train.csv','feature_test.csv']},'reference_sha256':{'csv':portal['csv_sha256'],'ipynb':portal['notebook_sha256']},'measured_epoch':time.time()}
    (ROOT/'history_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'audit':'PASS','P35_control_contexts':35,'V14_RMSE':metrics,'all_labels':3493}))
if __name__=='__main__':main()
