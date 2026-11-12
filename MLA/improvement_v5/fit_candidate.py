"""Fit the locked candidate and reproduce the portal-scored predecessor."""
from pipeline import *
import time

start=time.time()
contract=load_contract();train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
previous,candidate,fitted=full_fit(train,test,y,contract)
ledger=json.loads((ROOT/'portal_snapshot.json').read_text())
backup=Path(ledger['files']['csv']['path'])
assert hashlib.sha256(backup.read_bytes()).hexdigest()==ledger['files']['csv']['sha256']
old=pd.read_csv(backup);check_submission(old,test)
assert np.allclose(previous,old['Annual Turnover'],rtol=1e-12,atol=.01)
assert np.allclose(candidate,predict_final(test,fitted,contract),rtol=1e-12,atol=.01)
np.savez_compressed(OUT/'full_prediction_parts.npz',previous=previous,candidate=candidate)
frame=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':candidate})
check_submission(frame,test)
path=ROOT/'submission.csv';frame.to_csv(path,index=False)
result={'elapsed_seconds':time.time()-start,'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'confirmed_v3_maximum_reproduction_difference':float(np.max(np.abs(previous-old['Annual Turnover']))),
        'mean_absolute_prediction_change':float(np.mean(np.abs(candidate-previous))),
        'prediction_min':float(candidate.min()),'prediction_max':float(candidate.max()),'candidate_portal_rmse':None}
(OUT/'full_fit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
