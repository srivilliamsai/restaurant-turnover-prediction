from pipeline import *
import hashlib
contract=load_contract();train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
previous,candidate,fitted=full_fit(train,test,y,contract)
actual=pd.read_csv(ROOT/'rank3_confirmed_backup/submission.csv')['Annual Turnover'].to_numpy()
assert np.allclose(actual,previous,atol=.01,rtol=1e-12),'V3 reconstruction mismatch'
assert np.allclose(candidate,predict_final(test,fitted,contract),atol=.01,rtol=1e-12)
np.savez_compressed(OUT/'full_prediction_parts.npz',previous=previous,candidate=candidate)
submission=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':candidate})
check_submission(submission,test)
file=ROOT/'submission.csv';submission.to_csv(file,index=False)
result={'csv_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'v3_reconstruction_max_diff':float(abs(actual-previous).max()),'mean_absolute_prediction_change':float(abs(candidate-previous).mean()),'prediction_min':float(candidate.min()),'prediction_max':float(candidate.max()),'portal_rmse':None}
(OUT/'full_fit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
