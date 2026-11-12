from experiment import *
from scipy.optimize import minimize
import hashlib, sys

train,_=load_data(); y=train[TARGET].to_numpy(float)
names=pd.read_csv(ROOT/'results/scores_42.csv')['model'].tolist()
matrix=np.column_stack([np.load(ROOT/'results'/f'{k}_42.npz')['oof'] for k in names])/1e6
result=minimize(lambda w:np.mean((matrix@w-y/1e6)**2)+2*np.sum(w*w),np.ones(len(names))/len(names),bounds=[(0,1)]*len(names),constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],method='SLSQP',options={'maxiter':1000,'ftol':1e-10})
assert result.success,result.message
weights=np.where(result.x>=.05,result.x,0); weights/=weights.sum()
selected={name:float(w) for name,w in zip(names,weights) if w>0}
configs={}
for name in selected:
    config=SPECS[name].copy()
    if config['kind'] in ['cat','lgb','xgb']: config['iterations']=int(np.median(np.load(ROOT/'results'/f'{name}_42.npz')['trees']))
    configs[name]=config
locked=dict(weights=selected,configs=configs,confirmation_split_seeds=[71,2026,303],development_rmse=rmse(y,(matrix@weights)*1e6),selection='Nonnegative weights summing to 1; objective = MSE in millions + 2 * sum(weights squared); discard weights < 0.05 and renormalize. Weights use development OOF seed 42 only. Seeds 71 and 2026 were inspected for the earlier 30-model ensemble; seed 303 was reserved until after the expanded 38-model ensemble was frozen. All CV uses the same labeled data and is not an independent test set.',model_seed_schedule='42 + fold_number (one based)',final_seed=42)
(ROOT/'results/locked_candidate.json').write_text(json.dumps(locked,indent=2))
print(json.dumps(locked,indent=2))
baseline_file=ROOT/'rank3_backup/submission.csv'
audit={'rank3_csv_sha256':hashlib.sha256(baseline_file.read_bytes()).hexdigest(),'reported_public_rmse':12204615.6438253,'reported_rank':3,'reference_leader_rmse':11946659.0702364,'source':'User screenshots from 2026-10-11; hidden labels unavailable','python':sys.version,'files':{}}
for name in ['Train_dataset.csv','Test_dataset.csv','Sample_Solution.csv','Data_Dictionary.csv']:
    audit['files'][name]=hashlib.sha256((Path('~/Downloads')/name).read_bytes()).hexdigest()
(ROOT/'results/input_audit.json').write_text(json.dumps(audit,indent=2))
