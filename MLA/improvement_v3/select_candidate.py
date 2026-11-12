"""Select a simple addition on development folds; reserve seed909 until locked."""
from experiments import *

train,test=load_data(); y=train['Annual Turnover'].to_numpy(float)
base=v2_oof(train,y,42)
predictions={name:np.load(OUT/f'{name}_42.npz')['oof'] for name in TRIALS}
mixtures={name:{name:1.0} for name in predictions}
# Equal-weight seed-averaged families, defined independently of their scores.
families={
    'three_depths':['bag_h40_d4','bag_h40_d5','bag_h40_d6'],
    'three_features':['bag_h40_d5','bag_enhanced_h40','bag_drop_h40'],
    'robust_diverse':['bag_h40_d6','bag_mono_h40','bag_ordered_h40'],
    'all_bagged':list(TRIALS),
}
for name,parts in families.items():
    mixtures[name]={part:1/len(parts) for part in parts}
    predictions[name]=sum(w*predictions[part] for part,w in mixtures[name].items())
rows=[]
for name,pred in predictions.items():
    for weight in [.25,.5,.75,1.0]:
        mix=(1-weight)*base+weight*pred
        folds=[rmse(y[va],base[va])-rmse(y[va],mix[va]) for tr,va in split_indices(train,42)]
        rows.append(dict(name=name,new_weight=weight,rmse=rmse(y,mix),gain=rmse(y,base)-rmse(y,mix),fold_wins=sum(v>0 for v in folds)))
table=pd.DataFrame(rows).sort_values('rmse')
table.to_csv(OUT/'candidate_selection.csv',index=False)
print('BASELINE',rmse(y,base));print(table.head(15).to_string(index=False))
best=table.iloc[0]
new_weights={n:float(w*best.new_weight) for n,w in mixtures[best['name']].items()}
lock=dict(v2_weight=float(1-best.new_weight),additional_weights=new_weights,
          specs={n:TRIALS[n] for n in new_weights},model_seeds=[1042,2042,3042],
          development=best.to_dict(),baseline_development_rmse=rmse(y,base),
          development_split_seed=42,confirmation_split_seeds=[71,2026,303,909],
          selection='Fixed configurations; seed42 development only. Coarse 25% blending grid. Seed909 unseen during this round selection, but all labeled rows were involved in previous development. No independent external test is claimed.')
(OUT/'candidate_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
