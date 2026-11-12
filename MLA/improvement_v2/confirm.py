from experiment import *
from sklearn.model_selection import GroupKFold

locked=json.loads((ROOT/'results/locked_candidate.json').read_text())
names=['raw_baseline','huber_baseline']+list(locked['weights'])
train,test=load_data(); y=train[TARGET].to_numpy(float)
summaries=[]
for split_seed in locked['confirmation_split_seeds']:
    run(names,split_seed,confirmation=True)
    pp={n:np.load(ROOT/'results'/f'{n}_{split_seed}_fixed.npz')['oof'] for n in names}
    base=.55*pp['raw_baseline']+.45*pp['huber_baseline']
    new=sum(w*pp[n] for n,w in locked['weights'].items())
    np.savez_compressed(ROOT/'results'/f'comparison_{split_seed}.npz',baseline=base,candidate=new)
    summaries.append(dict(validation=f'Shuffled 5-fold seed {split_seed}',baseline_rmse=rmse(y,base),candidate_rmse=rmse(y,new),improvement=rmse(y,base)-rmse(y,new)))
    print('CONFIRMATION',summaries[-1],flush=True)

# Stress test: cities in each validation fold are absent from that fold's training.
scheme='city_canonical'
groups=features(train,'engineered')['City']
splits=list(GroupKFold(5).split(train,y,groups))
data={v:features(train,v) for v in {SPECS[n]['variant'] for n in names}}
pp={}
for name in names:
    file=ROOT/'results'/f'{name}_{scheme}.npz'
    if file.exists(): pp[name]=np.load(file)['oof']; continue
    spec=locked['configs'].get(name,SPECS[name]); X=data[spec['variant']]
    p=np.zeros(len(y))
    for fold,(tr,va) in enumerate(splits):
        p[va],_=fit_predict(X,y,tr,va,spec,43+fold,spec.get('iterations'))
    pp[name]=p; np.savez_compressed(file,oof=p)
    print('CITY',name,rmse(y,p),flush=True)
base=.55*pp['raw_baseline']+.45*pp['huber_baseline']
new=sum(w*pp[n] for n,w in locked['weights'].items())
np.savez_compressed(ROOT/'results/comparison_city_canonical.npz',baseline=base,candidate=new)
summaries.append(dict(validation='City GroupKFold stress test',baseline_rmse=rmse(y,base),candidate_rmse=rmse(y,new),improvement=rmse(y,base)-rmse(y,new)))
pd.DataFrame(summaries).to_csv(ROOT/'results/confirmation_summary.csv',index=False)
print(pd.DataFrame(summaries).to_string(index=False),flush=True)
