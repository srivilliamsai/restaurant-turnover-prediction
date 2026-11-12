from experiments import *

lock=json.loads((OUT/'candidate_lock.json').read_text())
train,test=load_data(); y=train['Annual Turnover'].to_numpy(float)
rows=[]
for seed in [71,2026,303,909,'city']:
    base=v2_oof(train,y,seed)
    p=lock['v2_weight']*base
    for name,w in lock['additional_weights'].items():
        p+=w*run_trial(train,y,name,seed)
    if str(seed) in ['71','2026','303','city']:
        suffix='city_canonical' if seed=='city' else str(seed)
        old=np.load(BASE/'improvement_v2/results'/f'comparison_{suffix}.npz')['baseline']
    else:
        old=np.zeros(len(y));X=features(train,'base')
        for weight,spec in [(0.55,dict(kind='cat',variant='base',depth=6,iterations=302,loss='RMSE')),
                            (0.45,dict(kind='cat',variant='base',depth=6,iterations=768,loss='Huber:delta=12000000'))]:
            component=np.zeros(len(y))
            for f,(tr,va) in enumerate(split_indices(train,seed)):
                component[va],_=fit_predict(X,y,tr,va,spec,43+f,spec['iterations'])
            old+=weight*component
    np.savez_compressed(OUT/f'comparison_{seed}.npz',old=old,v2=base,candidate=p)
    row=dict(validation=str(seed),old_rmse=rmse(y,old),v2_rmse=rmse(y,base),candidate_rmse=rmse(y,p),gain_from_v2=rmse(y,base)-rmse(y,p))
    rows.append(row); print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'confirmation.csv',index=False)

table=pd.DataFrame(rows)
random=table.iloc[:4]
print('MEAN RANDOM',random.drop(columns='validation').mean().to_dict(),flush=True)
