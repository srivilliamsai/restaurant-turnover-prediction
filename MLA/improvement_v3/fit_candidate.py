from experiments import *
import hashlib

STEM='submission'
lock=json.loads((OUT/'candidate_lock.json').read_text())
train,test=load_data(); y=train['Annual Turnover'].to_numpy(float)
v2_csv=BASE/'improvement_v2'/(STEM+'.csv')
v2=pd.read_csv(v2_csv)
assert hashlib.sha256(v2_csv.read_bytes()).hexdigest()=='af82750a880dc2856b94c5d1846c039df69eac00be48f45ea2a873ec9ea0f723'
assert v2['Registration Number'].equals(test['Registration Number'])
prediction=lock['v2_weight']*v2['Annual Turnover'].to_numpy()
parts={}
for name,w in lock['additional_weights'].items():
    spec=lock['specs'][name];X=features(train,spec['variant']);T=features(test,spec['variant'])
    pp=[]
    for seed in lock['model_seeds']:
        model=cat_fit(X,y,spec,seed)
        pred=np.maximum(model.predict(T),0)
        perm=np.random.default_rng(91).permutation(len(test))
        assert np.allclose(model.predict(T.iloc[perm]),pred[perm])
        edge=test.iloc[:2].copy();edge['City']='unseen_validation_city';edge['Cuisine']=pd.NA;edge['Facebook Popularity Quotient']=np.nan
        assert np.isfinite(model.predict(features(edge,spec['variant']))).all()
        pp.append(pred)
    parts[name]=np.mean(pp,axis=0);prediction+=w*parts[name]
submission=pd.DataFrame({'Registration Number':test['Registration Number'],'Annual Turnover':prediction})
assert submission.shape==(500,2) and np.isfinite(prediction).all() and (prediction>=0).all()
path=BASE/'improvement_v3'/(STEM+'.csv')
submission.to_csv(path,index=False)
np.savez_compressed(OUT/'full_prediction_parts.npz',v2=v2['Annual Turnover'].to_numpy(),**parts,candidate=prediction)
print('Candidate (not yet promoted):',path)
print('Mean absolute change:',float(np.abs(prediction-v2['Annual Turnover']).mean()))
print('SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
