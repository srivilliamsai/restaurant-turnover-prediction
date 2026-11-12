from experiment import *
from scipy.optimize import minimize

train,test=load_data(); y=train[TARGET].to_numpy(float)
scorefile=ROOT/'results/scores_42.csv'
names=pd.read_csv(scorefile)['model'].tolist()
p={k:np.load(ROOT/'results'/f'{k}_42.npz')['oof'] for k in names}
p['current_blend']=.55*p['raw_baseline']+.45*p['huber_baseline']
names=list(p)
base=p['current_blend']
rows=[]
for k in names:
    if k=='current_blend': continue
    for w in [.25,.5,.75,1]:
        pred=w*p[k]+(1-w)*base
        rows.append(dict(candidate=k,weight=w,rmse=rmse(y,pred),body=rmse(y[y<1e8],pred[y<1e8])))
print('Blends with current:\n',pd.DataFrame(rows).sort_values('rmse').head(20).to_string(index=False))
P=np.column_stack(list(p.values()))/1e6; yy=y/1e6
fit=minimize(lambda w:np.mean((P@w-yy)**2)+2*np.sum(w*w),np.ones(len(names))/len(names),bounds=[(0,1)]*len(names),constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],method='SLSQP')
print('Regularized stack weights',dict(zip(names,np.round(fit.x,3))))
print('Selection RMSE',rmse(y,(P@fit.x)*1e6))
print('Target correlations:')
print(train.select_dtypes('number').corrwith(train[TARGET]).sort_values().to_string())
