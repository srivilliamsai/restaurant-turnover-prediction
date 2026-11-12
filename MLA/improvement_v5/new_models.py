"""New training hypotheses; untouched original-scale validation targets."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE/'improvement_v2'))
from experiment import features,load_data,rmse

def v5_features(raw,variant='base'):
    X=features(raw,'engineered' if variant=='rich' else 'base')
    if variant in ['clean','rich']:
        X['City']=X['City'].replace({'bengaluru':'bangalore','banglore':'bangalore','gurugram':'gurgaon','gurgoan':'gurgaon','bombay':'mumbai','calcutta':'kolkata','delhi':'new delhi','bhubneshwar':'bhubaneswar','hderabad':'hyderabad','punr':'pune'})
    if variant=='rich':
        fb=X['Facebook Popularity Quotient'];ig=X['Instagram Popularity Quotient']
        X['Social ratio']=fb/(ig+1)
        X['Social product']=fb*ig
        X['Social minimum']=np.minimum(fb,ig)
        X['Social absolute gap']=np.abs(fb-ig)
        X['Quality spread']=X[['Food Rating','Hygiene Rating','Overall Restaurant Rating']].std(axis=1)
        X['Zomato x Hygiene']=X['Restaurant Zomato Rating']*X['Hygiene Rating']
        X['City tier x Restaurant tier']=X['Restaurant City Tier']*X['Resturant Tier']
    return X

def inverse_target(z,power):
    if power==0:return np.expm1(np.minimum(z,20))
    return np.maximum(z,0)**(1/power)

def fit_v5(raw,y,spec,seed):
    X=v5_features(raw,spec.get('variant','base'));target=np.asarray(y,dtype=float)/1e6
    if 'cap' in spec:target=np.minimum(target,spec['cap'])
    power=spec.get('power',1.)
    transformed=np.log1p(target) if power==0 else target**power
    loss=spec.get('loss','RMSE')
    params=dict(iterations=spec.get('iterations',850),learning_rate=.04,depth=spec.get('depth',5),
                l2_leaf_reg=10,bootstrap_type='Bernoulli',subsample=.7,loss_function=loss,
                random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    if 'rsm' in spec:params['rsm']=spec['rsm']
    if 'topk' in spec:
        pilot=CatBoostRegressor(**{**params,'iterations':300})
        pilot.fit(X,transformed,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
        ranked=np.argsort(-pilot.feature_importances_);selected=[X.columns[i] for i in ranked[:spec['topk']]]
        X=X[selected]
    model=CatBoostRegressor(**params)
    model.fit(X,transformed,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
    correction=0.
    if spec.get('correct_mean'):
        # Training-only retransformation correction; all validation labels remain unseen.
        fitted_original=inverse_target(model.predict(X),power)
        correction=float(np.mean(target-fitted_original))
    return {'model':model,'columns':X.columns.tolist(),'power':power,'correction':correction,'spec':spec}

def predict_v5(fitted,raw):
    X=v5_features(raw,fitted['spec'].get('variant','base'))[fitted['columns']]
    return np.maximum(0,(inverse_target(fitted['model'].predict(X),fitted['power'])+fitted['correction'])*1e6)

SPECS={}
for depth in [4,6]:
    for correction in [False,True]:
        SPECS[f'sqrt_d{depth}_correct{int(correction)}']=dict(power=.5,depth=depth,correct_mean=correction)
for correction in [False,True]:SPECS[f'power075_correct{int(correction)}']=dict(power=.75,depth=5,correct_mean=correction)
for depth in [4,6]:SPECS[f'log_d{depth}']=dict(power=0,depth=depth,correct_mean=True)
for alpha in [.5,.6,.7]:SPECS[f'quantile{alpha}']=dict(loss=f'Quantile:alpha={alpha}',depth=5)
for cap in [60,75,90]:SPECS[f'cap{cap}']=dict(cap=cap,depth=5)
for delta in [20,40]:
    for topk in [15,25]:SPECS[f'huber{delta}_top{topk}']=dict(loss=f'Huber:delta={delta}',topk=topk)
SPECS['clean_h40']=dict(variant='clean',loss='Huber:delta=40')
SPECS['rich_h40']=dict(variant='rich',loss='Huber:delta=40')
SPECS['rich_h20']=dict(variant='rich',loss='Huber:delta=20')
SPECS['subspace_h40']=dict(loss='Huber:delta=40',rsm=.5,iterations=1200)

def run_oof(train,y,name,scheme,seeds=(10242,20242)):
    spec=SPECS[name];file=ROOT/'results'/f'{name}_{scheme}.npz'
    context={'spec':spec,'scheme':scheme,'seeds':seeds,'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data':hashlib.sha256(Path('~/Downloads/Train_dataset.csv').read_bytes()).hexdigest()}
    digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    if file.exists():
        saved=np.load(file);assert str(saved['context'])==digest
        return saved['oof']
    individual=np.zeros((len(y),len(seeds)))
    for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=scheme).split(train)):
        for j,seed in enumerate(seeds):
            fitted=fit_v5(train.iloc[tr],y[tr],spec,seed+f)
            individual[va,j]=predict_v5(fitted,train.iloc[va])
    p=individual.mean(axis=1);np.savez_compressed(file,oof=p,individual=individual,context=digest)
    return p

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scheme',type=int,default=42);parser.add_argument('--names',nargs='*',default=list(SPECS));args=parser.parse_args()
    (ROOT/'results/specs.json').write_text(json.dumps(SPECS,indent=2))
    train,test=load_data();y=train['Annual Turnover'].to_numpy(float)
    if args.scheme==42:
        b=.5*np.load(BASE/'improvement_v3/results/v2_42.npz')['oof']+.5*np.load(BASE/'improvement_v3/results/bag_subsample_h40_42.npz')['oof']
    else:b=np.load(BASE/'improvement_v3/results'/f'comparison_{args.scheme}.npz')['candidate']
    rows=[];print('BASELINE',rmse(y,b),flush=True)
    for name in args.names:
        start=time.time();p=run_oof(train,y,name,args.scheme)
        row={'name':name,'scheme':args.scheme,'rmse':rmse(y,p),'blend25':rmse(y,.75*b+.25*p),'blend50':rmse(y,.5*b+.5*p),'seconds':time.time()-start}
        rows.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(ROOT/'results'/f'screen_{args.scheme}.csv',index=False)
